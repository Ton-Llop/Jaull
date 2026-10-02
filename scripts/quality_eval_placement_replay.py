"""Replay a completed bundle's scoring requests under several offload splits.

The pilot's reuse design rests on one untested assumption: that a score is a
property of ``(artifact, protocol)`` alone. ``pilot/quality_eval/compare.py``
encodes it by keeping hardware in provenance rather than in identity. Nothing
has measured it. Floating-point reduction is not associative, so a CUDA kernel
and a CPU kernel can disagree in the last bits, and a partial offload runs both
in the same forward pass.

This replays the exact requests captured in ``http.jsonl``, so the harness, the
dataset, the prompts and the forced target tokens are held fixed by
construction and the only variable left is where the layers ran. It is a
determinism probe, not a quality evaluation: three HellaSwag examples decide
nothing about a model, and this script never writes a quality record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from pilot.quality_eval.evaluate import (
    SERVER_SHA256,
    TASK,
    validate_artifact_identity,
    validate_scoring_request,
    validate_scoring_response,
    write_json,
)
from scripts.quality_eval_smoke import check_server_port, server_command, wait_for_server

PORT = 18083
BASELINE_ARM = "gpu_full_a"


def arms(partial_gpu_layers: int) -> list[dict[str, Any]]:
    """The pilot's own launch, a repeat of it, and three deliberate departures."""
    return [
        {"name": BASELINE_ARM, "n_gpu_layers": -1, "device": "CUDA0", "threads": 4},
        # Identical flags, fresh process: separates a score moved by placement
        # from a score moved by any relaunch at all.
        {"name": "gpu_full_b", "n_gpu_layers": -1, "device": "CUDA0", "threads": 4},
        # The interesting middle: one forward pass split across two backends.
        {"name": "gpu_partial", "n_gpu_layers": partial_gpu_layers, "device": "CUDA0",
         "threads": 4},
        {"name": "cpu_only", "n_gpu_layers": 0, "device": None, "threads": 4},
        # Thread count changes CPU reduction order the same way placement does.
        {"name": "cpu_threads8", "n_gpu_layers": 0, "device": None, "threads": 8},
    ]


def captured_scoring_requests(bundle: Path) -> tuple[list[dict[str, Any]], list[float]]:
    """Every /v1/completions exchange of a successful bundle, in capture order."""
    if list(bundle.glob("*-error.json")):
        raise ValueError("A failed bundle is not a baseline")
    payloads: list[dict[str, Any]] = []
    scores: list[float] = []
    for line in (bundle / "http.jsonl").read_text(encoding="utf-8").splitlines():
        exchange = json.loads(line)
        if "error" in exchange or "response" not in exchange:
            raise ValueError("Failed HTTP evidence")
        if not exchange["url"].endswith("/v1/completions"):
            continue
        payload, response = exchange["request"], exchange["response"]
        validate_scoring_request(payload)
        validate_scoring_response(payload, response)
        payloads.append(payload)
        scores.append(response["choices"][0]["logprobs"]["content"][0]["logprob"])
    if not payloads:
        raise ValueError("Bundle captured no scoring requests")
    return payloads, scores


def continuation_groups(payloads: list[dict[str, Any]]) -> list[list[int]]:
    """One request scores one token, so a continuation is a run of +1 prompt lengths."""
    groups: list[list[int]] = []
    current = [0]
    for index in range(1, len(payloads)):
        if len(payloads[index]["prompt"]) == len(payloads[index - 1]["prompt"]) + 1:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)
    return groups


def loglikelihoods(scores: list[float], groups: list[list[int]]) -> list[float]:
    return [sum(scores[index] for index in group) for group in groups]


def metrics(sums: list[float], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """HellaSwag's two metrics: raw argmax, and argmax over character-normalised.

    Reimplemented here rather than re-run, because the harness is exactly the
    variable this experiment holds fixed. ``verify_baseline`` checks the
    reimplementation against the numbers the harness itself wrote.
    """
    consumed = 0
    per_sample = []
    for sample in samples:
        pairs = sample["arguments"]
        group = sums[consumed : consumed + len(pairs)]
        consumed += len(pairs)
        gold = int(sample["target"])
        normalised = [score / len(pairs[index][1]) for index, score in enumerate(group)]
        # max() keeps the first maximum, which is what numpy argmax does.
        raw = max(range(len(group)), key=lambda index: group[index])
        norm = max(range(len(normalised)), key=lambda index: normalised[index])
        per_sample.append({
            "doc_id": sample["doc_id"], "gold": gold,
            "argmax_raw": raw, "argmax_norm": norm,
            "acc": int(raw == gold), "acc_norm": int(norm == gold),
            # How far the chosen answer won by. This, not the spread of the raw
            # scores, is what a placement change has to cross to flip an answer:
            # the candidates of one question share a prompt and a model, so their
            # perturbations are correlated and largely cancel in the difference.
            "margin_raw": group[raw] - max(
                score for index, score in enumerate(group) if index != raw
            ),
            "margin_norm": normalised[norm] - max(
                score for index, score in enumerate(normalised) if index != norm
            ),
            "loglikelihoods": group,
        })
    if consumed != len(sums):
        raise ValueError("Captured continuations do not match the recorded samples")
    return {
        "acc": sum(entry["acc"] for entry in per_sample) / len(per_sample),
        "acc_norm": sum(entry["acc_norm"] for entry in per_sample) / len(per_sample),
        "per_sample": per_sample,
    }


def verify_baseline(
    bundle_scores: list[float], groups: list[list[int]], samples: list[dict[str, Any]],
    reported: dict[str, Any],
) -> dict[str, Any]:
    """Prove the grouping and the metric code reproduce what the harness wrote."""
    sums = loglikelihoods(bundle_scores, groups)
    recorded = [score for sample in samples for score, _ in sample["filtered_resps"]]
    if len(sums) != len(recorded) or any(
        not math.isclose(ours, theirs, rel_tol=0, abs_tol=1e-9)
        for ours, theirs in zip(sums, recorded, strict=True)
    ):
        raise ValueError("Captured requests do not reconstruct the bundle continuation scores")
    computed = metrics(sums, samples)
    for metric in ("acc", "acc_norm"):
        if not math.isclose(
            computed[metric], reported[metric + ",none"], rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"Recomputed {metric} differs from the harness result")
    return computed


def replay(payloads: list[dict[str, Any]], log: Any) -> list[float]:
    scores = []
    for payload in payloads:
        validate_scoring_request(payload)
        request = Request(
            f"http://127.0.0.1:{PORT}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            answer = json.load(response)
        validate_scoring_response(payload, answer)
        log.write(json.dumps({"request": payload, "response": answer}) + "\n")
        scores.append(answer["choices"][0]["logprobs"]["content"][0]["logprob"])
    return scores


def deltas(left: list[float], right: list[float]) -> dict[str, Any]:
    diffs = [abs(one - two) for one, two in zip(left, right, strict=True)]
    return {
        "identical": all(diff == 0.0 for diff in diffs),
        "differing_tokens": sum(diff > 0.0 for diff in diffs),
        "tokens": len(diffs),
        "max_abs_delta": max(diffs),
    }


def answer_set(per_sample: list[dict[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    """The answers themselves, so a flip cannot hide inside an unchanged aggregate."""
    return tuple(
        (entry["doc_id"], entry["argmax_raw"], entry["argmax_norm"]) for entry in per_sample
    )


def margin_shift(baseline: list[dict[str, Any]], arm: list[dict[str, Any]]) -> dict[str, Any]:
    """How far the winning answers' leads moved, and how little of one is left."""
    keys = ("margin_raw", "margin_norm")
    shifts = {
        key: [abs(one[key] - two[key]) for one, two in zip(baseline, arm, strict=True)]
        for key in keys
    }
    return {
        "max_abs_shift": {key: max(values) for key, values in shifts.items()},
        "min_margin": {key: min(entry[key] for entry in arm) for key in keys},
        "per_sample": [
            {"doc_id": entry["doc_id"]} | {key: entry[key] for key in keys} for entry in arm
        ],
    }


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_arm(
    arm: dict[str, Any], binary: Path, model: Path, payloads: list[dict[str, Any]], output: Path,
) -> list[float]:
    directory = output / "arms" / arm["name"]
    directory.mkdir(parents=True, exist_ok=False)
    launch = server_command(
        binary, model,
        n_gpu_layers=arm["n_gpu_layers"], device=arm["device"], threads=arm["threads"],
    )
    write_json(directory / "command.json", {"arm": arm, "server": launch})
    check_server_port(PORT)
    server = None
    try:
        with (
            (directory / "server.log").open("x") as server_log,
            (directory / "replay.jsonl").open("x", encoding="utf-8") as log,
        ):
            server = subprocess.Popen(launch, stdout=server_log, stderr=subprocess.STDOUT)
            wait_for_server(server, PORT)
            return replay(payloads, log)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle.resolve()
    artifact = json.loads((bundle / "verified-artifact.json").read_text(encoding="utf-8"))
    validate_artifact_identity(artifact)
    model = Path(artifact["local_path"])
    binary = args.llama_server.resolve()
    if file_sha256(binary) != SERVER_SHA256:
        raise ValueError("llama-server binary differs from the audited pin")
    if file_sha256(model) != artifact["sha256"]:
        raise ValueError("Local artifact differs from the verified GGUF of the bundle")
    recorded = json.loads((bundle / "commands.json").read_text(encoding="utf-8"))["server"]
    if recorded != server_command(Path(recorded[0]), model):
        raise ValueError("Bundle was not launched with the default pilot placement")

    payloads, bundle_scores = captured_scoring_requests(bundle)
    results = json.loads((bundle / "lm-eval-results.json").read_text(encoding="utf-8"))
    samples = results["samples"][TASK]
    groups = continuation_groups(payloads)
    baseline = verify_baseline(bundle_scores, groups, samples, results["results"][TASK])

    measured: dict[str, list[float]] = {}
    for arm in arms(args.partial_gpu_layers):
        measured[arm["name"]] = run_arm(arm, binary, model, payloads, output)
    if file_sha256(model) != artifact["sha256"]:
        raise ValueError("Artifact changed during the replay")

    per_arm = {
        name: {
            "vs_bundle": deltas(scores, bundle_scores),
            "vs_baseline_arm": deltas(scores, measured[BASELINE_ARM]),
            "metrics": metrics(loglikelihoods(scores, groups), samples),
        }
        for name, scores in measured.items()
    }
    for arm in per_arm.values():
        arm["margin_shift_vs_baseline"] = margin_shift(
            baseline["per_sample"], arm["metrics"]["per_sample"]
        )
    identical = all(arm["vs_bundle"]["identical"] for arm in per_arm.values())
    # Per answer, never the aggregate: two samples can flip in opposite
    # directions and leave acc and acc_norm untouched.
    answers = {answer_set(baseline["per_sample"])} | {
        answer_set(arm["metrics"]["per_sample"]) for arm in per_arm.values()
    }
    write_json(output / "placement-replay.json", {
        "schema_version": 2,
        "purpose": "determinism_probe",
        "quality_evidence": False,
        "verdict": (
            "PLACEMENT_INDEPENDENT" if identical
            else "SCORES_DIFFER_ANSWERS_STABLE" if len(answers) == 1
            else "PLACEMENT_CHANGES_ANSWERS"
        ),
        "interpretation": [
            "PLACEMENT_INDEPENDENT: every replayed logprob is bit-equal to the one in the"
            " bundle, so on this artifact and build the score does not depend on the split.",
            "SCORES_DIFFER_ANSWERS_STABLE: the logprobs moved and no per-sample answer did."
            " Read it with margin_shift_vs_baseline: a surviving answer whose margin barely"
            " moved is a different finding from one that nearly crossed zero.",
            "PLACEMENT_CHANGES_ANSWERS: at least one answer moved, so a persisted score is"
            " only reusable under its own placement.",
            "One artifact, one build, one machine, this sample selection. Nothing here is a"
            " threshold for comparing two models, and nothing here bounds other artifacts,"
            " other llama.cpp builds or other GPUs.",
        ],
        "source_bundle": {
            "path": str(bundle), "artifact_sha256": artifact["sha256"],
            "server_sha256": SERVER_SHA256, "scoring_requests": len(payloads),
            "continuations": len(groups), "samples": len(samples),
            "reported_metrics": {
                metric: results["results"][TASK][metric + ",none"]
                for metric in ("acc", "acc_norm")
            },
        },
        "baseline_metrics": baseline,
        "arms": per_arm,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Interior for both pinned artifacts (22 and 28 blocks), so the split is real.
    parser.add_argument("--partial-gpu-layers", type=int, default=11)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
