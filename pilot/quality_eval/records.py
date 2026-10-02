"""Offline, write-once result snapshots; limited smokes never satisfy quality lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from pilot.quality_eval.evaluate import (
    SAMPLE_IDS,
    TASK,
    validate_artifact_identity,
    validate_scoring_request,
    validate_scoring_response,
    verified_dataset_bytes,
    write_json,
)

JSON_FILES = (
    "verified-artifact.json", "artifact-after.json", "container-artifact.json",
    "commands.json", "image.json", "hardware.json", "readiness.json",
    "memory-preflight.json", "server-props.json", "evaluator-config.json",
    "dataset.json", "lm-eval-results.json", "smoke-status.json", "runner-status.json",
)
TEXT_FILES = ("http.jsonl", "server.log", "evaluator.log")
EVALUATOR_FIELDS = {
    "evaluator_commit", "backend_sha256", "suite_sha256", "sample_ids", "num_fewshot",
    "context", "apply_chat_template", "system_instruction", "generation", "use_cache",
    "cache_requests", "seeds", "server_seed", "python", "packages",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def known(value: Any) -> bool:
    """Null/empty identity values are unknown, except explicitly checked absences."""
    if isinstance(value, dict):
        return bool(value) and all(known(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(known(item) for item in value)
    return value is not None and value != ""


def validate_identity(identity: dict[str, Any]) -> None:
    if set(identity) != {"artifact_sha256", "suite", "dataset", "samples",
                         "evaluator", "runtime", "protocol"}:
        raise ValueError("Missing/unknown identity fields")
    evaluator = identity["evaluator"]
    if set(evaluator) != EVALUATOR_FIELDS:
        raise ValueError("Incomplete evaluator identity")
    absent = {"system_instruction", "generation", "use_cache"}
    if (
        any(evaluator[key] is not None for key in absent)
        or evaluator["apply_chat_template"] is not False
        or evaluator["cache_requests"] is not False
        or not known({key: value for key, value in evaluator.items() if key not in absent})
        or not known({key: value for key, value in identity.items() if key != "evaluator"})
    ):
        raise ValueError("Unknown identity or unsupported chat/generation/cache mode")
    suite, dataset, runtime = (identity[key] for key in ("suite", "dataset", "runtime"))
    if (
        set(suite) != {"name", "sha256"}
        or set(dataset) != {"repo", "revision", "sha256", "sample_ids", "total_samples"}
        or set(runtime) != {"server_sha256", "image_id", "fingerprint", "backend_flags",
                            "server_defaults_sha256"}
        or identity["protocol"] != {
            "mode": "loglikelihood", "tokenize_add_special": True,
            "temperature": 0, "max_tokens": 1, "logprobs": 2, "id_slot": 0,
            "target_logit_bias": 100, "cached_prompt_tokens": 0,
        }
    ):
        raise ValueError("Unsupported identity schema/protocol")
    for sha in (identity["artifact_sha256"], suite["sha256"], dataset["sha256"],
                runtime["server_sha256"], evaluator["backend_sha256"],
                runtime["server_defaults_sha256"],
                runtime["image_id"].removeprefix("sha256:")):
        if not isinstance(sha, str) or re.fullmatch("[0-9a-f]{64}", sha) is None:
            raise ValueError("Unknown exact digest")
    ids = dataset["sample_ids"]
    if (
        type(dataset["total_samples"]) is not int or dataset["total_samples"] < 1
        or not isinstance(ids, list) or not ids
        or any(type(item) is not int or not 0 <= item < dataset["total_samples"] for item in ids)
        or len(set(ids)) != len(ids) or evaluator["sample_ids"] != ids
        or evaluator["suite_sha256"] != suite["sha256"]
        or type(evaluator["context"]) is not int or evaluator["context"] < 1
        or re.fullmatch("[0-9a-f]{40}", evaluator["evaluator_commit"]) is None
        or re.fullmatch("[0-9a-f]{40}", dataset["revision"]) is None
        or not isinstance(evaluator["packages"], dict) or not evaluator["packages"]
        or not isinstance(runtime["backend_flags"], list)
        or any(not isinstance(flag, str) for flag in runtime["backend_flags"])
    ):
        raise ValueError("Incomplete sample/config identity")
    samples = identity["samples"]
    if not isinstance(samples, list) or [sample["doc_id"] for sample in samples] != ids:
        raise ValueError("Incomplete selected sample identity")
    for sample in samples:
        if set(sample) != {"doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments"}:
            raise ValueError("Unknown sample identity")
        for field in ("doc_hash", "prompt_hash", "target_hash"):
            if re.fullmatch("[0-9a-f]{64}", sample[field]) is None:
                raise ValueError("Unknown sample/prompt digest")


def validate_record(record: dict[str, Any]) -> None:
    if record["schema_version"] != 1 or record["status"] != "completed":
        raise ValueError("Unsupported/unfinished result record")
    identity = record["identity"]
    validate_identity(identity)
    if not record["provenance"]["hardware"]:
        raise ValueError("Missing hardware provenance")
    if record["identity_sha256"] != digest(identity):
        raise ValueError("Identity checksum mismatch")
    result, task = record["result"], identity["suite"]["name"]
    ids = identity["dataset"]["sample_ids"]
    if (
        record["classification"] not in ("plumbing", "full")
        or result["n-samples"][task] != {
            "original": identity["dataset"]["total_samples"], "effective": len(ids),
        }
        or [sample["doc_id"] for sample in result["samples"][task]] != ids
        or not result["results"][task] or not result["configs"][task]
        or any(
            {key: sample[key] for key in expected} != expected
            for sample, expected in zip(result["samples"][task], identity["samples"], strict=True)
        )
    ):
        raise ValueError("Incomplete/mismatched raw results")
    for sample in result["samples"][task]:
        if (
            len(sample["resps"]) != len(sample["arguments"])
            or len(sample["filtered_resps"]) != len(sample["arguments"])
            or any(sample[metric] not in (0, 1) for metric in ("acc", "acc_norm"))
            or "error" in sample
        ):
            raise ValueError("Incomplete sample responses/metrics")
        for repeats in sample["resps"]:
            if not isinstance(repeats, list) or not repeats:
                raise ValueError("Missing continuation scores")
        for score in [value for repeats in sample["resps"] for value in repeats] + sample[
            "filtered_resps"
        ]:
            if (
                not isinstance(score, list) or len(score) != 2
                or type(score[0]) not in (int, float) or not math.isfinite(score[0])
                or score[0] > 0 or type(score[1]) is not bool
            ):
                raise ValueError("Invalid/missing continuation score")
    for metric in ("acc", "acc_norm"):
        value = result["results"][task][metric + ",none"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid/missing task metric")
        expected = sum(sample[metric] for sample in result["samples"][task]) / len(ids)
        if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Saved metric differs from completed per-sample results")
    if "evidence" in record["provenance"]:
        fingerprint = validate_http_coverage(record["provenance"]["evidence"]["http.jsonl"],
                                             result["samples"][task])
        if fingerprint != identity["runtime"]["fingerprint"]:
            raise ValueError("HTTP runtime differs from record identity")


def validate_http_coverage(text: str, samples: list[dict[str, Any]]) -> str:
    """Reconstruct every saved continuation from the pinned backend's HTTP evidence."""
    tokenizations: dict[str, list[int]] = {}
    scores: dict[tuple[tuple[int, ...], int], deque[float]] = defaultdict(deque)
    fingerprints = set()
    for line in text.splitlines():
        exchange = json.loads(line)
        if "error" in exchange or "response" not in exchange:
            raise ValueError("Failed HTTP evidence")
        request, response = exchange["request"], exchange["response"]
        if exchange["url"].endswith("/v1/completions"):
            validate_scoring_request(request)
            validate_scoring_response(request, response)
            key = (tuple(request["prompt"]), request["logit_bias"][0][0])
            scores[key].append(response["choices"][0]["logprobs"]["content"][0]["logprob"])
            fingerprints.add(response["system_fingerprint"])
        elif exchange["url"].endswith("/tokenize"):
            if (
                set(request) != {"content", "add_special"}
                or request["add_special"] is not True
                or not isinstance(request["content"], str)
                or not isinstance(response["tokens"], list)
                or not response["tokens"]
                or any(type(token) is not int or token < 0 for token in response["tokens"])
            ):
                raise ValueError("Unknown tokenizer protocol")
            content, tokens = request["content"], response["tokens"]
            if content in tokenizations and tokenizations[content] != tokens:
                raise ValueError("Conflicting tokenizer evidence")
            tokenizations[content] = tokens
        else:
            raise ValueError("Unknown HTTP protocol")
    for sample in samples:
        for index, (context, continuation) in enumerate(sample["arguments"]):
            # Mirror the pinned GGUF backend's trailing-whitespace migration.
            whole_text, prefix_text = context + continuation, context.rstrip()
            if whole_text not in tokenizations or prefix_text not in tokenizations:
                raise ValueError("Missing tokenizer evidence for selected continuation")
            whole, prefix = tokenizations[whole_text], tokenizations[prefix_text]
            if whole[:len(prefix)] != prefix:
                raise ValueError("Ambiguous tokenization boundary in HTTP evidence")
            total = 0.0
            for position in range(len(prefix), len(whole)):
                key = (tuple(whole[:position]), whole[position])
                if not scores[key]:
                    raise ValueError("Missing scoring evidence for selected continuation")
                total += scores[key].popleft()
            repeats = sample["resps"][index]
            if len(repeats) != 1 or any(
                not math.isclose(total, score[0], rel_tol=0, abs_tol=1e-9)
                for score in (repeats[0], sample["filtered_resps"][index])
            ):
                raise ValueError("HTTP logprobs do not reconstruct saved continuation score")
    if len(fingerprints) != 1 or any(scores.values()):
        raise ValueError("Missing/mismatched scoring coverage")
    return next(iter(fingerprints))


def save_record(path: Path, record: dict[str, Any]) -> None:
    validate_record(record)
    # Serialize before exclusive creation; interrupted/truncated files are lookup misses.
    envelope = {"record": record, "record_sha256": digest(record)}
    write_json(path, envelope)


def load_record(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    record = envelope["record"]
    validate_record(record)
    if envelope["record_sha256"] != digest(record):
        raise ValueError("Result record checksum mismatch")
    return record


def quality_lookup(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Read only. Return raw harness data; no speed evidence or hardware-based match."""
    try:
        validate_identity(identity)
        record = load_record(path)
        dataset = identity["dataset"]
        if (
            record["identity_sha256"] != digest(identity)
            or record["classification"] != "full"
            or identity["suite"]["name"] == TASK
            or dataset["sample_ids"] != list(range(dataset["total_samples"]))
        ):
            return None
        return record["result"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
        return None


def snapshot_bundle(bundle: Path) -> dict[str, Any]:
    if list(bundle.glob("*-error.json")):
        raise ValueError("Failed bundle cannot become a completed record")
    raw = {name: (bundle / name).read_text(encoding="utf-8") for name in JSON_FILES + TEXT_FILES}
    evidence = {name: json.loads(raw[name]) for name in JSON_FILES}
    artifact = evidence["verified-artifact.json"]
    validate_artifact_identity(artifact)
    if (
        evidence["smoke-status.json"] != {"status": "plumbing_passed", "quality_evidence": False}
        or evidence["runner-status.json"] != {
            "status": "plumbing_passed", "quality_evidence": False, "server_exit_code": 0,
        }
        or evidence["artifact-after.json"] != artifact
        or evidence["container-artifact.json"] != {
            "sha256": artifact["sha256"], "mount": "read-only",
        }
    ):
        raise ValueError("Incomplete execution/artifact verification")
    dataset_bytes = verified_dataset_bytes(bundle / "dataset-validation.parquet")
    evaluator, dataset, commands, props = (
        evidence[name] for name in ("evaluator-config.json", "dataset.json",
                                   "commands.json", "server-props.json")
    )
    result = evidence["lm-eval-results.json"]
    samples = result["samples"][TASK]
    fingerprint = validate_http_coverage(raw["http.jsonl"], samples)
    if (
        dataset["sample_ids"] != SAMPLE_IDS
        or props["total_slots"] != 1
        or props["default_generation_settings"]["n_ctx"] != evaluator["context"]
        or props["default_generation_settings"]["params"]["post_sampling_probs"] is not False
        or props["model_path"] != artifact["local_path"]
        or commands["server"][1:3] != ["--model", artifact["local_path"]]
        or dataset["sha256"] != hashlib.sha256(dataset_bytes).hexdigest()
    ):
        raise ValueError("Missing/mismatched effective protocol evidence")
    identity = {
        "artifact_sha256": artifact["sha256"],
        "suite": {"name": TASK, "sha256": evaluator["suite_sha256"]},
        "dataset": {key: dataset[key] for key in ("repo", "revision", "sha256", "sample_ids")}
        | {"total_samples": result["n-samples"][TASK]["original"]},
        "samples": [{key: sample[key] for key in (
            "doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments",
        )} for sample in samples],
        "evaluator": evaluator,
        "runtime": {
            "server_sha256": commands["server_sha256"], "image_id": evidence["image.json"]["id"],
            "fingerprint": fingerprint, "backend_flags": commands["server"][3:],
            "server_defaults_sha256": digest(props["default_generation_settings"]),
        },
        "protocol": {
            "mode": "loglikelihood", "tokenize_add_special": True,
            "temperature": 0, "max_tokens": 1, "logprobs": 2, "id_slot": 0,
            "target_logit_bias": 100, "cached_prompt_tokens": 0,
        },
    }
    record = {
        "schema_version": 1, "status": "completed", "classification": "plumbing",
        "identity": identity, "identity_sha256": digest(identity), "result": result,
        "provenance": {
            "source_bundle": str(bundle.resolve()), "hardware": evidence["hardware.json"],
            "evidence": raw,
            "dataset_file": {"sha256": dataset["sha256"], "size_bytes": len(dataset_bytes)},
        },
    }
    validate_record(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_record(args.output, snapshot_bundle(args.bundle))


if __name__ == "__main__":
    main()
