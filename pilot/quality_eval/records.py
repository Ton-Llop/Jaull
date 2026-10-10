"""Produce write-once snapshots of a finished bundle, validated harder than on read.

The reusable half of this module now lives in
:mod:`jaull.evaluation.quality_records`, because the advisor and the interface
read records and ``pilot/`` does not ship in the wheel. What stays here is what
only the producer can check: the audited sample selection of each profile, and
the reconstruction of every saved continuation from the captured HTTP evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from pilot.quality_eval import ifeval
from pilot.quality_eval.evaluate import (
    DATASET_SIZE,
    LIMITED_TASK,
    profile_for_task,
    validate_artifact_identity,
    validate_scoring_request,
    validate_scoring_response,
    verified_dataset_bytes,
)

from jaull.evaluation.quality_records import (
    GENERATION_PROTOCOL,
    GENERATION_SCHEMA_VERSION,
    digest,
    known,
    validate_identity,
)
from jaull.evaluation.quality_records import (
    load_record as _load_record_contract,
)
from jaull.evaluation.quality_records import (
    quality_lookup as _quality_lookup_contract,
)
from jaull.evaluation.quality_records import (
    save_record as _save_record_contract,
)
from jaull.evaluation.quality_records import (
    validate_record as _validate_record_contract,
)

JSON_FILES = (
    "verified-artifact.json", "artifact-after.json", "container-artifact.json",
    "commands.json", "image.json", "hardware.json", "readiness.json",
    "memory-preflight.json", "server-props.json", "evaluator-config.json",
    "dataset.json", "lm-eval-results.json", "smoke-status.json", "runner-status.json",
)
TEXT_FILES = ("http.jsonl", "server.log", "evaluator.log")
LOG_FILES = ("server.log", "evaluator.log")  # Stored, never parsed: decoded leniently.
def validate_record(record: dict[str, Any]) -> None:
    """The shared contract, plus the two things only the producer can verify."""
    _validate_record_contract(record)
    identity = record["identity"]
    task = identity["suite"]["name"]
    if record["schema_version"] == GENERATION_SCHEMA_VERSION:
        profile = profile_for_task(task)
        if (
            profile.get("suite") != "ifeval"
            or identity["dataset"]["sample_ids"] != profile["sample_ids"]
            or record["classification"] != profile["classification"]
        ):
            raise ValueError("Generation record does not match its fixed IFEval profile")
        if "evidence" in record["provenance"]:
            evidence = record["provenance"]["evidence"]
            fingerprint, outcome = validate_generation_http(
                evidence["http.jsonl"], record["result"]["samples"][task],
            )
            if fingerprint != identity["runtime"]["fingerprint"] or outcome != record["outcome"]:
                raise ValueError("HTTP evidence differs from record identity or outcome")
        return
    if task == LIMITED_TASK and (
        identity["dataset"]["sample_ids"] != profile_for_task(task)["sample_ids"]
        or identity["dataset"]["total_samples"] != DATASET_SIZE
    ):
        raise ValueError("Limited result does not match the fixed 100-example selection")
    if "evidence" in record["provenance"]:
        fingerprint = validate_http_coverage(
            record["provenance"]["evidence"]["http.jsonl"], record["result"]["samples"][task]
        )
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


def validate_generation_http(
    text: str, samples: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    """Every scored reply must be the exact text one audited exchange returned."""
    replies: dict[str, tuple[str, str]] = {}
    fingerprints = set()
    for line in text.splitlines():
        exchange = json.loads(line)
        if "error" in exchange or "response" not in exchange:
            raise ValueError("Failed HTTP evidence")
        if not exchange["url"].endswith("/v1/chat/completions"):
            raise ValueError("Unknown HTTP protocol")
        request, response = exchange["request"], exchange["response"]
        ifeval.validate_chat_request(request)
        finish = ifeval.validate_chat_response(response)
        rendered = exchange.get("rendered_prompt")
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValueError("Missing rendered chat prompt")
        prompt = request["messages"][0]["content"]
        if prompt in replies:
            raise ValueError("Duplicate generation for one prompt")
        replies[prompt] = (response["choices"][0]["message"]["content"], finish)
        fingerprints.add(response["system_fingerprint"])
    finishes = []
    for sample in samples:
        prompt = sample["doc"]["prompt"]
        if prompt not in replies:
            raise ValueError("Missing generation evidence for a selected prompt")
        reply, finish = replies.pop(prompt)
        if sample["resps"] != [[reply]] or sample["filtered_resps"] != [reply]:
            raise ValueError("Scored reply differs from the HTTP response")
        finishes.append(finish)
    if replies or len(fingerprints) != 1:
        raise ValueError("Missing/mismatched generation coverage")
    return next(iter(fingerprints)), {
        "responses": len(finishes), "truncated": finishes.count("length"),
    }


def _snapshot_generation(
    bundle: Path, evidence: dict[str, Any], raw: dict[str, str], profile: dict[str, Any],
) -> dict[str, Any]:
    artifact, result = evidence["verified-artifact.json"], evidence["lm-eval-results.json"]
    task = profile["task"]
    for name in ("chat-template.json", "generation-summary.json"):
        raw[name] = (bundle / name).read_text(encoding="utf-8")
        evidence[name] = json.loads(raw[name])
    dataset_bytes = ifeval.verified_dataset_bytes(bundle / "dataset-ifeval.jsonl")
    evaluator, dataset, commands, props, template = (
        evidence[name] for name in ("evaluator-config.json", "dataset.json", "commands.json",
                                   "server-props.json", "chat-template.json")
    )
    samples = result["samples"][task]
    fingerprint, outcome = validate_generation_http(raw["http.jsonl"], samples)
    served = props.get("chat_template")
    served_sha256 = hashlib.sha256(served.encode()).hexdigest() if isinstance(served, str) else ""
    server = commands["server"]
    if (
        dataset["sample_ids"] != profile["sample_ids"]
        or dataset["sha256"] != hashlib.sha256(dataset_bytes).hexdigest()
        or props["total_slots"] != 1
        or props["default_generation_settings"]["n_ctx"] != evaluator["context"]
        or props["model_path"] != artifact["local_path"]
        or server[1:3] != ["--model", artifact["local_path"]]
        or "--jinja" not in server
        or server[server.index("--reasoning") + 1:server.index("--reasoning") + 2] != ["off"]
        or template.get("source") != "gguf"
        or evaluator["chat_template_sha256"] != served_sha256
        # The server strips the template's trailing whitespace; nothing else is accepted.
        or served_sha256 not in {template.get("embedded_sha256"),
                                 template.get("embedded_stripped_sha256")}
        or evidence["generation-summary.json"] != outcome
    ):
        raise ValueError("Missing/mismatched effective protocol evidence")
    identity = {
        "artifact_sha256": artifact["sha256"],
        "suite": {"name": task, "sha256": evaluator["suite_sha256"]},
        "dataset": {key: dataset[key] for key in ("repo", "revision", "sha256", "sample_ids")}
        | {"total_samples": result["n-samples"][task]["original"]},
        "samples": [{key: sample[key] for key in (
            "doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments",
        )} for sample in samples],
        "evaluator": evaluator,
        "runtime": {
            "server_sha256": commands["server_sha256"], "image_id": evidence["image.json"]["id"],
            "fingerprint": fingerprint, "backend_flags": server[3:],
            "server_defaults_sha256": digest(props["default_generation_settings"]),
        },
        "protocol": dict(GENERATION_PROTOCOL),
    }
    record = {
        "schema_version": GENERATION_SCHEMA_VERSION, "status": "completed",
        "classification": profile["classification"],
        "identity": identity, "identity_sha256": digest(identity), "result": result,
        "outcome": outcome,
        "provenance": {
            "source_bundle": str(bundle.resolve()), "hardware": evidence["hardware.json"],
            "evidence": raw,
            "dataset_file": {"sha256": dataset["sha256"], "size_bytes": len(dataset_bytes)},
        },
    }
    validate_record(record)
    return record


def save_record(path: Path, record: dict[str, Any]) -> None:
    _save_record_contract(path, record, validate=validate_record)


def load_record(path: Path) -> dict[str, Any]:
    return _load_record_contract(path, validate=validate_record)


def quality_lookup(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    return _quality_lookup_contract(path, identity, validate=validate_record)


def snapshot_bundle(bundle: Path) -> dict[str, Any]:
    if list(bundle.glob("*-error.json")):
        raise ValueError("Failed bundle cannot become a completed record")
    raw = {name: (bundle / name).read_text(encoding="utf-8") for name in JSON_FILES + TEXT_FILES
           if name not in LOG_FILES}
    # llama.cpp truncates its GGUF metadata previews by bytes, splitting a UTF-8
    # character in some tokenizers (LFM2.5 merges). The logs are diagnostics that
    # nothing parses, so the stray bytes are kept as \xNN rather than dropped.
    raw |= {name: (bundle / name).read_bytes().decode("utf-8", "backslashreplace")
            for name in LOG_FILES}
    evidence = {name: json.loads(raw[name]) for name in JSON_FILES}
    artifact = evidence["verified-artifact.json"]
    validate_artifact_identity(artifact)
    result = evidence["lm-eval-results.json"]
    if len(result["samples"]) != 1:
        raise ValueError("Expected exactly one fixed evaluation task")
    task = next(iter(result["samples"]))
    profile = profile_for_task(task)
    if (
        evidence["smoke-status.json"] != {"status": profile["status"], "quality_evidence": False}
        or evidence["runner-status.json"] != {
            "status": profile["status"], "quality_evidence": False, "server_exit_code": 0,
        }
        or evidence["artifact-after.json"] != artifact
        or evidence["container-artifact.json"] != {
            "sha256": artifact["sha256"], "mount": "read-only",
        }
    ):
        raise ValueError("Incomplete execution/artifact verification")
    if profile.get("suite") == "ifeval":
        # Generation has its own record shape; it never passes the HellaSwag checks.
        return _snapshot_generation(bundle, evidence, raw, profile)
    dataset_bytes = verified_dataset_bytes(bundle / "dataset-validation.parquet")
    evaluator, dataset, commands, props = (
        evidence[name] for name in ("evaluator-config.json", "dataset.json",
                                   "commands.json", "server-props.json")
    )
    samples = result["samples"][task]
    fingerprint = validate_http_coverage(raw["http.jsonl"], samples)
    if (
        dataset["sample_ids"] != profile["sample_ids"]
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
        "suite": {"name": task, "sha256": evaluator["suite_sha256"]},
        "dataset": {key: dataset[key] for key in ("repo", "revision", "sha256", "sample_ids")}
        | {"total_samples": result["n-samples"][task]["original"]},
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
        "schema_version": 1, "status": "completed",
        "classification": profile["classification"],
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


__all__ = [
    "JSON_FILES",
    "TEXT_FILES",
    "digest",
    "known",
    "load_record",
    "quality_lookup",
    "save_record",
    "snapshot_bundle",
    "validate_http_coverage",
    "validate_identity",
    "validate_record",
]
