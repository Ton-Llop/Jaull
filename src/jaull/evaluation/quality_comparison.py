"""Per-task diagnostic comparison, independent of ranking and local speed."""

from __future__ import annotations

import random
from typing import Any

from jaull.evaluation.quality_records import (
    GENERATION_PROTOCOL,
    GENERATION_SCHEMA_VERSION,
    INSTRUCTION_METRICS,
    NON_REUSABLE_SUITES,
    PLUMBING_SUITES,
    PROMPT_METRICS,
    digest,
    identity_mode,
    validate_record,
)

_BOOTSTRAP_REPLICATES = 2_000


def compare_quality_records(
    left: dict[str, Any], right: dict[str, Any], *, left_source: str, right_source: str,
) -> dict[str, Any]:
    records = [left, right]
    for record in records:
        validate_record(record)
    identities = [record["identity"] for record in records]
    generation = [record["schema_version"] == GENERATION_SCHEMA_VERSION for record in records]
    checks: list[dict[str, Any]] = [
        {"field": field, "match": digest(_comparable(identities[0], field))
         == digest(_comparable(identities[1], field))}
        for field in _COMPARED_FIELDS
    ]
    checks.append({"field": "schema_version", "match": generation[0] == generation[1]})
    checks.append({"field": "classification", "match":
                   records[0]["classification"] == records[1]["classification"]})
    comparable = all(check["match"] for check in checks)
    plumbing = any(
        record["classification"] == "plumbing" or identity["suite"]["name"] in PLUMBING_SUITES
        for record, identity in zip(records, identities, strict=True)
    )
    limited = any(
        record["classification"] == "limited"
        or identity["suite"]["name"] in NON_REUSABLE_SUITES
        or len(identity["dataset"]["sample_ids"]) < identity["dataset"]["total_samples"]
        for record, identity in zip(records, identities, strict=True)
    )
    per_task = []
    if comparable:
        task = identities[0]["suite"]["name"]
        seed = int(digest(sorted(record["identity_sha256"] for record in records))[:16], 16)
        metrics = (
            (*PROMPT_METRICS, *INSTRUCTION_METRICS) if generation[0] else ("acc", "acc_norm")
        )
        for metric in metrics:
            values = []
            outcomes = []
            for record in records:
                flat = [
                    int(item) for sample in record["result"]["samples"][task]
                    for item in (sample[metric] if metric in INSTRUCTION_METRICS
                                 else [sample[metric]])
                ]
                outcomes.append(flat)
                value = record["result"]["results"][task][metric + ",none"]
                values.append({"value": value, "correct": sum(flat), "samples": len(flat)})
            paired_differences = [
                left - right for left, right in zip(outcomes[0], outcomes[1], strict=True)
            ]
            difference = sum(paired_differences) / len(paired_differences)
            per_task.append({
                "task": task,
                "metric": metric,
                "left": values[0],
                "right": values[1],
                "difference": {
                    "left_minus_right": difference,
                    "percentage_points": difference * 100,
                },
                "paired_outcomes": {
                    "left_only_correct": paired_differences.count(1),
                    "right_only_correct": paired_differences.count(-1),
                    "both_same": paired_differences.count(0),
                },
                "uncertainty": {
                    "status": "not_estimated",
                    "reason": "instructions are clustered within prompts; resampling them "
                              "independently would overstate precision",
                } if metric in INSTRUCTION_METRICS else _uncertainty(
                    paired_differences,
                    limited=limited,
                    plumbing=plumbing,
                    seed=seed,
                ),
            })
    return {
        "schema_version": 1, "purpose": "diagnostic_only",
        "status": ("COMPARABLE_PLUMBING" if plumbing else
                   "COMPARABLE_LIMITED" if limited else "COMPARABLE_DIAGNOSTIC")
        if comparable else "NOT_COMPARABLE",
        "checks": checks,
        "reasons": [
            check["field"] + (" matches" if check["match"] else " differs; comparison withheld")
            for check in checks
        ],
        "limitations": [
            "Limited evaluations are plumbing checks, not publishable or reusable quality evidence."
            if plumbing else "Subset diagnostics, not a full benchmark or general-quality verdict."
            if limited else "Full per-task diagnostics; no aggregate or suitability score.",
            "Artifacts include model weights, quantization and tokenizers; no base-model verdict.",
            "Hardware is provenance; no evaluation duration or inference-speed comparison.",
        ],
        "provenance": [
            {
                "side": side, "record_path": source, "record_sha256": digest(record),
                "identity": record["identity"], "hardware": record["provenance"]["hardware"],
            }
            for side, source, record in zip(
                ("left", "right"), (left_source, right_source), records, strict=True,
            )
        ],
        "per_task": per_task,
    }


_COMPARED_FIELDS = ("suite", "dataset", "samples", "evaluator", "runtime", "protocol")


def comparison_key(record: dict[str, Any]) -> str:
    """Equal keys exactly when ``compare_quality_records`` would find every check matching.

    Built from the same fields and the same normalisation, so a cohort grouped
    by this key can never hold two records the comparator would refuse.
    """
    identity = record["identity"]
    return digest({
        **{field: _comparable(identity, field) for field in _COMPARED_FIELDS},
        "schema_version": record["schema_version"],
        "classification": record["classification"],
    })


def comparison_fields(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """One digest per compared field, so a screen can say which one differs.

    Same fields and normalisation as ``comparison_key``: two records have equal
    fields exactly when they have equal keys.
    """
    identity = record["identity"]
    return (
        *((field, digest(_comparable(identity, field))) for field in _COMPARED_FIELDS),
        ("schema_version", digest(record["schema_version"])),
        ("classification", digest(record["classification"])),
    )


def _comparable(identity: dict[str, Any], field: str) -> Any:
    """The part of an identity field that must match for two records to compare.

    Under the chat contract each artifact runs its own embedded template, like
    its own tokenizer, so the template digest is artifact-bound and stays in
    provenance. Its source must still match: both sides used their GGUF's.
    """
    value = identity[field]
    if field == "evaluator" and identity_mode(identity) == GENERATION_PROTOCOL["mode"]:
        return {key: item for key, item in value.items() if key != "chat_template_sha256"}
    return value


def _uncertainty(
    paired_differences: list[int], *, limited: bool, plumbing: bool, seed: int,
) -> dict[str, Any]:
    if plumbing:
        return {"status": "not_estimated", "reason": "plumbing evaluation"}
    if not limited:
        return {
            "status": "not_estimated",
            "reason": "full fixed benchmark split; score is exact for this split",
        }

    rng = random.Random(seed)
    sample_count = len(paired_differences)
    if sample_count < 2 or len(set(paired_differences)) < 2:
        return {
            "status": "not_estimated",
            "reason": "paired outcomes have too little variation for bootstrap resampling",
        }
    estimates = sorted(
        sum(rng.choices(paired_differences, k=sample_count)) / sample_count
        for _ in range(_BOOTSTRAP_REPLICATES)
    )
    return {
        "status": "estimated",
        "method": "paired percentile bootstrap",
        "confidence_level": 0.95,
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": seed,
        "left_minus_right": {
            "lower": estimates[int(0.025 * _BOOTSTRAP_REPLICATES)],
            "upper": estimates[int(0.975 * _BOOTSTRAP_REPLICATES)],
        },
        "limitation": "Exploratory interval for this sample; not a general-quality verdict.",
    }
