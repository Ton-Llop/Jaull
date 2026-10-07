"""Per-task diagnostic comparison, independent of ranking and local speed."""

from __future__ import annotations

import random
from typing import Any

from jaull.evaluation.quality_records import (
    NON_REUSABLE_SUITES,
    PLUMBING_SUITES,
    digest,
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
    checks: list[dict[str, Any]] = [
        {"field": field, "match": digest(identities[0][field]) == digest(identities[1][field])}
        for field in ("suite", "dataset", "samples", "evaluator", "runtime", "protocol")
    ]
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
        for metric in ("acc", "acc_norm"):
            values = []
            paired_differences: list[int] = []
            for record in records:
                samples = record["result"]["samples"][task]
                correct = int(sum(sample[metric] for sample in samples))
                value = record["result"]["results"][task][metric + ",none"]
                values.append({"value": value, "correct": correct, "samples": len(samples)})
            left_samples, right_samples = (
                record["result"]["samples"][task] for record in records
            )
            paired_differences = [
                left_sample[metric] - right_sample[metric]
                for left_sample, right_sample in zip(
                    left_samples, right_samples, strict=True,
                )
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
                "uncertainty": _uncertainty(
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
