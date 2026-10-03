"""Per-task diagnostic comparison, independent of ranking and local speed."""

from __future__ import annotations

from typing import Any

from jaull.evaluation.quality_records import (
    NON_REUSABLE_SUITES,
    PLUMBING_SUITES,
    digest,
    validate_record,
)


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
        for metric in ("acc", "acc_norm"):
            values = []
            for record in records:
                samples = record["result"]["samples"][task]
                correct = int(sum(sample[metric] for sample in samples))
                value = record["result"]["results"][task][metric + ",none"]
                values.append({"value": value, "correct": correct, "samples": len(samples)})
            per_task.append({"task": task, "metric": metric,
                             "left": values[0], "right": values[1]})
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
