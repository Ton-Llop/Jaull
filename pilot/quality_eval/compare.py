"""Offline per-task diagnostics. No quality aggregate, ranking or speed evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pilot.quality_eval.evaluate import TASK, write_json
from pilot.quality_eval.records import digest, load_record


def compare_records(left: Path, right: Path) -> dict[str, Any]:
    records = [load_record(path) for path in (left, right)]
    identities = [record["identity"] for record in records]
    checks = [
        {"field": field, "match": digest(identities[0][field]) == digest(identities[1][field])}
        for field in ("suite", "dataset", "samples", "evaluator", "runtime", "protocol")
    ]
    checks.append({"field": "classification", "match":
                   records[0]["classification"] == records[1]["classification"]})
    comparable = all(check["match"] for check in checks)
    limited = any(
        record["classification"] == "plumbing"
        or identity["suite"]["name"] == TASK
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
        "status": ("COMPARABLE_PLUMBING" if limited else "COMPARABLE_DIAGNOSTIC")
        if comparable else "NOT_COMPARABLE",
        "checks": checks,
        "reasons": [
            check["field"] + (" matches" if check["match"] else " differs; comparison withheld")
            for check in checks
        ],
        "limitations": [
            "Limited evaluations are plumbing checks, not publishable or reusable quality evidence."
            if limited else "Full per-task diagnostics; no aggregate or suitability score.",
            "Artifacts include model weights, quantization and tokenizers; no base-model verdict.",
            "Hardware is provenance; no evaluation duration or inference-speed comparison.",
        ],
        "provenance": [
            {
                "side": side, "record_path": str(path.resolve()), "record_sha256": digest(record),
                "identity": record["identity"], "hardware": record["provenance"]["hardware"],
            }
            for side, path, record in zip(("left", "right"), (left, right), records, strict=True)
        ],
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, compare_records(args.left, args.right))


if __name__ == "__main__":
    main()
