"""Explicit quality evaluation and offline diagnostic comparison commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.evaluation.quality_storage import QualityStoreError
from jaull.runtime.quality_eval_runner import QualityEvaluationError, QualityRunRequest


def run_quality(
    requests: Sequence[QualityRunRequest], *, as_json: bool = False,
    advisor: AdvisorService | None = None,
) -> int:
    resolved = advisor or AdvisorService.default()
    completed: list[dict[str, str]] = []
    for request in requests:
        try:
            path = resolved.run_quality_evaluation(request)
        except (
            QualityEvaluationError, QualityStoreError, OSError, ValueError, KeyError, TypeError,
        ) as exc:
            payload = {"status": "failed", "purpose": "diagnostic_only",
                       "completed": completed, "error": str(exc),
                       "failed_output": str(request.output)}
            if as_json:
                sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            else:
                Console(stderr=True).print(str(exc), markup=False)
                for item in completed:
                    print(f"Preserved completed record: {item['identity_sha256']}")
            return 3
        completed.append({"identity_sha256": path.stem, "record_path": str(path),
                          "output": str(request.output), "profile": request.profile.value})
    if as_json:
        sys.stdout.write(json.dumps({"status": "completed", "purpose": "diagnostic_only",
                                    "completed": completed}, indent=2) + "\n")
    else:
        for item in completed:
            print(f"Stored quality diagnostic ({item['profile']}): {item['identity_sha256']}")
        print("Subset results are not full benchmarks or general-quality verdicts.")
    return 0


def run_quality_compare(
    left_id: str, right_id: str, *, as_json: bool = False,
    advisor: AdvisorService | None = None,
) -> int:
    resolved = advisor or AdvisorService.default()
    try:
        report = resolved.compare_quality(left_id, right_id)
    except (QualityStoreError, OSError, ValueError, KeyError, TypeError) as exc:
        if as_json:
            sys.stderr.write(json.dumps({"schema_version": 1, "error": str(exc)}) + "\n")
        else:
            Console(stderr=True).print(str(exc), markup=False)
        return 3
    if as_json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        _render_comparison(report)
    return 0


def _render_comparison(report: dict[str, Any]) -> None:
    console = Console()
    console.print(report["status"], markup=False)
    for row in report["per_task"]:
        console.print(
            f"{row['task']} · {row['metric']}: "
            f"left {row['left']['value']:.1%} "
            f"({row['left']['correct']}/{row['left']['samples']}) · "
            f"right {row['right']['value']:.1%} "
            f"({row['right']['correct']}/{row['right']['samples']})", markup=False,
        )
    if report["status"] == "NOT_COMPARABLE":
        for reason in report["reasons"]:
            console.print(reason, markup=False)
    for limitation in report["limitations"]:
        console.print(limitation, markup=False)
