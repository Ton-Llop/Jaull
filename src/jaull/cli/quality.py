"""Explicit quality evaluation and offline diagnostic comparison commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.evaluation.quality_storage import QualityStoreError
from jaull.paths import user_data_dir
from jaull.runtime.quality_eval_runner import QualityEvaluationError, QualityRunRequest


def run_quality_setup(
    *, dataset_file: Path | None = None, llama_server: Path | None = None,
    pilot_root: Path | None = None, image: str | None = None,
    allow_dataset_download: bool = False, as_json: bool = False,
    advisor: AdvisorService | None = None,
) -> int:
    resolved = advisor or AdvisorService.default()
    try:
        setup = resolved.quality_evaluation_setup()
        for key, value in (("dataset", dataset_file), ("server", llama_server),
                           ("root", pilot_root), ("image", image)):
            if value is not None:
                setup[key] = str(value)
        missing = [key for key in ("dataset", "server", "root", "image") if not setup.get(key)]
        if missing:
            raise QualityEvaluationError("Missing evaluator setup: " + ", ".join(missing))
        request = QualityRunRequest(
            artifact_json=None, dataset_file=Path(setup["dataset"]),
            llama_server=Path(setup["server"]), pilot_root=Path(setup["root"]),
            image=setup["image"], output=user_data_dir("quality-runs"),
            allow_dataset_download=allow_dataset_download,
        )
        resolved.prepare_quality_evaluation_setup(request)
        resolved.remember_quality_evaluation_setup(request)
    except (QualityEvaluationError, OSError, ValueError) as exc:
        payload = {"status": "blocked", "error": str(exc)}
        code = 3
    else:
        payload = {"status": "ready", "scope": "infrastructure_only"}
        code = 0
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        Console().print(payload.get("error", "Evaluator infrastructure ready."), markup=False)
    return code


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
