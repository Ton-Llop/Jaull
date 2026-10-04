"""Explicit CLI bridge to the audited, source-checkout quality pilot.

The wheel consumes records but does not contain the evaluator/container build.
Keep that boundary explicit rather than importing an optional pilot at startup.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from jaull.domain.artifacts import ModelArtifact
from jaull.evaluation.quality_records import load_record
from jaull.exceptions import JaullError


class QualityProfile(StrEnum):
    SMOKE = "smoke"
    HELLASWAG100 = "hellaswag100"


class QualityEvaluationError(JaullError):
    """The pilot failed; its output must not become stored quality evidence."""


class QualityEvaluationCancelled(QualityEvaluationError):
    """The operator cancelled; owned processes have finished cleanup."""


@dataclass(frozen=True)
class QualityRunRequest:
    artifact_json: Path
    dataset_file: Path
    llama_server: Path
    image: str
    output: Path
    pilot_root: Path
    profile: QualityProfile = QualityProfile.SMOKE


def _check_cancel(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise QualityEvaluationCancelled("Evaluation cancelled; no result imported.")


def _invoke(
    command: list[str], root: Path, log: Path,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    _check_cancel(is_cancelled)
    environment = os.environ.copy()
    # The evaluator remains an explicit source-checkout dependency, even when
    # Jaull itself was installed from a wheel. Never execute through a shell.
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    with (
        log.open("x", encoding="utf-8") as handle,
        subprocess.Popen(
            command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            if is_cancelled is None:
                code = process.wait()
            else:
                while True:
                    _check_cancel(is_cancelled)
                    try:
                        code = process.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        except QualityEvaluationCancelled:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
        except KeyboardInterrupt:
            # Let the pilot's finally stop its owned container/server and
            # finish artifact-after verification; do not kill its owner.
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
    if code:
        raise QualityEvaluationError(f"Quality pilot exited with {code}; inspect {log}.")


def run_quality_evaluation(
    request: QualityRunRequest, *, is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one bounded profile, then reconstruct its HTTP-validated snapshot.

    No download, build, cache reuse or model substitution is performed here.
    The existing pilot owns verification, fit/readiness and process cleanup.
    """
    _check_cancel(is_cancelled)
    if os.name != "posix":
        raise QualityEvaluationError("This pinned host/container pilot requires Linux or WSL.")
    root = request.pilot_root.expanduser().resolve()
    required = ("scripts/quality_eval_smoke.py", "pilot/quality_eval/evaluate.py",
                "pilot/quality_eval/records.py", "pilot/quality_eval/suite.yaml")
    if any(not (root / name).is_file() for name in required):
        raise QualityEvaluationError(
            "Quality execution requires the Jaull pilot checkout; set --pilot-root to its root."
        )
    if not isinstance(request.profile, QualityProfile):
        raise QualityEvaluationError("Unsupported fixed evaluation profile.")
    artifact = ModelArtifact.model_validate_json(
        request.artifact_json.expanduser().read_text(encoding="utf-8")
    )
    if artifact.local_path is None or not artifact.sha256:
        raise QualityEvaluationError("An exact local artifact path and SHA256 are required.")
    local_path = artifact.local_path.expanduser().resolve()
    artifact = artifact.model_copy(update={"local_path": local_path})
    dataset = request.dataset_file.expanduser().resolve()
    server = request.llama_server.expanduser().resolve()
    if not local_path.is_file() or not dataset.is_file() or not server.is_file():
        raise QualityEvaluationError(
            "Artifact, dataset and llama-server must already exist locally."
        )
    output = request.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact_json = output / "artifact.json"
    artifact_json.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    bundle, snapshot = output / "bundle", output / "record.json"
    # Keep the synchronous CLI call unchanged; TUI workers opt into polling.
    invoke_args = () if is_cancelled is None else (is_cancelled,)
    _invoke(
        [sys.executable, "-m", "scripts.quality_eval_smoke",
         "--artifact-json", str(artifact_json), "--dataset-file", str(dataset),
         "--llama-server", str(server), "--image", request.image,
         "--profile", request.profile.value, "--output", str(bundle)],
        root, output / "run.log", *invoke_args,
    )
    _invoke(
        [sys.executable, "-m", "pilot.quality_eval.records",
         "--bundle", str(bundle), "--output", str(snapshot)],
        root, output / "snapshot.log", *invoke_args,
    )
    record = load_record(snapshot)
    if record["identity"]["artifact_sha256"] != artifact.sha256:
        raise QualityEvaluationError("Produced record belongs to a different artifact.")
    expected_grade = "plumbing" if request.profile is QualityProfile.SMOKE else "limited"
    if record["classification"] != expected_grade:
        raise QualityEvaluationError("Produced record belongs to a different evaluation profile.")
    _check_cancel(is_cancelled)
    return record
