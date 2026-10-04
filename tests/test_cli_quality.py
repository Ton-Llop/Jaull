"""Offline phase-C tests. Synthetic records; no Docker, models or GPU runs."""

from __future__ import annotations

import json
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from jaull.advisor.service import AdvisorService
from jaull.cli.app import app
from jaull.domain.artifacts import ModelArtifact
from jaull.evaluation.quality_records import digest, save_record
from jaull.evaluation.quality_storage import QualityEvidenceStore
from jaull.runtime import quality_eval_runner as runner
from jaull.runtime.quality_eval_runner import (
    QualityEvaluationError,
    QualityProfile,
    QualityRunRequest,
)
from tests.test_quality_eval_records import synthetic_full_record


def _request(tmp_path: Path) -> QualityRunRequest:
    root = tmp_path / "trusted checkout"
    for name in ("scripts/quality_eval_smoke.py", "pilot/quality_eval/evaluate.py",
                 "pilot/quality_eval/records.py", "pilot/quality_eval/suite.yaml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic placeholder, never executed", encoding="utf-8")
    model, server, dataset = [tmp_path / name for name in ("model.gguf", "server", "data.parquet")]
    for path in (model, server, dataset):
        path.write_bytes(b"synthetic, not real model/runtime/data")
    artifact = ModelArtifact(
        repo_id="synthetic/model", revision="f" * 40, filename=model.name, format="gguf",
        local_path=model, sha256="d" * 64,
    )
    artifact_json = tmp_path / "input.json"
    artifact_json.write_text(artifact.model_dump_json(), encoding="utf-8")
    return QualityRunRequest(
        artifact_json=artifact_json, dataset_file=dataset, llama_server=server,
        image="synthetic:local", output=tmp_path / "new run", pilot_root=root,
    )


def _record() -> dict[str, Any]:
    return synthetic_full_record() | {"classification": "plumbing"}


def _advisor(tmp_path: Path) -> AdvisorService:
    return AdvisorService(services=None, quality_store=QualityEvidenceStore(tmp_path))  # type: ignore[arg-type]


@pytest.mark.skipif(os.name != "posix", reason="Host/container execution requires Linux or WSL")
def test_runner_preserves_inputs_and_uses_existing_producer_before_storing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    before = request.artifact_json.read_bytes()
    calls: list[str] = []

    def invoke(command: list[str], root: Path, log: Path) -> None:
        assert root == request.pilot_root
        calls.append(command[2])
        output = Path(command[command.index("--output") + 1])
        if command[2] == "scripts.quality_eval_smoke":
            artifact_path = Path(command[command.index("--artifact-json") + 1])
            assert artifact_path != request.artifact_json
            assert Path(json.loads(artifact_path.read_text())["local_path"]).is_absolute()
            assert command[command.index("--profile") + 1] == "smoke"
            assert command[command.index("--image") + 1] == request.image
            output.mkdir()
        else:
            assert command[2] == "pilot.quality_eval.records"
            assert (request.output / "bundle").is_dir()
            save_record(output, _record())

    monkeypatch.setattr(runner, "_invoke", invoke)
    advisor = _advisor(tmp_path / "store")
    stored = advisor.run_quality_evaluation(request)
    assert advisor.load_quality_record(stored.stem) == _record()
    assert advisor.lookup_quality(_record()["identity"]) is None
    assert calls == ["scripts.quality_eval_smoke", "pilot.quality_eval.records"]
    assert request.artifact_json.read_bytes() == before
    with pytest.raises(FileExistsError):
        advisor.run_quality_evaluation(request)
    assert len(calls) == 2  # No rerun, overwrite or automatic cache hit.


@pytest.mark.parametrize("stage", ["scripts.quality_eval_smoke", "pilot.quality_eval.records"])
@pytest.mark.skipif(os.name != "posix", reason="Host/container execution requires Linux or WSL")
def test_failed_execution_or_snapshot_is_never_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    request = _request(tmp_path)

    def invoke(command: list[str], root: Path, log: Path) -> None:
        if command[2] == stage:
            log.write_text("synthetic retained error", encoding="utf-8")
            raise QualityEvaluationError("synthetic failure")

    monkeypatch.setattr(runner, "_invoke", invoke)
    advisor = _advisor(tmp_path / "store")
    with pytest.raises(QualityEvaluationError):
        advisor.run_quality_evaluation(request)
    assert advisor.list_quality_ids() == []
    assert (request.output / ("run.log" if stage.endswith("smoke") else "snapshot.log")).exists()


@pytest.mark.parametrize("mutation", ["artifact", "grade", "checksum", "partial"])
@pytest.mark.skipif(os.name != "posix", reason="Host/container execution requires Linux or WSL")
def test_mismatched_or_corrupt_snapshot_cannot_be_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    request = _request(tmp_path)

    def invoke(command: list[str], root: Path, log: Path) -> None:
        if command[2] != "pilot.quality_eval.records":
            return
        record = _record()
        if mutation == "artifact":
            record["identity"]["artifact_sha256"] = "5" * 64
            record["identity_sha256"] = digest(record["identity"])
        elif mutation == "grade":
            record["classification"] = "full"
        path = Path(command[command.index("--output") + 1])
        save_record(path, record)
        if mutation in ("checksum", "partial"):
            contents = json.loads(path.read_text())
            if mutation == "checksum":
                contents["record_sha256"] = "0" * 64
            else:
                contents["record"]["status"] = "partial"
                contents["record_sha256"] = digest(contents["record"])
            path.write_text(json.dumps(contents), encoding="utf-8")

    monkeypatch.setattr(runner, "_invoke", invoke)
    advisor = _advisor(tmp_path / "store")
    with pytest.raises((QualityEvaluationError, ValueError)):
        advisor.run_quality_evaluation(request)
    assert advisor.list_quality_ids() == []


@pytest.mark.skipif(os.name != "posix", reason="Host/container execution requires Linux or WSL")
def test_missing_checkout_or_local_inputs_fail_before_starting_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(runner, "_invoke", lambda *args: pytest.fail("Unexpected launch"))
    for bad in (
        replace(request, pilot_root=tmp_path / "missing"),
        replace(request, dataset_file=tmp_path / "missing.parquet"),
    ):
        with pytest.raises(QualityEvaluationError):
            runner.run_quality_evaluation(bad)
    assert not request.output.exists()


def test_native_windows_execution_fails_clearly_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runner, "_invoke", lambda *args: pytest.fail("Unexpected launch"))
    with pytest.raises(QualityEvaluationError, match="requires Linux or WSL"):
        runner.run_quality_evaluation(request)
    assert not request.output.exists()


def test_subprocess_failure_and_cancellation_allow_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    waits: list[bool] = []

    class Child:
        interrupt = False

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            assert kwargs["start_new_session"] is True
            assert kwargs["cwd"] == tmp_path
            assert kwargs["env"]["PYTHONPATH"].startswith(str(tmp_path))
            assert "shell" not in kwargs

        def __enter__(self) -> Child:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def wait(self) -> int:
            waits.append(True)
            if self.interrupt and len(waits) == 1:
                raise KeyboardInterrupt
            return 1

        def send_signal(self, value: int) -> None:
            signals.append(value)

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(QualityEvaluationError, match="exited with 1"):
        runner._invoke(["synthetic", "argument with spaces"], tmp_path, tmp_path / "failed.log")
    Child.interrupt = True
    waits.clear()
    with pytest.raises(KeyboardInterrupt):
        runner._invoke(["synthetic"], tmp_path, tmp_path / "cancelled.log")
    assert signals == [signal.SIGINT]
    assert len(waits) == 2  # Wait for the pilot's finally; never forcibly kill its owner.


def test_process_bridge_captures_real_child_output_without_a_gpu(tmp_path: Path) -> None:
    log = tmp_path / "python.log"
    with pytest.raises(QualityEvaluationError, match="exited with 7"):
        runner._invoke(
            [sys.executable, "-c", "print('synthetic child output'); raise SystemExit(7)"],
            tmp_path, log,
        )
    assert "synthetic child output" in log.read_text()


@pytest.mark.skipif(os.name != "posix", reason="Pilot signal handling requires Linux or WSL")
def test_worker_cancellation_waits_for_real_child_cleanup(tmp_path: Path) -> None:
    log = tmp_path / "cancel.log"
    code = (
        "import signal,time; "
        "signal.signal(signal.SIGINT, lambda *a: (print('cleanup', flush=True), exit(0))); "
        "print('ready', flush=True); time.sleep(20)"
    )
    with pytest.raises(runner.QualityEvaluationCancelled):
        runner._invoke(
            [sys.executable, "-c", code], tmp_path, log,
            lambda: log.exists() and "ready" in log.read_text(),
        )
    assert "cleanup" in log.read_text()  # Returned only after the owner handled SIGINT.


def test_cancelled_before_launch_never_creates_output_or_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(runner, "_invoke", lambda *a: pytest.fail("Unexpected launch"))
    advisor = _advisor(tmp_path / "store")
    with pytest.raises(runner.QualityEvaluationCancelled):
        advisor.run_quality_evaluation(request, is_cancelled=lambda: True)
    assert not request.output.exists()
    assert advisor.list_quality_ids() == []


@pytest.mark.skipif(os.name != "posix", reason="Host/container execution requires Linux or WSL")
def test_artifact_manifest_accepts_home_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(_request(tmp_path), artifact_json=Path("~/input.json"))
    monkeypatch.setenv("HOME", str(tmp_path))

    def invoke(command: list[str], root: Path, log: Path) -> None:
        if command[2] == "pilot.quality_eval.records":
            save_record(Path(command[command.index("--output") + 1]), _record())

    monkeypatch.setattr(runner, "_invoke", invoke)
    assert runner.run_quality_evaluation(request) == _record()


@pytest.mark.skipif(os.name != "posix", reason="Host/container execution requires Linux or WSL")
@pytest.mark.parametrize("stage", ["scripts.quality_eval_smoke", "pilot.quality_eval.records"])
def test_cancellation_between_stages_does_not_import_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    request = _request(tmp_path)
    cancel = Event()
    calls: list[str] = []

    def invoke(command: list[str], root: Path, log: Path, is_cancelled: Any) -> None:
        runner._check_cancel(is_cancelled)
        calls.append(command[2])
        if command[2] == "pilot.quality_eval.records":
            save_record(Path(command[command.index("--output") + 1]), _record())
        if command[2] == stage:
            cancel.set()

    monkeypatch.setattr(runner, "_invoke", invoke)
    advisor = _advisor(tmp_path / "store")
    with pytest.raises(runner.QualityEvaluationCancelled):
        advisor.run_quality_evaluation(request, is_cancelled=cancel.is_set)
    assert advisor.list_quality_ids() == []
    assert len(calls) == (1 if stage.endswith("smoke") else 2)


def _args(request: QualityRunRequest) -> list[str]:
    return ["quality", "run", "--artifact", str(request.artifact_json),
            "--dataset-file", str(request.dataset_file),
            "--llama-server", str(request.llama_server),
            "--image", request.image, "--output", str(request.output),
            "--pilot-root", str(request.pilot_root), "--json"]


@pytest.mark.parametrize("fail_second", [False, True])
def test_cli_runs_sequentially_and_preserves_completed_records_if_the_second_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_second: bool,
) -> None:
    request = _request(tmp_path)
    calls: list[QualityRunRequest] = []

    class FakeAdvisor:
        def run_quality_evaluation(self, actual: QualityRunRequest) -> Path:
            calls.append(actual)
            if fail_second and len(calls) == 2:
                raise QualityEvaluationError("second synthetic run failed")
            return tmp_path / ((str(len(calls)) * 64) + ".json")

    monkeypatch.setattr("jaull.cli.quality.AdvisorService.default", lambda: FakeAdvisor())
    args = [*_args(request), "--artifact", str(tmp_path / "second.json")]
    result = CliRunner().invoke(app, args, catch_exceptions=False)
    payload = json.loads(result.stdout)
    assert result.exit_code == (3 if fail_second else 0)
    assert payload["purpose"] == "diagnostic_only"
    assert len(payload["completed"]) == (1 if fail_second else 2)
    assert [item.output.name for item in calls] == ["run-01", "run-02"]
    assert [item.artifact_json for item in calls] == [
        request.artifact_json, tmp_path / "second.json",
    ]
    assert all(item.profile is QualityProfile.SMOKE for item in calls)


def test_cli_unsupported_profile_and_help_do_not_launch_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr("jaull.cli.quality.AdvisorService.default",
                        lambda: pytest.fail("Unexpected evaluation"))
    result = CliRunner().invoke(app, [*_args(request), "--profile", "full"])
    assert result.exit_code == 2
    for command in (["quality", "--help"], ["quality", "run", "--help"]):
        assert CliRunner().invoke(app, command).exit_code == 0


def test_cli_comparison_uses_store_and_withholds_incompatible_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = _advisor(tmp_path / "store")
    left = _record()
    right = _record()
    right["identity"]["artifact_sha256"] = "5" * 64
    right["identity_sha256"] = digest(right["identity"])
    ids = [advisor.save_quality_record(record).stem for record in (left, right)]
    monkeypatch.setattr("jaull.cli.quality.AdvisorService.default", lambda: advisor)
    before = [advisor.load_quality_record(value) for value in ids]
    result = CliRunner().invoke(app, ["quality", "compare", *ids, "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["status"] == "COMPARABLE_PLUMBING" and len(report["per_task"]) == 2
    assert Path(report["provenance"][0]["record_path"]).is_file()
    assert [advisor.load_quality_record(value) for value in ids] == before
    right["identity"]["runtime"]["backend_flags"] = ["--n-gpu-layers", "0"]
    right["identity_sha256"] = digest(right["identity"])
    right_id = advisor.save_quality_record(right).stem
    result = CliRunner().invoke(app, ["quality", "compare", ids[0], right_id, "--json"])
    assert json.loads(result.stdout)["status"] == "NOT_COMPARABLE"
    assert json.loads(result.stdout)["per_task"] == []
    missing = CliRunner().invoke(app, ["quality", "compare", ids[0], "0" * 64, "--json"])
    assert missing.exit_code == 3 and "error" in json.loads(missing.stderr)
