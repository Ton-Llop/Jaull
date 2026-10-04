"""Synthetic infrastructure preparation; no network, Docker or GPU access."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pilot.quality_eval import evaluate, setup
from typer.testing import CliRunner

from jaull.advisor.service import AdvisorService
from jaull.cli.app import app
from jaull.runtime import quality_eval_runner as runner
from tests.test_cli_quality import _request
from tests.test_quality_preparation import _fixture


@pytest.fixture
def synthetic_pins(monkeypatch: pytest.MonkeyPatch) -> bytes:
    data = b"synthetic fixed dataset"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(setup, "DATASET_SHA256", digest)
    monkeypatch.setattr(evaluate, "DATASET_SHA256", digest)
    return data


def test_dataset_consent_hash_publication_and_local_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_pins: bytes,
) -> None:
    path = tmp_path / "dataset" / "validation.parquet"
    calls: list[str] = []

    def download(url: str, **kwargs: Any) -> BytesIO:
        assert url == evaluate.DATASET_URL and evaluate.DATASET_REVISION in url
        assert kwargs == {"timeout": 30}
        assert not path.exists()  # No publication before the complete hash check.
        calls.append(url)
        return BytesIO(synthetic_pins)

    monkeypatch.setattr(setup, "urlopen", download)
    with pytest.raises(ValueError, match="permission"):
        setup.prepare_dataset(path)
    assert not path.parent.exists() and calls == []
    setup.prepare_dataset(path, allow_download=True)
    setup.prepare_dataset(path)  # Verified local bytes work without download consent.
    assert path.read_bytes() == synthetic_pins and len(calls) == 1
    assert list(path.parent.iterdir()) == [path]
    path.write_bytes(b"x" * len(synthetic_pins))
    with pytest.raises(ValueError, match="SHA256"):
        setup.prepare_dataset(path, allow_download=True)
    assert len(calls) == 1  # Corrupt local input is not silently replaced.


@pytest.mark.parametrize("failure", ["digest", "interrupted", "cancelled", "oversized"])
def test_failed_dataset_never_publishes_and_removes_temporary_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_pins: bytes, failure: str,
) -> None:
    path = tmp_path / "validation.parquet"

    class Response(BytesIO):
        calls = 0

        def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if failure in ("interrupted", "cancelled") and self.calls == 1:
                return b"partial dataset bytes"
            if failure == "interrupted":
                raise OSError("Synthetic lost connection")
            if failure == "cancelled":
                raise KeyboardInterrupt
            if failure == "oversized":
                return b"x" * (64 * 1024**2 + 1)
            return super().read(size)

    monkeypatch.setattr(setup, "urlopen", lambda *a, **kw: Response(b"wrong dataset"))
    with pytest.raises((ValueError, OSError, KeyboardInterrupt)):
        setup.prepare_dataset(path, allow_download=True)
    assert not path.exists() and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["none", "runtime", "docker", "image", "image_id"])
def test_infrastructure_blocks_before_a_permitted_dataset_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_pins: bytes, failure: str,
) -> None:
    server = tmp_path / "llama-server"
    server.write_bytes(b"synthetic runtime")
    server.chmod(0o700)
    monkeypatch.setattr(setup, "SERVER_SHA256", hashlib.sha256(server.read_bytes()).hexdigest())
    if failure == "runtime":
        server.write_bytes(b"modified runtime")
    downloads: list[str] = []

    def inspect(command: list[str], **kwargs: Any) -> str:
        assert command == ["docker", "image", "inspect", "synthetic:local"]
        assert kwargs["timeout"] == 15
        if failure == "docker":
            raise subprocess.TimeoutExpired(command, 15)
        labels = {} if failure == "image" else {
            setup.ARTIFACT_CONTRACT_LABEL: setup.ARTIFACT_CONTRACT,
        }
        return json.dumps([{"Id": "bad" if failure == "image_id" else "sha256:" + "d" * 64,
                            "Config": {"Labels": labels}}])

    def download(url: str, **kwargs: Any) -> BytesIO:
        downloads.append(url)
        return BytesIO(synthetic_pins)

    monkeypatch.setattr(setup.subprocess, "check_output", inspect)
    monkeypatch.setattr(setup, "urlopen", download)
    dataset = tmp_path / "validation.parquet"
    if failure == "none":
        image = setup.prepare_infrastructure(
            dataset, server, "synthetic:local", allow_download=True,
        )
        assert image["Id"] == "sha256:" + "d" * 64
        assert dataset.read_bytes() == synthetic_pins and len(downloads) == 1
    else:
        with pytest.raises(ValueError):
            setup.prepare_infrastructure(dataset, server, "synthetic:local", allow_download=True)
        assert not dataset.exists() and downloads == []


@pytest.mark.parametrize("reply", ["ready", "blocked", "malformed", "bad_json", "absent"])
@pytest.mark.skipif(os.name != "posix", reason="Pilot bridge requires Linux or WSL")
def test_shared_bridge_checks_without_evaluation_and_preserves_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str,
) -> None:
    request = _request(tmp_path / "inputs")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    commands: list[list[str]] = []

    def invoke(command: list[str], root: Path, log: Path, *args: Any) -> None:
        commands.append(command)
        assert command[2] == "pilot.quality_eval.setup" and root == request.pilot_root
        report = Path(command[command.index("--output") + 1])
        if reply == "absent":
            return
        if reply == "bad_json":
            report.write_text("not JSON")
            return
        if reply == "blocked":
            report.write_text(json.dumps({"status": "blocked", "error": "Pinned runtime differs"}))
            raise runner.QualityEvaluationError("Synthetic exit")
        report.write_text(json.dumps({"status": "ready", "image_id": (
            "bad" if reply == "malformed" else "sha256:" + "d" * 64
        )}))

    monkeypatch.setattr(runner, "_invoke", invoke)
    if reply == "ready":
        runner.prepare_quality_setup(request)
        assert "--allow-dataset-download" not in commands[-1]
        runner.prepare_quality_setup(replace(request, allow_dataset_download=True))
        assert "--allow-dataset-download" in commands[-1]
    else:
        with pytest.raises(runner.QualityEvaluationError, match=(
            "Pinned runtime differs" if reply == "blocked" else
            "Missing or invalid" if reply in ("bad_json", "absent") else "Invalid infrastructure"
        )):
            runner.prepare_quality_setup(request)
    assert not request.output.exists()  # No model run, record or output bundle.


def test_setup_failure_stops_selected_model_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise runner.QualityEvaluationError("Evaluator image unavailable")

    monkeypatch.setattr(AdvisorService, "prepare_quality_evaluation_setup", blocked)
    with pytest.raises(runner.QualityEvaluationError, match="image unavailable"):
        advisor.run_quality_evaluation_for_plan(plan, _request(tmp_path / "inputs"),
                                                allow_download=True)
    assert calls == []


def test_defaults_use_source_location_and_user_data_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    defaults = runner.quality_setup_defaults()
    assert defaults["root"] == str(Path(__file__).resolve().parents[1])
    assert defaults["dataset"].startswith(str(tmp_path / "user"))
    assert ".codex-night" not in defaults["dataset"]
    assert not Path(defaults["dataset"]).exists()


def test_wheel_defaults_never_execute_a_checkout_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "cwd"
    _request(cwd)  # A complete-looking unrelated checkout, still not selected.
    monkeypatch.chdir(cwd / "trusted checkout")
    monkeypatch.setattr(runner, "__file__", str(
        tmp_path / "installed" / "jaull" / "runtime" / "quality_eval_runner.py"
    ))
    assert "root" not in runner.quality_setup_defaults()


def test_dataset_deadline_does_not_publish_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_pins: bytes,
) -> None:
    ticks = iter((0, 121))
    monkeypatch.setattr(setup.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(setup, "urlopen", lambda *a, **kw: BytesIO(synthetic_pins))
    with pytest.raises(TimeoutError, match="120 seconds"):
        setup.prepare_dataset(tmp_path / "validation.parquet", allow_download=True)
    assert list(tmp_path.iterdir()) == []


def test_existing_output_stops_before_setup_or_model_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)
    request = _request(tmp_path / "inputs")
    request.output.mkdir()
    monkeypatch.setattr(AdvisorService, "prepare_quality_evaluation_setup",
                        lambda *a, **kw: pytest.fail("No setup for an occupied run output"))
    with pytest.raises(FileExistsError):
        advisor.run_quality_evaluation_for_plan(plan, request, allow_download=True)
    assert calls == []


@pytest.mark.parametrize("allow_download", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_cli_setup_remembers_only_success_and_never_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_download: bool, failed: bool,
) -> None:
    requests: list[runner.QualityRunRequest] = []
    saved: list[runner.QualityRunRequest] = []
    request = _request(tmp_path)

    class Advisor:
        def quality_evaluation_setup(self) -> dict[str, str]:
            return {"dataset": str(request.dataset_file), "server": str(request.llama_server),
                    "root": str(request.pilot_root), "image": request.image}

        def prepare_quality_evaluation_setup(self, actual: runner.QualityRunRequest) -> None:
            requests.append(actual)
            if failed:
                raise runner.QualityEvaluationError("Synthetic blocked setup")

        def remember_quality_evaluation_setup(self, actual: runner.QualityRunRequest) -> None:
            saved.append(actual)

    monkeypatch.setattr("jaull.cli.quality.AdvisorService.default", lambda: Advisor())
    args = ["quality", "setup", "--json"]
    if allow_download:
        args.append("--allow-dataset-download")
    result = CliRunner().invoke(app, args, catch_exceptions=False)
    assert result.exit_code == (3 if failed else 0)
    assert json.loads(result.stdout)["status"] == ("blocked" if failed else "ready")
    assert requests[0].allow_dataset_download is allow_download
    assert requests[0].artifact_json is None
    assert saved == ([] if failed else requests)
