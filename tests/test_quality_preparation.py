"""Synthetic automatic preparation; no Hub, Docker, GPU or real model access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.artifacts.errors import ArtifactVerificationError
from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import ExecutionPlan
from jaull.runtime.quality_eval_runner import (
    QualityEvaluationCancelled,
    QualityEvaluationError,
    load_quality_setup,
    save_quality_setup,
)
from tests.test_cli_quality import _request
from tests.test_tui_evidence import _plan


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[
    AdvisorService, ExecutionPlan, list[str], list[ModelArtifact],
]:
    data = b"synthetic model bytes"
    digest = hashlib.sha256(data).hexdigest()
    plan = _plan(sha256=digest).model_copy(update={
        "artifact": _plan(sha256=digest).artifact.model_copy(update={
            "revision": "f" * 40, "size_bytes": len(data), "file_count": 1,
        }),
    })
    calls: list[str] = []
    manifests: list[ModelArtifact] = []

    class Resolver:
        def resolve(self, *args: Any, **kwargs: Any) -> ModelArtifact:
            calls.append("resolve")
            return plan.artifact.to_model_artifact().model_copy(update={"sha256": digest})

    def download(**kwargs: Any) -> str:
        calls.append("download")
        assert kwargs["revision"] == "f" * 40
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(data)
        return str(path)

    service = ArtifactService(Resolver(), ArtifactStorage(tmp_path / "models"), download)
    advisor = AdvisorService(services=None, artifacts=service)  # type: ignore[arg-type]
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))

    def preflight(command: list[str], root: Path, log: Path, *args: Any) -> None:
        assert command[2] == "pilot.quality_eval.setup"
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "status": "ready", "image_id": "sha256:" + "c" * 64,
        }))

    monkeypatch.setattr("jaull.runtime.quality_eval_runner._invoke", preflight)
    # Artifact preparation tests isolate the optional Linux/WSL execution boundary.
    monkeypatch.setattr("jaull.runtime.quality_eval_runner._pilot_root",
                        lambda request: request.pilot_root)

    def evaluate(self: AdvisorService, request: Any, **kwargs: Any) -> Path:
        assert request.artifact_json is not None
        artifact = ModelArtifact.model_validate_json(request.artifact_json.read_text())
        assert artifact.is_verified and artifact.sha256 == digest
        assert artifact.local_path is not None and artifact.local_path.read_bytes() == data
        manifests.append(artifact)
        calls.append("evaluate")
        return tmp_path / "completed.json"

    monkeypatch.setattr(AdvisorService, "run_quality_evaluation", evaluate)
    return advisor, plan, calls, manifests


def test_missing_selected_gguf_requires_download_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)
    request = _request(tmp_path / "inputs")
    with pytest.raises(QualityEvaluationError, match="download permission"):
        advisor.run_quality_evaluation_for_plan(plan, request)
    assert calls == []
    result = advisor.run_quality_evaluation_for_plan(plan, request, allow_download=True)
    assert result.name == "completed.json"
    assert calls == ["download", "evaluate"]


def test_local_selected_artifact_is_fully_verified_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, manifests = _fixture(tmp_path, monkeypatch)
    request = _request(tmp_path / "inputs")
    advisor.run_quality_evaluation_for_plan(plan, request, allow_download=True)
    calls.clear()
    advisor.run_quality_evaluation_for_plan(plan, request)
    assert calls == ["evaluate"]
    assert manifests[0] == manifests[1]
    path = manifests[0].local_path
    assert path is not None
    # Same size and old sidecar: preparation must hash the real bytes.
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        advisor.run_quality_evaluation_for_plan(plan, request)
    assert calls == ["evaluate"]


def test_main_is_pinned_and_selection_is_checked_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, manifests = _fixture(tmp_path, monkeypatch)
    mutable = plan.model_copy(update={"artifact": plan.artifact.model_copy(
        update={"revision": "main"},
    )})
    request = _request(tmp_path / "inputs")
    advisor.run_quality_evaluation_for_plan(mutable, request, allow_download=True)
    assert calls == ["resolve", "download", "evaluate"]
    assert manifests[0].revision == "f" * 40
    calls.clear()
    mismatch = mutable.model_copy(update={"artifact": mutable.artifact.model_copy(
        update={"filename": "other.gguf"},
    )})
    with pytest.raises(ValueError, match="does not match"):
        advisor.run_quality_evaluation_for_plan(mismatch, request, allow_download=True)
    assert calls == ["resolve"]


def test_main_without_published_digest_cannot_inherit_a_local_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)
    request = _request(tmp_path / "inputs")
    advisor.run_quality_evaluation_for_plan(plan, request, allow_download=True)
    calls.clear()
    assert advisor.artifacts is not None
    # Valid cached bytes/sidecar, but the new commit has no published digest.
    monkeypatch.setattr(advisor.artifacts.resolver, "resolve", lambda *args, **kwargs:
                        plan.artifact.to_model_artifact())
    mutable = plan.model_copy(update={"artifact": plan.artifact.model_copy(
        update={"revision": "main", "sha256": None},
    )})
    with pytest.raises(QualityEvaluationError, match="published SHA256"):
        advisor.run_quality_evaluation_for_plan(mutable, request, allow_download=True)
    assert calls == []


def test_preparation_rejects_multipart_missing_inputs_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)
    request = _request(tmp_path / "inputs")
    multipart = plan.model_copy(update={"artifact": plan.artifact.model_copy(
        update={"file_count": 3},
    )})
    with pytest.raises(ValueError, match="Multipart"):
        advisor.run_quality_evaluation_for_plan(multipart, request, allow_download=True)
    with pytest.raises(QualityEvaluationError, match="input missing"):
        advisor.run_quality_evaluation_for_plan(
            plan, replace(request, dataset_file=tmp_path / "missing"), allow_download=True,
        )
    with pytest.raises(QualityEvaluationCancelled):
        advisor.run_quality_evaluation_for_plan(
            plan, request, allow_download=True, is_cancelled=lambda: True,
        )
    assert calls == []


@pytest.mark.parametrize("field,value", [("sha256", None), ("size_bytes", None),
                                         ("revision", "release-tag")])
def test_unknown_identity_never_downloads_or_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)
    changed = plan.model_copy(update={"artifact": plan.artifact.model_copy(
        update={field: value},
    )})
    with pytest.raises(QualityEvaluationError):
        advisor.run_quality_evaluation_for_plan(
            changed, _request(tmp_path / "inputs"), allow_download=True,
        )
    assert calls == []


def test_cancellation_after_preparation_never_starts_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, plan, calls, _ = _fixture(tmp_path, monkeypatch)
    cancelled = False

    def progress(message: str) -> None:
        nonlocal cancelled
        if message == "Fully verifying selected GGUF":
            cancelled = True

    with pytest.raises(QualityEvaluationCancelled):
        advisor.run_quality_evaluation_for_plan(
            plan, _request(tmp_path / "inputs"), allow_download=True,
            is_cancelled=lambda: cancelled, on_progress=progress,
        )
    assert calls == ["download"]


def test_quality_setup_remembers_only_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    assert load_quality_setup() == {}
    request = replace(_request(tmp_path / "inputs"), allow_dataset_download=True)
    save_quality_setup(request)
    setup = load_quality_setup()
    assert set(setup) == {"dataset", "server", "root", "image"}
    assert setup["dataset"] == str(request.dataset_file.resolve())
    assert setup["image"] == request.image
    changed = replace(request, image="another-image")
    save_quality_setup(changed)
    assert load_quality_setup()["image"] == "another-image"


@pytest.mark.parametrize("payload", [{}, [], {"schema_version": 2},
                                   {"schema_version": 1, "dataset": 1},
                                   {"schema_version": True, "dataset": "d", "server": "s",
                                    "root": "r", "image": "i"}])
def test_invalid_quality_setup_is_reported_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: Any,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = tmp_path / "jaull" / "quality-setup.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    with pytest.raises(QualityEvaluationError, match="Invalid quality setup"):
        load_quality_setup()
