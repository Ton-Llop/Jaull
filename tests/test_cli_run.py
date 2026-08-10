from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from jaull.advisor.service import AdvisorService
from jaull.artifacts.errors import ArtifactDownloadError
from jaull.cli import run as cli_run
from jaull.cli.run import RunOptions, run_model
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import InferenceResult
from jaull.domain.runtime import RuntimeRecommendation
from jaull.exceptions import InvalidModelReferenceError, QuantizationNotFoundError
from jaull.execution.errors import ExecutableNotFoundError, ExecutionError
from jaull.workflow.container import ServiceContainer


def _artifact(**updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "owner/repo",
        "revision": "main",
        "filename": "model-q4_k_m.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": 4,
        "local_path": Path("/tmp/model-q4_k_m.gguf"),
        "sha256": "deadbeef",
        "is_downloaded": True,
        "is_verified": True,
    }
    data.update(updates)
    return ModelArtifact(**data)


class _FakeAdvisor:
    def __init__(
        self,
        artifact: ModelArtifact | None = None,
        *,
        fail_at: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.artifact = artifact or _artifact()
        self.fail_at = fail_at
        self.error = error
        self.operations: list[str] = []
        self.resolved: list[tuple[str, str | None, str | None]] = []
        self.downloaded: list[ModelArtifact] = []
        self.verified: list[tuple[ModelArtifact, bool]] = []
        self.runs: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []

    def resolve_artifact(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        self.operations.append("resolve")
        self.resolved.append((repo_id, quantization, revision))
        self._maybe_fail("resolve")
        return self.artifact

    def download_artifact(
        self,
        artifact: ModelArtifact,
        on_progress: object | None = None,
    ) -> ModelArtifact:
        del on_progress
        self.operations.append("download")
        self.downloaded.append(artifact)
        self._maybe_fail("download")
        downloaded = artifact.model_copy(update={"is_downloaded": True})
        self.artifact = downloaded
        return downloaded

    def verify_artifact(
        self,
        artifact: ModelArtifact,
        *,
        full: bool = False,
    ) -> ModelArtifact:
        self.operations.append("verify")
        self.verified.append((artifact, full))
        self._maybe_fail("verify")
        verified = artifact.model_copy(update={"is_verified": True})
        self.artifact = verified
        return verified

    def run_artifact(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        self.operations.append("run")
        self.runs.append((artifact, prompt, runtime))
        self._maybe_fail("run")
        return InferenceResult(
            text="generated answer",
            duration_seconds=0.2,
            exit_code=0,
            runtime="llama.cpp",
            model_path=artifact.local_path or Path("/tmp/model.gguf"),
        )

    def _maybe_fail(self, stage: str) -> None:
        if self.fail_at == stage and self.error is not None:
            raise self.error


def test_run_model_resolves_downloads_verifies_and_runs_through_advisor(
    capsys: Any,
) -> None:
    artifact = _artifact(is_downloaded=False, is_verified=False)
    advisor = _FakeAdvisor(artifact)
    options = RunOptions(
        quantization="Q4_K_M",
        prompt="Explain local AI",
        revision="abc123",
        llama_cli_path="/opt/llama.cpp/llama-cli",
        context_size=8192,
        n_gpu_layers=12,
        timeout_seconds=42.0,
        full_verify=True,
    )

    exit_code = run_model(
        "https://huggingface.co/owner/repo/tree/main",
        options,
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "generated answer\n"
    assert advisor.operations == ["resolve", "download", "verify", "run"]
    assert advisor.resolved == [("owner/repo", "Q4_K_M", "abc123")]
    assert advisor.downloaded == [artifact]
    assert advisor.verified == [
        (artifact.model_copy(update={"is_downloaded": True}), True)
    ]

    ran_artifact, prompt, runtime = advisor.runs[0]
    assert ran_artifact.is_verified is True
    assert prompt == "Explain local AI"
    assert runtime is not None
    assert [(flag.name, flag.value) for flag in runtime.flags] == [
        ("--ctx-size", "8192"),
        ("--n-gpu-layers", "12"),
    ]


def test_run_model_skips_download_when_artifact_is_already_local(capsys: Any) -> None:
    artifact = _artifact(is_downloaded=True, is_verified=False)
    advisor = _FakeAdvisor(artifact)

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert "generated answer" in capsys.readouterr().out
    assert advisor.operations == ["resolve", "verify", "run"]
    assert advisor.downloaded == []
    assert advisor.verified == [(artifact, False)]


def test_run_model_configures_default_advisor_with_llama_cli_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    advisor = _FakeAdvisor()

    class _AdvisorFactory:
        calls: ClassVar[list[tuple[str | Path | None, float]]] = []

        @staticmethod
        def default(
            *,
            llama_cli_path: str | Path | None = None,
            llama_cli_timeout_seconds: float = 300.0,
        ) -> _FakeAdvisor:
            _AdvisorFactory.calls.append(
                (llama_cli_path, llama_cli_timeout_seconds)
            )
            return advisor

    monkeypatch.setattr(cli_run, "AdvisorService", _AdvisorFactory)

    exit_code = run_model(
        "owner/repo",
        RunOptions(
            quantization=None,
            prompt="Hello",
            llama_cli_path="/custom/llama-cli",
            timeout_seconds=7.5,
        ),
    )

    assert exit_code == 0
    assert "generated answer" in capsys.readouterr().out
    assert _AdvisorFactory.calls == [("/custom/llama-cli", 7.5)]
    assert advisor.operations == ["resolve", "verify", "run"]


def test_advisor_run_artifact_uses_configured_llama_cli_path(tmp_path: Path) -> None:
    fake_llama_cli = tmp_path / "fake-llama-cli"
    fake_llama_cli.write_text("#!/bin/sh\nprintf advisor-output\n", encoding="utf-8")
    fake_llama_cli.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    advisor = AdvisorService(
        services=_unused_services(),
        llama_cli_path=fake_llama_cli,
        llama_cli_timeout_seconds=5.0,
    )

    result = advisor.run_artifact(
        artifact=_artifact(local_path=model_path, size_bytes=model_path.stat().st_size),
        prompt="Hello",
    )

    assert result.text == "advisor-output"
    assert result.model_path == model_path


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (InvalidModelReferenceError("bad reference"), 2),
        (QuantizationNotFoundError("Q2_K", ["Q4_K_M"]), 3),
        (ArtifactDownloadError("download failed"), 4),
    ],
)
def test_run_model_maps_pre_execution_errors_to_exit_codes(
    exc: Exception,
    expected_code: int,
    capsys: Any,
) -> None:
    advisor = _FakeAdvisor(fail_at="resolve", error=exc)

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == expected_code
    assert str(exc) in capsys.readouterr().out


@pytest.mark.parametrize(
    "exc",
    [
        ExecutableNotFoundError("llama-cli missing"),
        ExecutionError("runtime failed"),
    ],
)
def test_run_model_maps_advisor_execution_errors_to_exit_code_5(
    exc: ExecutionError,
    capsys: Any,
) -> None:
    advisor = _FakeAdvisor(fail_at="run", error=exc)

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 5
    assert str(exc) in capsys.readouterr().out


def _unused_services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
