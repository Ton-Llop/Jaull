from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from jaull.artifacts.errors import ArtifactDownloadError
from jaull.cli import run as cli_run
from jaull.cli.run import RunOptions, run_model
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import InferenceResult
from jaull.exceptions import InvalidModelReferenceError, QuantizationNotFoundError
from jaull.execution.errors import ExecutableNotFoundError
from jaull.runtime.llama_cpp_runner import LlamaCppRunnerError


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
    def __init__(self, artifact: ModelArtifact | None = None) -> None:
        self.artifact = artifact or _artifact()
        self.resolved: list[tuple[str, str | None, str | None]] = []
        self.downloaded: list[ModelArtifact] = []
        self.verified: list[tuple[ModelArtifact, bool]] = []

    def resolve_artifact(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        self.resolved.append((repo_id, quantization, revision))
        return self.artifact

    def download_artifact(
        self,
        artifact: ModelArtifact,
        on_progress: object | None = None,
    ) -> ModelArtifact:
        del on_progress
        self.downloaded.append(artifact)
        downloaded = artifact.model_copy(update={"is_downloaded": True})
        self.artifact = downloaded
        return downloaded

    def verify_artifact(
        self,
        artifact: ModelArtifact,
        *,
        full: bool = False,
    ) -> ModelArtifact:
        self.verified.append((artifact, full))
        verified = artifact.model_copy(update={"is_verified": True})
        self.artifact = verified
        return verified


class _FakeRunner:
    instances: ClassVar[list[_FakeRunner]] = []
    error: ClassVar[Exception | None] = None

    def __init__(
        self,
        *,
        backend: object,
        llama_cli_path: str | Path | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.backend = backend
        self.llama_cli_path = llama_cli_path
        self.timeout_seconds = timeout_seconds
        self.calls: list[tuple[ModelArtifact, str, object]] = []
        _FakeRunner.instances.append(self)

    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: object | None = None,
    ) -> InferenceResult:
        self.calls.append((artifact, prompt, runtime))
        if _FakeRunner.error is not None:
            raise _FakeRunner.error
        return InferenceResult(
            text="generated answer",
            duration_seconds=0.2,
            exit_code=0,
            runtime="llama.cpp",
            model_path=artifact.local_path or Path("/tmp/model.gguf"),
        )


@pytest.fixture(autouse=True)
def _fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRunner.instances = []
    _FakeRunner.error = None
    monkeypatch.setattr(cli_run, "LlamaCppRunner", _FakeRunner)


def test_run_model_resolves_downloads_verifies_and_runs(capsys: Any) -> None:
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
    assert advisor.resolved == [("owner/repo", "Q4_K_M", "abc123")]
    assert advisor.downloaded == [artifact]
    assert advisor.verified == [
        (artifact.model_copy(update={"is_downloaded": True}), True)
    ]

    runner = _FakeRunner.instances[0]
    assert runner.llama_cli_path == "/opt/llama.cpp/llama-cli"
    assert runner.timeout_seconds == 42.0
    ran_artifact, prompt, runtime = runner.calls[0]
    assert ran_artifact.is_verified is True
    assert prompt == "Explain local AI"
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
    assert advisor.downloaded == []
    assert advisor.verified == [(artifact, False)]


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (InvalidModelReferenceError("bad reference"), 2),
        (QuantizationNotFoundError("Q2_K", ["Q4_K_M"]), 3),
        (ArtifactDownloadError("download failed"), 4),
        (ExecutableNotFoundError("llama-cli missing"), 5),
        (LlamaCppRunnerError("runtime failed"), 5),
    ],
)
def test_run_model_maps_known_errors_to_exit_codes(
    exc: Exception,
    expected_code: int,
    capsys: Any,
) -> None:
    class _FailingAdvisor(_FakeAdvisor):
        def resolve_artifact(
            self,
            repo_id: str,
            quantization: str | None = None,
            revision: str | None = None,
        ) -> ModelArtifact:
            del repo_id, quantization, revision
            raise exc

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello"),
        advisor=_FailingAdvisor(),  # type: ignore[arg-type]
    )

    assert exit_code == expected_code
    assert str(exc) in capsys.readouterr().out


def test_run_model_maps_runner_errors_to_execution_exit_code(capsys: Any) -> None:
    _FakeRunner.error = LlamaCppRunnerError("bad runtime flag")

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello"),
        advisor=_FakeAdvisor(),  # type: ignore[arg-type]
    )

    assert exit_code == 5
    assert "bad runtime flag" in capsys.readouterr().out
