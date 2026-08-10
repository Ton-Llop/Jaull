from __future__ import annotations

from pathlib import Path

import pytest

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution import ExecutionRequest, ExecutionResult
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.errors import ExecutableNotFoundError
from jaull.runtime.llama_cpp_runner import (
    InvalidLlamaCppArtifactError,
    InvalidPromptError,
    LlamaCppRunner,
)


class _FakeExecutionBackend:
    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.requests: list[ExecutionRequest] = []
        self.result = result or ExecutionResult(
            exit_code=0,
            stdout="generated text",
            stderr="diagnostic",
            duration_seconds=1.25,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return self.result


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "bin dir" / "llama cli"
    path.parent.mkdir()
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return path


def _artifact(path: Path, **updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "owner/repo",
        "revision": "main",
        "filename": "placeholder-name.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": path.stat().st_size if path.exists() else None,
        "local_path": path,
        "sha256": "deadbeef",
        "is_downloaded": True,
        "is_verified": True,
    }
    data.update(updates)
    return ModelArtifact(**data)


def _runtime(*, ctx_size: int = 2048, n_gpu_layers: int = 0) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(ctx_size),
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value=str(n_gpu_layers),
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def test_runner_builds_command_with_exact_local_path(tmp_path: Path) -> None:
    model_path = tmp_path / "models with spaces" / "real model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend()
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    result = runner.run(
        artifact=_artifact(model_path),
        prompt="Explain GGUF: caf\u00e9",
        runtime=_runtime(ctx_size=8192, n_gpu_layers=12),
    )

    assert result.text == "generated text"
    assert result.model_path == model_path
    command = backend.requests[0].command
    assert command == (
        str(runner._llama_cli),
        "--model",
        str(model_path),
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "12",
        "--prompt",
        "Explain GGUF: caf\u00e9",
    )
    assert "placeholder-name.gguf" not in command


def test_runner_defaults_to_cpu_gpu_layers(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend()
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    runner.run(artifact=_artifact(model_path), prompt="Hello")

    command = backend.requests[0].command
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--n-gpu-layers") + 1] == "0"


def test_runner_rejects_safetensors(tmp_path: Path) -> None:
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"weights")
    runner = LlamaCppRunner(
        backend=_FakeExecutionBackend(),
        llama_cli_path=_executable(tmp_path),
    )

    with pytest.raises(InvalidLlamaCppArtifactError):
        runner.run(
            artifact=_artifact(model_path, format="safetensors"),
            prompt="Hello",
        )


def test_runner_rejects_not_downloaded_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    runner = LlamaCppRunner(
        backend=_FakeExecutionBackend(),
        llama_cli_path=_executable(tmp_path),
    )

    with pytest.raises(InvalidLlamaCppArtifactError):
        runner.run(
            artifact=_artifact(model_path, is_downloaded=False),
            prompt="Hello",
        )


def test_runner_rejects_not_verified_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    runner = LlamaCppRunner(
        backend=_FakeExecutionBackend(),
        llama_cli_path=_executable(tmp_path),
    )

    with pytest.raises(InvalidLlamaCppArtifactError):
        runner.run(
            artifact=_artifact(model_path, is_verified=False),
            prompt="Hello",
        )


def test_runner_rejects_missing_local_file(tmp_path: Path) -> None:
    runner = LlamaCppRunner(
        backend=_FakeExecutionBackend(),
        llama_cli_path=_executable(tmp_path),
    )

    with pytest.raises(InvalidLlamaCppArtifactError):
        runner.run(
            artifact=_artifact(tmp_path / "missing.gguf"),
            prompt="Hello",
        )


def test_runner_rejects_empty_prompt(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    runner = LlamaCppRunner(
        backend=_FakeExecutionBackend(),
        llama_cli_path=_executable(tmp_path),
    )

    with pytest.raises(InvalidPromptError):
        runner.run(artifact=_artifact(model_path), prompt="  ")


def test_runner_reports_executable_not_found(tmp_path: Path) -> None:
    with pytest.raises(ExecutableNotFoundError):
        LlamaCppRunner(
            backend=_FakeExecutionBackend(),
            llama_cli_path=tmp_path / "missing llama-cli",
        )
