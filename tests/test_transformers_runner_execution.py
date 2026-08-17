from __future__ import annotations

from pathlib import Path

import pytest

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution import ExecutionObservation, ExecutionRequest, ExecutionResult
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.errors import ExecutableNotFoundError, ExecutionFailedError
from jaull.runtime.transformers_runner import (
    InvalidTransformersArtifactError,
    InvalidTransformersPromptError,
    TransformersRunner,
    TransformersRunnerError,
)


def _observation(
    *,
    duration_seconds: float = 1.25,
    exit_code: int = 0,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=exit_code == 0,
        duration_seconds=duration_seconds,
        peak_ram_bytes=None,
        peak_vram_bytes=None,
        exit_code=exit_code,
        failure_reason=None,
    )


class _FakeExecutionBackend:
    def __init__(
        self,
        result: ExecutionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[ExecutionRequest] = []
        self.result = result or ExecutionResult(
            stdout='{"success": true, "text": "generated text"}\n',
            stderr="diagnostic",
            observation=_observation(duration_seconds=1.25),
        )
        self.error = error

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "bin dir" / "python exe"
    path.parent.mkdir()
    path.write_text("# test executable placeholder\n", encoding="utf-8")
    return path


def _artifact(**updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "main",
        "filename": "Qwen/Qwen2.5-0.5B-Instruct",
        "format": "safetensors",
        "quantization": None,
        "size_bytes": None,
        "local_path": None,
        "sha256": None,
        "is_downloaded": True,
        "is_verified": True,
    }
    data.update(updates)
    return ModelArtifact(**data)


def _runtime(
    *,
    device_map: str = "cpu",
    torch_dtype: str = "torch.float32",
    max_new_tokens: int = 12,
    runtime: RuntimeName = RuntimeName.TRANSFORMERS,
) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=runtime,
        flags=[
            RuntimeFlag(
                name="device_map",
                value=device_map,
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            ),
            RuntimeFlag(
                name="torch_dtype",
                value=torch_dtype,
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            ),
            RuntimeFlag(
                name="max_new_tokens",
                value=str(max_new_tokens),
                source=RuntimeFlagSource.USER_INPUT,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def test_runner_builds_isolated_worker_command_from_repo_ref(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend()
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    result = runner.run(
        artifact=_artifact(),
        prompt="Explain mutexes: caf\u00e9",
        runtime=_runtime(device_map="cpu", torch_dtype="torch.float32"),
    )

    assert result.text == "generated text"
    assert result.runtime == RuntimeName.TRANSFORMERS.value
    command = backend.requests[0].command
    assert command == (
        str(runner._python),
        "-m",
        "jaull.runtime.transformers_worker",
        "--model-ref",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--prompt",
        "Explain mutexes: caf\u00e9",
        "--device-map",
        "cpu",
        "--max-new-tokens",
        "12",
        "--revision",
        "main",
        "--torch-dtype",
        "torch.float32",
    )
    assert "PYTHONPATH" in backend.requests[0].environment
    assert backend.requests[0].environment["PYTHONPATH"]


def test_runner_uses_local_model_directory_with_spaces(tmp_path: Path) -> None:
    model_dir = tmp_path / "models with spaces" / "qwen local"
    model_dir.mkdir(parents=True)
    backend = _FakeExecutionBackend()
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    result = runner.run(
        artifact=_artifact(local_path=model_dir),
        prompt="Hello",
        runtime=_runtime(),
    )

    assert result.model_path == model_dir
    command = backend.requests[0].command
    assert str(model_dir) in command


def test_runner_defaults_to_cpu_and_small_generation(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend()
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    runner.run(artifact=_artifact(), prompt="Hello")

    command = backend.requests[0].command
    assert command[command.index("--device-map") + 1] == "cpu"
    assert command[command.index("--max-new-tokens") + 1] == "64"


def test_runner_parses_last_json_line_after_startup_logging(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend(
        ExecutionResult(
            stdout='loading model\n{"success": true, "text": "final answer"}\n',
            stderr="",
            observation=_observation(duration_seconds=0.2),
        )
    )
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    result = runner.run(artifact=_artifact(), prompt="Hello")

    assert result.text == "final answer"


def test_runner_rejects_empty_prompt(tmp_path: Path) -> None:
    runner = TransformersRunner(
        backend=_FakeExecutionBackend(), python_executable=_executable(tmp_path)
    )

    with pytest.raises(InvalidTransformersPromptError):
        runner.run(artifact=_artifact(), prompt=" ")


def test_runner_rejects_gguf_artifact(tmp_path: Path) -> None:
    runner = TransformersRunner(
        backend=_FakeExecutionBackend(), python_executable=_executable(tmp_path)
    )

    with pytest.raises(InvalidTransformersArtifactError):
        runner.run(artifact=_artifact(format="gguf"), prompt="Hello")


def test_runner_rejects_single_file_local_artifact(tmp_path: Path) -> None:
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights")
    runner = TransformersRunner(
        backend=_FakeExecutionBackend(), python_executable=_executable(tmp_path)
    )

    with pytest.raises(InvalidTransformersArtifactError):
        runner.run(artifact=_artifact(local_path=model_file), prompt="Hello")


def test_runner_rejects_non_transformers_runtime(tmp_path: Path) -> None:
    runner = TransformersRunner(
        backend=_FakeExecutionBackend(), python_executable=_executable(tmp_path)
    )

    with pytest.raises(TransformersRunnerError):
        runner.run(
            artifact=_artifact(),
            prompt="Hello",
            runtime=_runtime(runtime=RuntimeName.LLAMA_CPP),
        )


def test_runner_rejects_malformed_worker_output(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend(
        ExecutionResult(
            stdout="not json\n",
            stderr="",
            observation=_observation(duration_seconds=0.2),
        )
    )
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    with pytest.raises(TransformersRunnerError):
        runner.run(artifact=_artifact(), prompt="Hello")


def test_runner_rejects_worker_reported_failure(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend(
        ExecutionResult(
            stdout='{"success": false, "error": "model load failed"}\n',
            stderr="",
            observation=_observation(duration_seconds=0.2),
        )
    )
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    with pytest.raises(TransformersRunnerError, match="model load failed"):
        runner.run(artifact=_artifact(), prompt="Hello")


def test_runner_surfaces_worker_failure_from_non_zero_exit(tmp_path: Path) -> None:
    failed_result = ExecutionResult(
        stdout='{"success": false, "error": "transformers import failed"}\n',
        stderr="Traceback...\nImportError: transformers import failed\n",
        observation=_observation(duration_seconds=0.2, exit_code=1),
    )
    backend = _FakeExecutionBackend(
        error=ExecutionFailedError(
            "Command failed with exit code 1: '/venv/bin/python'.",
            failed_result,
        )
    )
    runner = TransformersRunner(backend=backend, python_executable=_executable(tmp_path))

    with pytest.raises(TransformersRunnerError, match="transformers import failed"):
        runner.run(artifact=_artifact(), prompt="Hello")


def test_runner_reports_missing_python_executable(tmp_path: Path) -> None:
    with pytest.raises(ExecutableNotFoundError):
        TransformersRunner(
            backend=_FakeExecutionBackend(),
            python_executable=tmp_path / "missing-python",
        )
