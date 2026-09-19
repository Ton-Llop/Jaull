from __future__ import annotations

from pathlib import Path

import pytest

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution import (
    ExecutionObservation,
    ExecutionRequest,
    ExecutionResult,
    RuntimeBufferCategory,
)
from jaull.domain.hardware import ComputeBackend
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
    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.requests: list[ExecutionRequest] = []
        self.result = result or ExecutionResult(
            stdout="generated text",
            stderr="diagnostic",
            observation=_observation(duration_seconds=1.25),
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


def _runtime(
    *, ctx_size: int = 2048, n_gpu_layers: int = 0, device: str | None = None
) -> RuntimeRecommendation:
    flags = [
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
    ]
    if device is not None:
        flags.append(
            RuntimeFlag(
                name="--device",
                value=device,
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            )
        )
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=flags,
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
        "--no-display-prompt",
        "--color",
        "off",
        "--no-show-timings",
        "--simple-io",
        "--single-turn",
        "--verbose",
        "--prompt",
        "Explain GGUF: caf\u00e9",
    )
    assert "placeholder-name.gguf" not in command
    assert result.command == command


def test_runner_records_cuda_only_when_llama_cpp_reports_it(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend(
        ExecutionResult(
            stdout="generated",
            stderr="ggml_cuda_init: found 1 CUDA devices",
            observation=_observation(),
        )
    )
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    result = runner.run(artifact=_artifact(model_path), prompt="Hello")

    assert result.observed_backend is ComputeBackend.CUDA
    assert result.observed_backend_source == "llama.cpp runtime output"


def test_runner_defaults_to_cpu_gpu_layers(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend()
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    runner.run(artifact=_artifact(model_path), prompt="Hello")

    command = backend.requests[0].command
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--n-gpu-layers") + 1] == "0"


def test_runner_passes_confirmed_runtime_device_to_llama_cpp(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend()
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    runner.run(
        artifact=_artifact(model_path),
        prompt="Hello",
        runtime=_runtime(device="Vulkan0"),
    )

    command = backend.requests[0].command
    assert command[command.index("--device") + 1] == "Vulkan0"


def test_runner_always_asks_for_the_load_log(tmp_path: Path) -> None:
    """Without it llama.cpp writes nothing to stderr and there is no observation.

    Found by re-running B001 on real hardware: the run succeeded, and
    `runtime_allocation` came back `None` because `--verbose` was conditional on
    a flag nothing sets. The whole observation contract was unreachable from the
    path that actually runs models.
    """
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend()
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    # No `--verbose` flag anywhere on the runtime recommendation.
    runner.run(artifact=_artifact(model_path), prompt="Hello", runtime=_runtime())

    assert "--verbose" in backend.requests[0].command


def test_runner_sanitizes_terminal_sequences_from_response(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend(
        ExecutionResult(
            stdout="\x1b[2J\x1b[HGenerated [literal]\x07",
            stderr="",
            observation=_observation(duration_seconds=0.2),
        )
    )
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    result = runner.run(artifact=_artifact(model_path), prompt="Hello")

    assert result.text == "Generated [literal]"


def test_runner_strips_llama_cli_preamble_prompt_and_exit_marker(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend(
        ExecutionResult(
            stdout=(
                "Loading model...\n"
                "build      : b10357\n"
                "available commands:\n\n"
                "> Explain GGUF\n"
                "GGUF is a compact model format.\n\n"
                "Exiting...\n"
            ),
            stderr="",
            observation=_observation(duration_seconds=0.2),
        )
    )
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    result = runner.run(artifact=_artifact(model_path), prompt="Explain GGUF")

    assert result.text == "GGUF is a compact model format."


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


# Verbatim from a real B001 re-run on this project's RTX 2060, llama.cpp
# 689e227db, `--n-gpu-layers 25`. Two properties of the real log that the
# recorded August sweep did not have, and that a hand-written fixture would
# not have guessed:
#
#   * every buffer is printed twice, and the *first* pass reports 0.00 MiB
#     because it is the reserve pass before the weights are read;
#   * the KV lines come from `llama_kv_cache:` and compute from
#     `sched_reserve:`, not from `llama_context:`.
B001_R7_STDERR = """
0.00.488.277 I load_tensors:        CUDA0 model buffer size =     0.00 MiB
0.00.488.278 I load_tensors:    CUDA_Host model buffer size =     0.00 MiB
0.00.492.675 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.00.492.865 I llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.00.492.867 I llama_kv_cache:      CUDA0 KV buffer size =     0.00 MiB
0.00.497.968 I sched_reserve:      CUDA0 compute buffer size =   183.44 MiB
0.00.497.989 I sched_reserve:  CUDA_Host compute buffer size =    18.01 MiB
0.00.795.071 I load_tensors:   CPU_Mapped model buffer size =   844.04 MiB
0.00.795.071 I load_tensors:        CUDA0 model buffer size =  3616.41 MiB
0.04.355.258 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.04.355.400 I llama_kv_cache:        CPU KV buffer size =    32.00 MiB
0.04.369.084 I llama_kv_cache:      CUDA0 KV buffer size =   192.00 MiB
0.04.395.075 I sched_reserve:      CUDA0 compute buffer size =   183.44 MiB
0.04.395.097 I sched_reserve:  CUDA_Host compute buffer size =    18.01 MiB
"""


def test_a_real_load_log_becomes_a_runtime_allocation(tmp_path: Path) -> None:
    """The contract, end to end through the runner, on a real capture."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    backend = _FakeExecutionBackend(
        result=ExecutionResult(
            stdout="generated text",
            stderr=B001_R7_STDERR,
            observation=_observation(duration_seconds=18.6),
        )
    )
    runner = LlamaCppRunner(backend=backend, llama_cli_path=_executable(tmp_path))

    result = runner.run(
        artifact=_artifact(model_path), prompt="Hello", runtime=_runtime()
    )

    allocation = result.observation.runtime_allocation
    assert allocation is not None
    assert allocation.device == "CUDA0"

    mib = 1024 * 1024
    # The reserve pass must not win, and the duplicate compute line must not
    # be added twice.
    assert allocation.device_bytes_for(RuntimeBufferCategory.MODEL) == round(
        3616.41 * mib
    )
    assert allocation.device_bytes_for(RuntimeBufferCategory.KV) == round(192.00 * mib)
    assert allocation.device_bytes_for(RuntimeBufferCategory.COMPUTE) == round(
        183.44 * mib
    )
    assert allocation.total_device_bytes == pytest.approx(
        round((3616.41 + 192.00 + 183.44) * mib), abs=2
    )
    # `CUDA_Host` is host memory despite the prefix.
    assert allocation.total_device_bytes < round(4000 * mib)
