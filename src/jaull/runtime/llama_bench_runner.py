"""Runtime runner for llama.cpp performance benchmarks through llama-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shlex import join as shell_join

from jaull.benchmarks.errors import (
    BenchmarkConfigurationError,
    BenchmarkParseError,
    BenchmarkRunnerError,
    BenchmarkUnavailableError,
)
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkFailureReason,
    BenchmarkGpuLayers,
    BenchmarkObservation,
    BenchmarkRequest,
    LlamaBenchBinaryStatus,
)
from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.domain.hardware import ComputeBackend
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.ports import ExecutionBackendProtocol
from jaull.runtime.llama_bench_capability import resolve_llama_bench_binary
from jaull.runtime.llama_bench_parser import parse_llama_bench_output


@dataclass(frozen=True)
class LlamaBenchRunner:
    backend: ExecutionBackendProtocol
    llama_bench_path: str | Path | None = None
    _llama_bench: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved = resolve_llama_bench_binary(self.llama_bench_path)
        if resolved.binary_status is LlamaBenchBinaryStatus.MISSING:
            raise BenchmarkUnavailableError(
                resolved.message or "llama-bench executable not found."
            )
        if resolved.binary_status is LlamaBenchBinaryStatus.NOT_EXECUTABLE:
            raise BenchmarkUnavailableError(
                resolved.message or "llama-bench is not executable."
            )
        if resolved.binary_path is None:
            raise BenchmarkUnavailableError("llama-bench executable could not be resolved.")
        object.__setattr__(self, "_llama_bench", resolved.binary_path)

    def run(self, request: BenchmarkRequest) -> BenchmarkObservation:
        command = build_llama_bench_command(self._llama_bench, request)
        try:
            result = self.backend.execute(
                ExecutionRequest(
                    command=command,
                    timeout_seconds=request.timeout_seconds,
                )
            )
        except ExecutableNotFoundError as exc:
            raise BenchmarkUnavailableError(str(exc)) from exc
        except ExecutionTimeoutError as exc:
            if exc.result is None:
                raise BenchmarkRunnerError(str(exc)) from exc
            return _failed_observation(
                command=command,
                request=request,
                failure_reason=BenchmarkFailureReason.TIMEOUT,
                message=str(exc),
                stdout=exc.result.stdout,
                stderr=exc.result.stderr,
                duration_seconds=exc.result.duration_seconds,
                exit_code=exc.result.exit_code,
                peak_ram_bytes=exc.result.observation.peak_ram_bytes,
                peak_vram_bytes=exc.result.observation.peak_vram_bytes,
            )
        except ExecutionFailedError as exc:
            return _failed_observation(
                command=command,
                request=request,
                failure_reason=BenchmarkFailureReason.NON_ZERO_EXIT,
                message=str(exc),
                stdout=exc.result.stdout,
                stderr=exc.result.stderr,
                duration_seconds=exc.result.duration_seconds,
                exit_code=exc.result.exit_code,
                peak_ram_bytes=exc.result.observation.peak_ram_bytes,
                peak_vram_bytes=exc.result.observation.peak_vram_bytes,
            )
        except ExecutionError as exc:
            if exc.observation is None:
                raise BenchmarkRunnerError(str(exc)) from exc
            return _failed_observation(
                command=command,
                request=request,
                failure_reason=_benchmark_failure_reason(exc.observation.failure_reason),
                message=str(exc),
                stdout="",
                stderr="",
                duration_seconds=exc.observation.duration_seconds,
                exit_code=exc.observation.exit_code,
                peak_ram_bytes=exc.observation.peak_ram_bytes,
                peak_vram_bytes=exc.observation.peak_vram_bytes,
            )

        raw_output = "\n".join((result.stdout, result.stderr))
        try:
            measurements = parse_llama_bench_output(
                raw_output,
                repetitions=request.repetitions,
            )
        except BenchmarkParseError:
            raise
        except Exception as exc:
            raise BenchmarkParseError(str(exc)) from exc
        return BenchmarkObservation(
            success=True,
            measurements=measurements,
            repetitions=request.repetitions,
            duration_seconds=result.duration_seconds,
            command=command,
            exit_code=result.exit_code,
            peak_ram_bytes=result.observation.peak_ram_bytes,
            peak_vram_bytes=result.observation.peak_vram_bytes,
            raw_stdout=result.stdout,
            raw_stderr=result.stderr,
        )


def build_llama_bench_command(
    llama_bench: str,
    request: BenchmarkRequest,
) -> tuple[str, ...]:
    model_path = _validate_artifact(request.artifact)
    device = _device_arg(request)
    return (
        llama_bench,
        "-m",
        str(model_path),
        "-dev",
        device,
        "-ngl",
        _gpu_layers_arg(request.gpu_layers),
        "-p",
        ",".join(str(size) for size in request.prefill_sizes),
        "-n",
        ",".join(str(size) for size in request.generation_sizes),
        "-r",
        str(request.repetitions),
    )


def _validate_artifact(artifact: ModelArtifact) -> Path:
    if artifact.format.lower() != "gguf":
        raise BenchmarkConfigurationError(
            f"llama-bench only supports GGUF artifacts, got {artifact.format!r}."
        )
    if artifact.local_path is None:
        raise BenchmarkConfigurationError("Artifact has no local_path.")
    if not artifact.is_downloaded:
        raise BenchmarkConfigurationError("Artifact must be downloaded before benchmark.")
    if not artifact.is_verified:
        raise BenchmarkConfigurationError("Artifact must be verified before benchmark.")
    if not artifact.local_path.is_file():
        raise BenchmarkConfigurationError(
            f"Artifact file does not exist: {artifact.local_path}."
        )
    return artifact.local_path


def _device_arg(request: BenchmarkRequest) -> str:
    if request.backend is ComputeBackend.CPU:
        return "none"
    if not request.device:
        raise BenchmarkConfigurationError(
            f"{request.backend.value} benchmark requires a runtime device id."
        )
    return request.device


def _gpu_layers_arg(gpu_layers: BenchmarkGpuLayers) -> str:
    if gpu_layers.full_offload:
        return "-1"
    assert gpu_layers.count is not None
    return str(gpu_layers.count)


def _failed_observation(
    *,
    command: tuple[str, ...],
    request: BenchmarkRequest,
    failure_reason: BenchmarkFailureReason,
    message: str,
    stdout: str,
    stderr: str,
    duration_seconds: float,
    exit_code: int | None,
    peak_ram_bytes: int | None,
    peak_vram_bytes: int | None,
) -> BenchmarkObservation:
    return BenchmarkObservation(
        success=False,
        measurements=[],
        repetitions=request.repetitions,
        duration_seconds=duration_seconds,
        peak_ram_bytes=peak_ram_bytes,
        peak_vram_bytes=peak_vram_bytes,
        command=command,
        exit_code=exit_code,
        failure_reason=failure_reason,
        message=_failure_message(
            command=command,
            exit_code=exit_code,
            message=message,
            stdout=stdout,
            stderr=stderr,
        ),
        raw_stdout=stdout,
        raw_stderr=stderr,
    )


def _benchmark_failure_reason(
    reason: ExecutionFailureReason | None,
) -> BenchmarkFailureReason:
    if reason is ExecutionFailureReason.EXECUTABLE_NOT_FOUND:
        return BenchmarkFailureReason.EXECUTABLE_NOT_FOUND
    if reason is ExecutionFailureReason.SPAWN_ERROR:
        return BenchmarkFailureReason.SPAWN_ERROR
    if reason is ExecutionFailureReason.TIMEOUT:
        return BenchmarkFailureReason.TIMEOUT
    if reason is ExecutionFailureReason.NON_ZERO_EXIT:
        return BenchmarkFailureReason.NON_ZERO_EXIT
    return BenchmarkFailureReason.SPAWN_ERROR


def _failure_message(
    *,
    command: tuple[str, ...],
    exit_code: int | None,
    message: str,
    stdout: str,
    stderr: str,
) -> str:
    parts = [
        message,
        f"argv: {shell_join(command)}",
        f"exit_code: {exit_code if exit_code is not None else 'unknown'}",
    ]
    stderr_line = _last_useful_line(stderr)
    stdout_line = _last_useful_line(stdout)
    if stderr_line:
        parts.append(f"stderr: {stderr_line}")
    if stdout_line:
        parts.append(f"stdout: {stdout_line}")
    return "\n".join(parts)


def _last_useful_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


__all__ = ["LlamaBenchRunner", "build_llama_bench_command"]
