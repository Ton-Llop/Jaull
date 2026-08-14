"""Runtime runner for Transformers/PyTorch steady-state benchmarks."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jaull.benchmarks.errors import (
    BenchmarkConfigurationError,
    BenchmarkParseError,
    BenchmarkRunnerError,
)
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkFailureReason,
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRequest,
)
from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeName
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.ports import ExecutionBackendProtocol

_SUPPORTED_BACKENDS = {ComputeBackend.CPU, ComputeBackend.CUDA, ComputeBackend.HIP}


@dataclass(frozen=True)
class TransformersBenchmarkRunner:
    backend: ExecutionBackendProtocol
    python_executable: str | Path | None = None
    _python: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_python", _resolve_python(self.python_executable))

    def run(self, request: BenchmarkRequest) -> BenchmarkObservation:
        command = build_transformers_benchmark_command(self._python, request)
        try:
            result = self.backend.execute(
                ExecutionRequest(
                    command=command,
                    timeout_seconds=request.timeout_seconds,
                )
            )
        except ExecutableNotFoundError as exc:
            raise BenchmarkRunnerError(str(exc)) from exc
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
                message=_worker_failure_message(exc.result.stdout, str(exc)),
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

        payload = _parse_worker_payload(result.stdout)
        try:
            measurements = [
                _measurement_from_payload(item)
                for item in _list_payload(payload.get("measurements"))
            ]
        except (TypeError, ValueError) as exc:
            raise BenchmarkParseError(str(exc)) from exc
        if not measurements:
            raise BenchmarkParseError("Transformers benchmark emitted no measurements.")
        return BenchmarkObservation(
            success=True,
            measurements=measurements,
            repetitions=request.repetitions,
            duration_seconds=result.duration_seconds,
            methodology=_str_or_none(payload.get("methodology")),
            model_load_seconds=_float_or_none(payload.get("model_load_seconds")),
            warmup_seconds=_float_or_none(payload.get("warmup_seconds")),
            time_to_first_token_seconds=_float_or_none(
                payload.get("time_to_first_token_seconds")
            ),
            generation_latency_seconds=_float_or_none(
                payload.get("generation_latency_seconds")
            ),
            peak_ram_bytes=result.observation.peak_ram_bytes,
            peak_vram_bytes=result.observation.peak_vram_bytes,
            command=command,
            exit_code=result.exit_code,
            raw_stdout=result.stdout,
            raw_stderr=result.stderr,
        )


def build_transformers_benchmark_command(
    python: str,
    request: BenchmarkRequest,
) -> tuple[str, ...]:
    model_ref = _model_ref(request.artifact)
    if request.runtime.runtime is not RuntimeName.TRANSFORMERS:
        raise BenchmarkConfigurationError(
            "Transformers benchmark requires a transformers runtime recommendation."
        )
    if request.backend not in _SUPPORTED_BACKENDS:
        raise BenchmarkConfigurationError(
            f"Transformers benchmark does not support {request.backend.value} backend."
        )
    command: tuple[str, ...] = (
        python,
        "-m",
        "jaull.runtime.transformers_benchmark_worker",
        "--model-ref",
        model_ref,
        "--device-map",
        _device_map(request.backend),
        "--prefill-sizes",
        ",".join(str(size) for size in request.prefill_sizes),
        "--generation-sizes",
        ",".join(str(size) for size in request.generation_sizes),
        "--repetitions",
        str(request.repetitions),
    )
    command = _append_optional(command, "--revision", request.artifact.revision)
    torch_dtype = next(
        (flag.value for flag in request.runtime.flags if flag.name == "torch_dtype"),
        None,
    )
    if torch_dtype is not None:
        command = _append_optional(command, "--torch-dtype", torch_dtype)
    return command


def _resolve_python(configured: str | Path | None) -> str:
    if configured is None:
        return sys.executable
    configured_str = str(configured)
    configured_path = Path(configured).expanduser()
    if configured_path.parent != Path("."):
        if not configured_path.is_file():
            raise BenchmarkRunnerError(f"Python executable not found: {configured_path}.")
        return str(configured_path)
    found = shutil.which(configured_str)
    if found is None:
        raise BenchmarkRunnerError(f"Python executable not found: {configured_str!r}.")
    return found


def _model_ref(artifact: ModelArtifact) -> str:
    if artifact.format.lower() == "gguf":
        raise BenchmarkConfigurationError("Transformers benchmark cannot run GGUF.")
    if artifact.local_path is not None:
        if artifact.local_path.is_dir():
            return str(artifact.local_path)
        if artifact.local_path.exists():
            raise BenchmarkConfigurationError(
                "Transformers benchmark needs a repository or local model directory, "
                f"not a single file: {artifact.local_path}."
            )
    if not artifact.repo_id.strip():
        raise BenchmarkConfigurationError("Artifact repo_id must not be empty.")
    return artifact.repo_id


def _device_map(backend: ComputeBackend) -> str:
    if backend is ComputeBackend.CPU:
        return "cpu"
    if backend in {ComputeBackend.CUDA, ComputeBackend.HIP}:
        return "cuda"
    raise BenchmarkConfigurationError(f"Unsupported backend: {backend.value}.")


def _append_optional(
    command: tuple[str, ...],
    name: str,
    value: str | None,
) -> tuple[str, ...]:
    if value is None or not value.strip():
        return command
    return (*command, name, value)


def _parse_worker_payload(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if payload.get("success") is False:
                raise BenchmarkRunnerError(str(payload.get("error") or "worker failed"))
            return dict(payload)
    raise BenchmarkParseError("Transformers benchmark worker did not emit JSON output.")


def _measurement_from_payload(payload: object) -> BenchmarkMeasurement:
    if not isinstance(payload, dict):
        raise TypeError("measurement payload must be an object")
    return BenchmarkMeasurement(
        kind=BenchmarkMeasurementKind(str(payload["kind"])),
        tokens=int(payload["tokens"]),
        mean_tokens_per_second=float(payload["mean_tokens_per_second"]),
        stddev_tokens_per_second=float(payload["stddev_tokens_per_second"]),
        repetitions=_int_or_none(payload.get("repetitions")),
        source_label=str(payload["source_label"]),
        raw_backend=RuntimeName.TRANSFORMERS.value,
    )


def _list_payload(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("measurements must be a list")
    return value


def _worker_failure_message(stdout: str, fallback: str) -> str:
    try:
        payload = _parse_worker_payload(stdout)
    except BenchmarkRunnerError as exc:
        return str(exc)
    except BenchmarkParseError:
        return fallback
    return str(payload.get("error") or fallback)


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
        methodology="transformers_generate_steady_state_v1",
        peak_ram_bytes=peak_ram_bytes,
        peak_vram_bytes=peak_vram_bytes,
        command=command,
        exit_code=exit_code,
        failure_reason=failure_reason,
        message=message,
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


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}.")


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | str):
        return int(value)
    raise TypeError(f"Expected integer value, got {type(value).__name__}.")


__all__ = [
    "TransformersBenchmarkRunner",
    "build_transformers_benchmark_command",
]
