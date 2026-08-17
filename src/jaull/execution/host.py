"""Host subprocess execution backend."""

from __future__ import annotations

import os
import subprocess
import time

import psutil

from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionMeasurementMetadata,
    ExecutionObservation,
    ExecutionRequest,
    ExecutionResult,
)
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.hardware.nvidia import NvidiaProcessMemorySampler, NvmlProvider

_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.05
_TERMINATE_GRACE_SECONDS = 1.0


class HostExecutionBackend:
    """Execute a command directly on the host with ``shell=False``."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float = _DEFAULT_SAMPLE_INTERVAL_SECONDS,
        nvml_provider: NvmlProvider | None = None,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than 0.")
        self._sample_interval_seconds = sample_interval_seconds
        self._nvml_provider = nvml_provider

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        start = time.perf_counter()
        metadata = ExecutionMeasurementMetadata(
            sample_interval_seconds=self._sample_interval_seconds
        )
        try:
            process = subprocess.Popen(
                request.command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_process_environment(request),
            )
        except FileNotFoundError as exc:
            executable = request.command[0]
            observation = _observation(
                success=False,
                start=start,
                metadata=metadata,
                exit_code=None,
                failure_reason=ExecutionFailureReason.EXECUTABLE_NOT_FOUND,
            )
            raise ExecutableNotFoundError(
                f"Executable not found: {executable!r}.",
                observation,
            ) from exc
        except OSError as exc:
            executable = request.command[0]
            observation = _observation(
                success=False,
                start=start,
                metadata=metadata,
                exit_code=None,
                failure_reason=ExecutionFailureReason.SPAWN_ERROR,
            )
            raise ExecutionError(
                f"Failed to execute {executable!r}: {exc}.",
                observation,
            ) from exc

        sampler = _ProcessExecutionSampler(
            pid=process.pid,
            sample_interval_seconds=self._sample_interval_seconds,
            nvml_provider=self._nvml_provider,
        )
        try:
            sampler.sample()
            stdout, stderr, timed_out = self._communicate_with_sampling(
                process=process,
                request=request,
                start=start,
                sampler=sampler,
            )
            sampler.sample()
            exit_code = process.returncode
            if timed_out:
                result = ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    observation=sampler.observation(
                        success=False,
                        start=start,
                        exit_code=exit_code,
                        failure_reason=ExecutionFailureReason.TIMEOUT,
                    ),
                )
                raise ExecutionTimeoutError(
                    f"Command timed out after {request.timeout_seconds:.1f}s "
                    f"(duration {result.duration_seconds:.3f}s): {request.command[0]!r}.",
                    result,
                )

            success = exit_code == 0
            result = ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                observation=sampler.observation(
                    success=success,
                    start=start,
                    exit_code=exit_code,
                    failure_reason=None
                    if success
                    else ExecutionFailureReason.NON_ZERO_EXIT,
                ),
            )
            if not success:
                raise ExecutionFailedError(
                    f"Command failed with exit code {result.exit_code}: "
                    f"{request.command[0]!r}.",
                    result,
                )
            return result
        finally:
            sampler.close()

    def _communicate_with_sampling(
        self,
        *,
        process: subprocess.Popen[str],
        request: ExecutionRequest,
        start: float,
        sampler: _ProcessExecutionSampler,
    ) -> tuple[str, str, bool]:
        while True:
            elapsed = time.perf_counter() - start
            remaining = request.timeout_seconds - elapsed
            if remaining <= 0:
                sampler.sample()
                return (*self._terminate_and_drain(process), True)

            try:
                stdout, stderr = process.communicate(
                    timeout=min(self._sample_interval_seconds, remaining)
                )
                return stdout, stderr, False
            except subprocess.TimeoutExpired:
                sampler.sample()

    def _terminate_and_drain(
        self, process: subprocess.Popen[str]
    ) -> tuple[str, str]:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        return stdout, stderr


def _process_environment(request: ExecutionRequest) -> dict[str, str] | None:
    if not request.environment:
        return None
    return {**os.environ, **request.environment}


class _ProcessExecutionSampler:
    def __init__(
        self,
        *,
        pid: int,
        sample_interval_seconds: float,
        nvml_provider: NvmlProvider | None,
    ) -> None:
        self._pid = pid
        self._metadata = ExecutionMeasurementMetadata(
            sample_interval_seconds=sample_interval_seconds
        )
        self._process = _psutil_process(pid)
        self._nvidia = NvidiaProcessMemorySampler(nvml_provider)
        self._peak_ram_bytes: int | None = None
        self._peak_vram_bytes: int | None = None

    def sample(self) -> None:
        ram = self._sample_ram()
        if ram is not None:
            self._peak_ram_bytes = _max_optional(self._peak_ram_bytes, ram)

        vram = self._nvidia.sample_pid_bytes(self._pid)
        if vram is not None:
            self._peak_vram_bytes = _max_optional(self._peak_vram_bytes, vram)

    def observation(
        self,
        *,
        success: bool,
        start: float,
        exit_code: int | None,
        failure_reason: ExecutionFailureReason | None,
    ) -> ExecutionObservation:
        return _observation(
            success=success,
            start=start,
            metadata=self._metadata,
            exit_code=exit_code,
            failure_reason=failure_reason,
            peak_ram_bytes=self._peak_ram_bytes,
            peak_vram_bytes=self._peak_vram_bytes,
        )

    def close(self) -> None:
        self._nvidia.close()

    def _sample_ram(self) -> int | None:
        if self._process is None:
            return None
        try:
            return int(self._process.memory_info().rss)
        except (psutil.Error, OSError):
            return None


def _psutil_process(pid: int) -> psutil.Process | None:
    try:
        return psutil.Process(pid)
    except psutil.Error:
        return None


def _observation(
    *,
    success: bool,
    start: float,
    metadata: ExecutionMeasurementMetadata,
    exit_code: int | None,
    failure_reason: ExecutionFailureReason | None,
    peak_ram_bytes: int | None = None,
    peak_vram_bytes: int | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=time.perf_counter() - start,
        peak_ram_bytes=peak_ram_bytes,
        peak_vram_bytes=peak_vram_bytes,
        exit_code=exit_code,
        failure_reason=failure_reason,
        measurement=metadata,
    )


def _max_optional(current: int | None, value: int) -> int:
    if current is None:
        return value
    return max(current, value)


__all__ = ["HostExecutionBackend"]
