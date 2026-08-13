from __future__ import annotations

import gc
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest

from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.host import HostExecutionBackend


@dataclass
class _GpuProc:
    pid: int
    usedGpuMemory: int


class _FakeNvmlProcessProvider:
    def __init__(self) -> None:
        self._used = 0

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        del index
        return object()

    def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> list[_GpuProc]:
        del handle
        self._used += 1024
        return [
            _GpuProc(pid=child.pid, usedGpuMemory=self._used)
            for child in psutil.Process().children()
        ]


def test_host_backend_captures_stdout_and_stderr() -> None:
    backend = HostExecutionBackend(sample_interval_seconds=0.01)

    result = backend.execute(
        ExecutionRequest(
            command=(
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            )
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.duration_seconds >= 0
    assert result.observation.success is True
    assert result.observation.exit_code == 0
    assert result.observation.failure_reason is None
    assert result.observation.peak_ram_bytes is not None
    assert result.observation.measurement.ram_measurement == "process_rss"
    assert result.observation.measurement.vram_measurement == "nvml_process_memory"


def test_host_backend_translates_missing_executable() -> None:
    backend = HostExecutionBackend()

    with pytest.raises(ExecutableNotFoundError) as ctx:
        backend.execute(
            ExecutionRequest(command=("definitely-not-a-jaull-test-executable",))
        )
    assert ctx.value.observation is not None
    assert ctx.value.observation.success is False
    assert ctx.value.observation.exit_code is None
    assert (
        ctx.value.observation.failure_reason
        is ExecutionFailureReason.EXECUTABLE_NOT_FOUND
    )


def test_host_backend_translates_timeout() -> None:
    backend = HostExecutionBackend(sample_interval_seconds=0.01)

    with pytest.raises(ExecutionTimeoutError) as ctx:
        backend.execute(
            ExecutionRequest(
                command=(sys.executable, "-c", "import time; time.sleep(1)"),
                timeout_seconds=0.01,
            )
        )
    assert ctx.value.result is not None
    assert ctx.value.result.observation.success is False
    assert ctx.value.result.observation.failure_reason is ExecutionFailureReason.TIMEOUT
    assert ctx.value.result.observation.duration_seconds >= 0.01


def test_host_backend_raises_failed_with_captured_result() -> None:
    backend = HostExecutionBackend(sample_interval_seconds=0.01)

    with pytest.raises(ExecutionFailedError) as ctx:
        backend.execute(
            ExecutionRequest(
                command=(
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)",
                )
            )
        )

    assert ctx.value.result.exit_code == 7
    assert ctx.value.result.stdout == "out\n"
    assert ctx.value.result.stderr == "err\n"
    assert ctx.value.result.observation.success is False
    assert (
        ctx.value.result.observation.failure_reason
        is ExecutionFailureReason.NON_ZERO_EXIT
    )


_MIN_DELTA_BYTES = 8 * 1024 * 1024
# 32 MiB of pages, rewritten in a loop until the sampler has had time to
# poll several times. Windows' working set is trimmed aggressively between
# touches, so a one-shot allocation does not stay resident long enough for
# psutil's `memory_info().rss` (which returns WorkingSetSize) to catch the
# peak.
_TOUCH_PAGES_SNIPPET = textwrap.dedent(
    """
    import time
    _size = 32 * 1024 * 1024
    _payload = bytearray(_size)
    _deadline = time.monotonic() + 0.25
    _counter = 0
    while time.monotonic() < _deadline:
        _offset = 0
        while _offset < _size:
            _payload[_offset] = _counter & 0xFF
            _offset += 4096
        _counter += 1
    """
)


def _baseline_peak_ram(backend: HostExecutionBackend) -> int:
    baseline = backend.execute(
        ExecutionRequest(
            command=(sys.executable, "-c", "import time; time.sleep(0.1)"),
            timeout_seconds=5,
        )
    )
    peak = baseline.observation.peak_ram_bytes
    assert peak is not None
    return peak


def test_host_backend_observes_peak_ram_and_duration() -> None:
    backend = HostExecutionBackend(sample_interval_seconds=0.01)
    baseline = _baseline_peak_ram(backend)

    result = backend.execute(
        ExecutionRequest(
            command=(
                sys.executable,
                "-c",
                _TOUCH_PAGES_SNIPPET + 'print("done")\n',
            ),
            timeout_seconds=5,
        )
    )

    assert result.stdout == "done\n"
    assert result.observation.duration_seconds >= 0.1
    assert result.observation.duration_seconds < 5
    assert result.observation.peak_ram_bytes is not None
    assert result.observation.peak_ram_bytes > baseline + _MIN_DELTA_BYTES


def test_host_backend_drains_large_stdout_and_stderr_while_sampling() -> None:
    backend = HostExecutionBackend(sample_interval_seconds=0.01)

    result = backend.execute(
        ExecutionRequest(
            command=(
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import time
                    for _ in range(512):
                        print("x" * 4096)
                        print("y" * 4096, file=sys.stderr)
                    time.sleep(0.05)
                    """
                ),
            ),
            timeout_seconds=5,
        )
    )

    assert len(result.stdout) > 1024 * 1024
    assert len(result.stderr) > 1024 * 1024
    assert result.observation.success is True


def test_host_backend_observes_peak_vram_with_fake_nvml_provider() -> None:
    backend = HostExecutionBackend(
        sample_interval_seconds=0.01,
        nvml_provider=_FakeNvmlProcessProvider(),  # type: ignore[arg-type]
    )

    result = backend.execute(
        ExecutionRequest(
            command=(sys.executable, "-c", "import time; time.sleep(0.08)"),
            timeout_seconds=5,
        )
    )

    assert result.observation.peak_vram_bytes is not None
    assert result.observation.peak_vram_bytes >= 1024


def test_host_backend_failed_process_keeps_observation() -> None:
    backend = HostExecutionBackend(sample_interval_seconds=0.01)
    baseline = _baseline_peak_ram(backend)

    with pytest.raises(ExecutionFailedError) as ctx:
        backend.execute(
            ExecutionRequest(
                command=(
                    sys.executable,
                    "-c",
                    _TOUCH_PAGES_SNIPPET
                    + 'import sys\nprint("failing")\nsys.exit(3)\n',
                ),
                timeout_seconds=5,
            )
        )

    observation = ctx.value.result.observation
    assert observation.success is False
    assert observation.exit_code == 3
    assert observation.failure_reason is ExecutionFailureReason.NON_ZERO_EXIT
    assert observation.duration_seconds >= 0.05
    assert observation.peak_ram_bytes is not None
    assert observation.peak_ram_bytes > baseline + _MIN_DELTA_BYTES


def test_host_backend_timeout_kills_process_without_zombie(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    backend = HostExecutionBackend(sample_interval_seconds=0.01)

    with pytest.raises(ExecutionTimeoutError) as ctx:
        backend.execute(
            ExecutionRequest(
                command=(
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import os
                        import sys
                        import time
                        with open(sys.argv[1], "w", encoding="utf-8") as handle:
                            handle.write(str(os.getpid()))
                            handle.flush()
                        time.sleep(10)
                        """
                    ),
                    str(pid_file),
                ),
                timeout_seconds=0.5,
            )
        )

    observation = ctx.value.result.observation
    assert observation.success is False
    assert observation.failure_reason is ExecutionFailureReason.TIMEOUT
    assert observation.peak_ram_bytes is not None

    child_pid = int(pid_file.read_text(encoding="utf-8"))
    # Drop the exception traceback so Popen (which holds the Windows process
    # handle) can be garbage-collected. Windows keeps the PID slot reserved
    # until every handle to the terminated process is closed, so without this
    # `psutil.pid_exists` would still see the dead child.
    ctx.value.__traceback__ = None
    del ctx
    gc.collect()

    assert not _pid_is_live_or_zombie(child_pid)


def _pid_is_live_or_zombie(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        psutil.Process(pid).status()
        return True
    except psutil.Error:
        return False
