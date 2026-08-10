from __future__ import annotations

import sys

import pytest

from jaull.domain.execution import ExecutionRequest
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.host import HostExecutionBackend


def test_host_backend_captures_stdout_and_stderr() -> None:
    backend = HostExecutionBackend()

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


def test_host_backend_translates_missing_executable() -> None:
    backend = HostExecutionBackend()

    with pytest.raises(ExecutableNotFoundError):
        backend.execute(
            ExecutionRequest(command=("definitely-not-a-jaull-test-executable",))
        )


def test_host_backend_translates_timeout() -> None:
    backend = HostExecutionBackend()

    with pytest.raises(ExecutionTimeoutError):
        backend.execute(
            ExecutionRequest(
                command=(sys.executable, "-c", "import time; time.sleep(1)"),
                timeout_seconds=0.01,
            )
        )


def test_host_backend_raises_failed_with_captured_result() -> None:
    backend = HostExecutionBackend()

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
