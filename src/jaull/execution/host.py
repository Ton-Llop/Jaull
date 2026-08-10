"""Host subprocess execution backend."""

from __future__ import annotations

import subprocess
import time

from jaull.domain.execution import ExecutionRequest, ExecutionResult
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)


class HostExecutionBackend:
    """Execute a command directly on the host with ``shell=False``."""

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        start = time.monotonic()
        try:
            completed = subprocess.run(
                request.command,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=request.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            executable = request.command[0]
            raise ExecutableNotFoundError(
                f"Executable not found: {executable!r}."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            raise ExecutionTimeoutError(
                f"Command timed out after {request.timeout_seconds:.1f}s "
                f"(duration {duration:.3f}s): {request.command[0]!r}."
            ) from exc
        except OSError as exc:
            executable = request.command[0]
            raise ExecutionError(
                f"Failed to execute {executable!r}: {exc}."
            ) from exc

        result = ExecutionResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.monotonic() - start,
        )
        if result.exit_code != 0:
            raise ExecutionFailedError(
                f"Command failed with exit code {result.exit_code}: "
                f"{request.command[0]!r}.",
                result,
            )
        return result


__all__ = ["HostExecutionBackend"]
