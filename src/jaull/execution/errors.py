"""Errors raised by execution backends."""

from __future__ import annotations

from jaull.domain.execution import ExecutionResult
from jaull.exceptions import JaullError


class ExecutionError(JaullError):
    """Base class for command execution failures."""


class ExecutableNotFoundError(ExecutionError):
    """The requested executable is not installed or not reachable."""


class ExecutionTimeoutError(ExecutionError):
    """The command exceeded its configured timeout."""


class ExecutionFailedError(ExecutionError):
    """The command completed but returned a non-zero exit code."""

    def __init__(self, message: str, result: ExecutionResult) -> None:
        self.result = result
        super().__init__(message)


__all__ = [
    "ExecutableNotFoundError",
    "ExecutionError",
    "ExecutionFailedError",
    "ExecutionTimeoutError",
]
