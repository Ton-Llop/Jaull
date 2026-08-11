"""Errors raised by execution backends."""

from __future__ import annotations

from jaull.domain.execution import ExecutionObservation, ExecutionResult
from jaull.exceptions import JaullError


class ExecutionError(JaullError):
    """Base class for command execution failures."""

    def __init__(
        self,
        message: str,
        observation: ExecutionObservation | None = None,
    ) -> None:
        self.observation = observation
        super().__init__(message)


class ExecutableNotFoundError(ExecutionError):
    """The requested executable is not installed or not reachable."""


class ExecutionTimeoutError(ExecutionError):
    """The command exceeded its configured timeout."""

    def __init__(
        self,
        message: str,
        result: ExecutionResult | None = None,
    ) -> None:
        self.result = result
        super().__init__(
            message,
            result.observation if result is not None else None,
        )


class ExecutionFailedError(ExecutionError):
    """The command completed but returned a non-zero exit code."""

    def __init__(self, message: str, result: ExecutionResult) -> None:
        self.result = result
        super().__init__(message, result.observation)


__all__ = [
    "ExecutableNotFoundError",
    "ExecutionError",
    "ExecutionFailedError",
    "ExecutionTimeoutError",
]
