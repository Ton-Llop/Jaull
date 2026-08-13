"""Errors raised by experiment persistence."""

from __future__ import annotations

from jaull.domain.experiments import ExperimentRecord
from jaull.domain.runtime import ExecutionReadiness, LlamaCppRuntimeCapability
from jaull.exceptions import JaullError


class ExperimentStoreError(JaullError):
    """Base class for experiment store failures."""


class ExperimentRecordNotFoundError(ExperimentStoreError):
    """The requested experiment record does not exist."""


class InvalidExperimentIdError(ExperimentStoreError):
    """The experiment id cannot be mapped safely to a local path."""


class ExperimentRunnerError(JaullError):
    """Base class for controlled experiment run failures."""


class ExperimentConfigurationError(ExperimentRunnerError):
    """The experiment request is inconsistent before execution starts."""


class ExperimentNotReadyError(ExperimentRunnerError):
    """The runtime preflight is not ready, so execution was not attempted."""

    def __init__(
        self,
        message: str,
        *,
        readiness: ExecutionReadiness,
        runtime_capability: LlamaCppRuntimeCapability,
    ) -> None:
        self.readiness = readiness
        self.runtime_capability = runtime_capability
        super().__init__(message)


class ExperimentPersistenceError(ExperimentRunnerError):
    """The experiment completed, but storing its record failed."""

    def __init__(self, message: str, *, record: ExperimentRecord) -> None:
        self.record = record
        super().__init__(message)


__all__ = [
    "ExperimentConfigurationError",
    "ExperimentNotReadyError",
    "ExperimentPersistenceError",
    "ExperimentRecordNotFoundError",
    "ExperimentRunnerError",
    "ExperimentStoreError",
    "InvalidExperimentIdError",
]
