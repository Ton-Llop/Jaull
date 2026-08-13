"""Errors raised by benchmark execution and persistence."""

from __future__ import annotations

from jaull.domain.benchmarks import BenchmarkRecord
from jaull.exceptions import JaullError


class BenchmarkRunnerError(JaullError):
    """Base class for benchmark runner failures."""


class BenchmarkConfigurationError(BenchmarkRunnerError):
    """The benchmark request is inconsistent before execution starts."""


class BenchmarkUnavailableError(BenchmarkRunnerError):
    """The benchmark binary or selected backend is unavailable."""


class BenchmarkParseError(BenchmarkRunnerError):
    """llama-bench completed but its benchmark table could not be parsed."""


class BenchmarkPersistenceError(BenchmarkRunnerError):
    """The benchmark completed, but storing its record failed."""

    def __init__(self, message: str, *, record: BenchmarkRecord) -> None:
        self.record = record
        super().__init__(message)


class BenchmarkStoreError(JaullError):
    """Base class for benchmark store failures."""


class BenchmarkRecordNotFoundError(BenchmarkStoreError):
    """The requested benchmark record does not exist."""


class InvalidBenchmarkIdError(BenchmarkStoreError):
    """The benchmark id cannot be mapped safely to a local path."""


__all__ = [
    "BenchmarkConfigurationError",
    "BenchmarkParseError",
    "BenchmarkPersistenceError",
    "BenchmarkRecordNotFoundError",
    "BenchmarkRunnerError",
    "BenchmarkStoreError",
    "BenchmarkUnavailableError",
    "InvalidBenchmarkIdError",
]
