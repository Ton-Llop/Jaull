"""Small domain contracts for running external commands.

These objects deliberately know nothing about subprocesses, llama.cpp, Docker
or Hugging Face. They describe an execution request and the observed result.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExecutionFailureReason(StrEnum):
    """Stable, objective reasons observed at the process boundary."""

    EXECUTABLE_NOT_FOUND = "executable_not_found"
    SPAWN_ERROR = "spawn_error"
    TIMEOUT = "timeout"
    NON_ZERO_EXIT = "non_zero_exit"


class ExecutionMeasurementMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    ram_measurement: str = "process_rss"
    vram_measurement: str = "nvml_process_memory"
    sample_interval_seconds: float = Field(default=0.05, gt=0.0)


class ExecutionObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    duration_seconds: float = Field(ge=0.0)
    peak_ram_bytes: int | None = Field(default=None, ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    exit_code: int | None
    failure_reason: ExecutionFailureReason | None = None
    measurement: ExecutionMeasurementMetadata = Field(
        default_factory=ExecutionMeasurementMetadata
    )


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    observation: ExecutionObservation

    @property
    def exit_code(self) -> int | None:
        return self.observation.exit_code

    @property
    def duration_seconds(self) -> float:
        return self.observation.duration_seconds


class InferenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    runtime: str
    model_path: Path
    observation: ExecutionObservation

    @property
    def exit_code(self) -> int | None:
        return self.observation.exit_code

    @property
    def duration_seconds(self) -> float:
        return self.observation.duration_seconds


__all__ = [
    "ExecutionFailureReason",
    "ExecutionMeasurementMetadata",
    "ExecutionObservation",
    "ExecutionRequest",
    "ExecutionResult",
    "InferenceResult",
]
