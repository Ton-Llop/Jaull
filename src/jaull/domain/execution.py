"""Small domain contracts for running external commands.

These objects deliberately know nothing about subprocesses, llama.cpp, Docker
or Hugging Face. They describe an execution request and the observed result.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

from jaull.domain.hardware import ComputeBackend


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


class RuntimeBufferLocation(StrEnum):
    """Which memory a runtime buffer occupies."""

    DEVICE = "device"
    HOST = "host"


class RuntimeBufferCategory(StrEnum):
    MODEL = "model"
    KV = "kv"
    COMPUTE = "compute"
    OUTPUT = "output"
    OTHER = "other"


class RuntimeBuffer(BaseModel):
    """One allocation a runtime reported, normalised but still traceable.

    ``raw_label`` is kept beside the normalised fields on purpose: the parser
    is specific to one runtime and one build, and labels change. Without the
    original string a surprising category is impossible to debug.
    """

    model_config = ConfigDict(frozen=True)

    category: RuntimeBufferCategory
    location: RuntimeBufferLocation
    bytes: int = Field(ge=0)
    raw_label: str


class RuntimeReportedAllocation(BaseModel):
    """Memory a runtime says it allocated, as opposed to what a driver confirms.

    This is a second, independent observation source next to NVML's
    process-attributed figure, and it exists because NVML cannot attribute
    memory per process on a GPU in WDDM mode — every consumer card driving a
    display. The runtime's own report is available there.

    It is deliberately literal. ``total_device_bytes`` is the sum of the buffers
    the runtime enumerated and nothing else: no CUDA context estimate, no
    allocator guess, no device reserve, no safety margin. What the runtime does
    not report does not exist in this observation. That is what makes the
    difference against a driver-attributed measurement meaningful later —
    ``nvml_process_allocation`` minus ``total_device_bytes`` is exactly the device
    memory the runtime never enumerated.
    """

    model_config = ConfigDict(frozen=True)

    runtime: str
    device: str | None = None
    runtime_build: str | None = None
    buffers: tuple[RuntimeBuffer, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_device_bytes(self) -> int:
        return sum(
            buffer.bytes
            for buffer in self.buffers
            if buffer.location is RuntimeBufferLocation.DEVICE
        )

    def device_bytes_for(self, category: RuntimeBufferCategory) -> int | None:
        """Device bytes for one category, or ``None`` when it was not reported."""
        matching = [
            buffer.bytes
            for buffer in self.buffers
            if buffer.location is RuntimeBufferLocation.DEVICE
            and buffer.category is category
        ]
        return sum(matching) if matching else None


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
    # A second observation source, not a replacement for ``peak_vram_bytes``:
    # that one is the driver's view and stays authoritative where it exists.
    runtime_allocation: RuntimeReportedAllocation | None = None


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    environment: dict[str, str] = Field(default_factory=dict)


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
    command: tuple[str, ...] = Field(default_factory=tuple)
    observed_backend: ComputeBackend | None = None
    observed_backend_source: str | None = None
    raw_stdout: str | None = None
    raw_stderr: str | None = None

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
    "RuntimeBuffer",
    "RuntimeBufferCategory",
    "RuntimeBufferLocation",
    "RuntimeReportedAllocation",
]
