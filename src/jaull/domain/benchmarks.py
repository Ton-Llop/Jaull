"""Domain objects for reproducible llama.cpp runtime benchmarks."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jaull import __version__ as jaull_version
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.runtime import RuntimeRecommendation

BENCHMARK_RECORD_SCHEMA_VERSION = 1
DEFAULT_PREFILL_SIZES = (128, 512, 2048)
DEFAULT_GENERATION_SIZES = (128,)
DEFAULT_BENCHMARK_REPETITIONS = 5


class BenchmarkMeasurementKind(StrEnum):
    PREFILL = "prefill"
    GENERATION = "generation"


class BenchmarkFailureReason(StrEnum):
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    SPAWN_ERROR = "spawn_error"
    TIMEOUT = "timeout"
    NON_ZERO_EXIT = "non_zero_exit"
    PARSE_ERROR = "parse_error"
    NOT_RUNNABLE = "not_runnable"


class LlamaBenchBinaryStatus(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    NOT_EXECUTABLE = "not_executable"
    PROBE_FAILED = "probe_failed"
    UNKNOWN = "unknown"


class BenchmarkGpuLayers(BaseModel):
    """Requested GPU layer offload without exposing llama.cpp's ``-1`` sentinel."""

    model_config = ConfigDict(frozen=True)

    count: int | None = Field(default=0, ge=0)
    full_offload: bool = False

    @classmethod
    def count_layers(cls, count: int) -> BenchmarkGpuLayers:
        return cls(count=count, full_offload=False)

    @classmethod
    def full(cls) -> BenchmarkGpuLayers:
        return cls(count=None, full_offload=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.full_offload and self.count is not None:
            raise ValueError("full_offload cannot also set count")
        if not self.full_offload and self.count is None:
            raise ValueError("count is required unless full_offload is true")
        return self

    @property
    def label(self) -> str:
        if self.full_offload:
            return "all"
        return str(self.count)


class BenchmarkMeasurement(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: BenchmarkMeasurementKind
    tokens: int = Field(gt=0)
    mean_tokens_per_second: float = Field(ge=0.0)
    stddev_tokens_per_second: float = Field(ge=0.0)
    repetitions: int | None = Field(default=None, ge=1)
    source_label: str
    raw_backend: str | None = None
    raw_device: str | None = None
    raw_ngl: str | None = None


class BenchmarkObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    measurements: list[BenchmarkMeasurement] = Field(default_factory=list)
    repetitions: int = Field(ge=1)
    duration_seconds: float = Field(ge=0.0)
    command: tuple[str, ...] = Field(default_factory=tuple)
    exit_code: int | None = None
    failure_reason: BenchmarkFailureReason | None = None
    message: str | None = None
    raw_stdout: str = ""
    raw_stderr: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.success and not self.measurements:
            raise ValueError("successful BenchmarkObservation requires measurements")
        if self.success and self.failure_reason is not None:
            raise ValueError("successful BenchmarkObservation cannot have failure_reason")
        if not self.success and self.failure_reason is None:
            raise ValueError("failed BenchmarkObservation requires failure_reason")
        return self


class BenchmarkRequest(BaseModel):
    """One concrete llama-bench configuration."""

    model_config = ConfigDict(frozen=True)

    artifact: ModelArtifact
    runtime: RuntimeRecommendation
    backend: ComputeBackend
    device: str | None = None
    gpu_layers: BenchmarkGpuLayers = Field(default_factory=BenchmarkGpuLayers)
    prefill_sizes: tuple[int, ...] = DEFAULT_PREFILL_SIZES
    generation_sizes: tuple[int, ...] = DEFAULT_GENERATION_SIZES
    repetitions: int = Field(default=DEFAULT_BENCHMARK_REPETITIONS, ge=1)
    timeout_seconds: float = Field(default=900.0, gt=0.0)
    notes: list[str] = Field(default_factory=list)

    @field_validator("prefill_sizes", "generation_sizes")
    @classmethod
    def _sizes_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("benchmark token size list must not be empty")
        if any(item <= 0 for item in value):
            raise ValueError("benchmark token sizes must be positive")
        return value

    @model_validator(mode="after")
    def _cpu_request_is_explicit(self) -> Self:
        if self.backend is ComputeBackend.CPU:
            if self.device not in {None, "none"}:
                raise ValueError("CPU benchmark device must be none")
            if self.gpu_layers.full_offload or self.gpu_layers.count != 0:
                raise ValueError("CPU benchmark requires gpu_layers=0")
        return self


class BenchmarkIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    created_at: datetime

    @classmethod
    def create(cls) -> BenchmarkIdentity:
        return cls(
            benchmark_id=f"bench-{uuid4()}",
            created_at=datetime.now(UTC),
        )

    @field_validator("benchmark_id")
    @classmethod
    def _benchmark_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("benchmark_id must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class BenchmarkEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True)

    jaull_version: str | None = None
    python_version: str | None = None
    python_implementation: str | None = None
    platform: str | None = None
    git_commit: str | None = None

    @classmethod
    def capture(cls, *, git_commit: str | None = None) -> BenchmarkEnvironment:
        return cls(
            jaull_version=jaull_version,
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            platform=platform.platform(),
            git_commit=git_commit,
        )


class LlamaBenchCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    binary_path: str | None = None
    binary_status: LlamaBenchBinaryStatus
    version_text: str | None = None
    probe_source: str | None = None
    message: str | None = None


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = BENCHMARK_RECORD_SCHEMA_VERSION
    identity: BenchmarkIdentity
    environment: BenchmarkEnvironment
    hardware: HardwareProfile
    artifact: ModelArtifact
    runtime: RuntimeRecommendation
    requested_backend: ComputeBackend
    requested_device: str | None = None
    gpu_layers: BenchmarkGpuLayers
    request: BenchmarkRequest
    observation: BenchmarkObservation
    llama_bench_capability: LlamaBenchCapability
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        hardware: HardwareProfile,
        request: BenchmarkRequest,
        observation: BenchmarkObservation,
        llama_bench_capability: LlamaBenchCapability,
        identity: BenchmarkIdentity | None = None,
        environment: BenchmarkEnvironment | None = None,
        notes: list[str] | None = None,
    ) -> BenchmarkRecord:
        return cls(
            identity=identity or BenchmarkIdentity.create(),
            environment=environment or BenchmarkEnvironment.capture(),
            hardware=hardware,
            artifact=request.artifact,
            runtime=request.runtime,
            requested_backend=request.backend,
            requested_device=request.device,
            gpu_layers=request.gpu_layers,
            request=request,
            observation=observation,
            llama_bench_capability=llama_bench_capability,
            notes=notes or request.notes,
        )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.schema_version != BENCHMARK_RECORD_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported BenchmarkRecord schema_version: "
                f"{self.schema_version}"
            )
        if self.artifact != self.request.artifact:
            raise ValueError("BenchmarkRecord artifact must match request artifact")
        if self.runtime != self.request.runtime:
            raise ValueError("BenchmarkRecord runtime must match request runtime")
        if self.request.backend is not self.requested_backend:
            raise ValueError("BenchmarkRecord requested_backend must match request")
        if self.request.device != self.requested_device:
            raise ValueError("BenchmarkRecord requested_device must match request")
        if self.request.gpu_layers != self.gpu_layers:
            raise ValueError("BenchmarkRecord gpu_layers must match request")
        return self


class BenchmarkRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: BenchmarkRecord
    persisted_path: Path | None = None


__all__ = [
    "BENCHMARK_RECORD_SCHEMA_VERSION",
    "DEFAULT_BENCHMARK_REPETITIONS",
    "DEFAULT_GENERATION_SIZES",
    "DEFAULT_PREFILL_SIZES",
    "BenchmarkEnvironment",
    "BenchmarkFailureReason",
    "BenchmarkGpuLayers",
    "BenchmarkIdentity",
    "BenchmarkMeasurement",
    "BenchmarkMeasurementKind",
    "BenchmarkObservation",
    "BenchmarkRecord",
    "BenchmarkRequest",
    "BenchmarkRunResult",
    "LlamaBenchBinaryStatus",
    "LlamaBenchCapability",
]
