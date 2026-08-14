"""Immutable records for completed Jaull execution experiments."""

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
from jaull.domain.comparison import PredictionComparison
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution import ExecutionObservation
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.runtime import (
    ExecutionReadiness,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeRecommendation,
)

EXPERIMENT_RECORD_SCHEMA_VERSION = 1


class RequestedComputeBackend(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    VULKAN = "vulkan"
    HIP = "hip"


class ExperimentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_id: str
    created_at: datetime

    @classmethod
    def create(cls) -> ExperimentIdentity:
        return cls(
            experiment_id=f"exp-{uuid4()}",
            created_at=datetime.now(UTC),
        )

    @field_validator("experiment_id")
    @classmethod
    def _experiment_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("experiment_id must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class ExperimentEnvironment(BaseModel):
    """Small provenance snapshot not already covered by nested domain objects."""

    model_config = ConfigDict(frozen=True)

    jaull_version: str | None = None
    python_version: str | None = None
    python_implementation: str | None = None
    git_commit: str | None = None

    @classmethod
    def capture(cls, *, git_commit: str | None = None) -> ExperimentEnvironment:
        return cls(
            jaull_version=jaull_version,
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            git_commit=git_commit,
        )


class ExperimentPreflight(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_capability: RuntimeCapability
    execution_readiness: ExecutionReadiness


class ExperimentWorkload(BaseModel):
    """Minimal reproducible workload for one inference experiment."""

    model_config = ConfigDict(frozen=True)

    prompt: str

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value


class ExperimentRequest(BaseModel):
    """Concrete experiment configuration.

    This object does not ask Jaull to recommend anything. It carries an already
    selected artifact, runtime configuration, prediction and backend selection.
    """

    model_config = ConfigDict(frozen=True)

    hardware: HardwareProfile
    artifact: ModelArtifact
    runtime: RuntimeRecommendation
    prediction: MemoryEstimate
    backend_selection: RuntimeBackendSelection
    requested_backend: RequestedComputeBackend = RequestedComputeBackend.AUTO
    workload: ExperimentWorkload
    persist: bool = True
    notes: list[str] = Field(default_factory=list)


class ExperimentBackendTrace(BaseModel):
    """Backend intent, selection and observed runtime use for this experiment."""

    model_config = ConfigDict(frozen=True)

    requested_backend: RequestedComputeBackend = RequestedComputeBackend.AUTO
    observed_backend: ComputeBackend | None = None
    observed_source: str | None = None


class ExperimentRecord(BaseModel):
    """A reproducible snapshot of one completed execution experiment.

    The record intentionally links existing domain objects instead of moving
    provenance into ``ExecutionObservation``. A failed execution is still a
    valid experiment because failures are part of the empirical evidence.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = EXPERIMENT_RECORD_SCHEMA_VERSION
    identity: ExperimentIdentity
    environment: ExperimentEnvironment
    hardware: HardwareProfile
    artifact: ModelArtifact
    workload: ExperimentWorkload | None = None
    backend_trace: ExperimentBackendTrace = Field(default_factory=ExperimentBackendTrace)
    runtime: RuntimeRecommendation
    prediction: MemoryEstimate
    preflight: ExperimentPreflight
    observation: ExecutionObservation
    comparison: PredictionComparison
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        hardware: HardwareProfile,
        artifact: ModelArtifact,
        workload: ExperimentWorkload | None = None,
        backend_trace: ExperimentBackendTrace | None = None,
        runtime: RuntimeRecommendation,
        prediction: MemoryEstimate,
        runtime_capability: RuntimeCapability,
        execution_readiness: ExecutionReadiness,
        observation: ExecutionObservation,
        comparison: PredictionComparison,
        identity: ExperimentIdentity | None = None,
        environment: ExperimentEnvironment | None = None,
        notes: list[str] | None = None,
    ) -> ExperimentRecord:
        return cls(
            identity=identity or ExperimentIdentity.create(),
            environment=environment or ExperimentEnvironment.capture(),
            hardware=hardware,
            artifact=artifact,
            workload=workload,
            backend_trace=backend_trace or ExperimentBackendTrace(),
            runtime=runtime,
            prediction=prediction,
            preflight=ExperimentPreflight(
                runtime_capability=runtime_capability,
                execution_readiness=execution_readiness,
            ),
            observation=observation,
            comparison=comparison,
            notes=notes or [],
        )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.schema_version != EXPERIMENT_RECORD_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported ExperimentRecord schema_version: "
                f"{self.schema_version}"
            )
        predicted_runtime = self.prediction.runtime_recommendation
        if predicted_runtime is not None and predicted_runtime != self.runtime:
            raise ValueError(
                "Experiment runtime must match MemoryEstimate.runtime_recommendation"
            )
        if self.preflight.execution_readiness.runtime_capability != (
            self.preflight.runtime_capability
        ):
            raise ValueError(
                "ExecutionReadiness must refer to the same runtime capability "
                "stored in preflight"
            )
        return self


class ExperimentRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: ExperimentRecord
    persisted_path: Path | None = None


__all__ = [
    "EXPERIMENT_RECORD_SCHEMA_VERSION",
    "ExperimentBackendTrace",
    "ExperimentEnvironment",
    "ExperimentIdentity",
    "ExperimentPreflight",
    "ExperimentRecord",
    "ExperimentRequest",
    "ExperimentRunResult",
    "ExperimentWorkload",
    "RequestedComputeBackend",
]
