"""Domain models for comparing predictions against observed executions."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.execution import ExecutionFailureReason


class MetricComparisonAvailability(StrEnum):
    AVAILABLE = "available"
    PREDICTION_UNAVAILABLE = "prediction_unavailable"
    MEASUREMENT_UNAVAILABLE = "measurement_unavailable"
    METHODOLOGICALLY_UNAVAILABLE = "methodologically_unavailable"


class CompatibilityOutcome(StrEnum):
    CORRECT_SUCCESS = "correct_success"
    CORRECT_FAILURE = "correct_failure"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    UNKNOWN = "unknown"


class MemoryObservationSource(StrEnum):
    """Where a measured memory figure came from.

    The two are not interchangeable. NVML reports what the driver attributed to
    the process; a runtime reports what it asked for and enumerated. They can
    coexist for the same run, and on a GPU in WDDM mode only the second exists.
    """

    NVML_PROCESS_ALLOCATION = "nvml_process_allocation"
    RUNTIME_REPORTED_ALLOCATION = "runtime_reported_allocation"


class ComparisonSemantics(StrEnum):
    """How closely a predicted quantity and an observed one correspond.

    ``PROXY`` is the load-bearing one. It marks a pair that is informative but
    not equivalent, so a large error cannot be read as "the prediction is that
    wrong" without first establishing what the observation leaves out.
    """

    DIRECT = "direct"
    PARTIAL = "partial"
    PROXY = "proxy"
    UNAVAILABLE = "unavailable"


class MetricComparison(BaseModel):
    """Point comparison for one resource metric.

    Error convention:

    ``error_bytes = measured_bytes - predicted_bytes``.

    Positive error means the real execution used more than Jaull predicted;
    negative error means Jaull overestimated resource usage.
    """

    model_config = ConfigDict(frozen=True)

    predicted_bytes: int | None = Field(default=None, ge=0)
    measured_bytes: int | None = Field(default=None, ge=0)
    error_bytes: int | None = None
    absolute_error_bytes: int | None = Field(default=None, ge=0)
    error_percent: float | None = None
    availability: MetricComparisonAvailability
    unavailable_reason: str | None = None
    # Provenance of ``measured_bytes``. Absent on records written before the
    # observation contract existed, and on metrics with a single source.
    source: MemoryObservationSource | None = None
    runtime: str | None = None
    driver_confirmed: bool | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.availability is MetricComparisonAvailability.AVAILABLE:
            if self.predicted_bytes is None or self.measured_bytes is None:
                raise ValueError("available comparisons require prediction and measurement")
            if self.error_bytes != self.measured_bytes - self.predicted_bytes:
                raise ValueError("error_bytes must equal measured_bytes - predicted_bytes")
            if self.absolute_error_bytes != abs(self.error_bytes):
                raise ValueError("absolute_error_bytes must equal abs(error_bytes)")
            return self

        unavailable_fields = (
            self.error_bytes,
            self.absolute_error_bytes,
            self.error_percent,
        )
        if any(value is not None for value in unavailable_fields):
            raise ValueError("unavailable comparisons cannot contain error values")
        if not self.unavailable_reason:
            raise ValueError("unavailable comparisons require a reason")
        return self


class CompatibilityComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicted_status: CompatibilityStatus
    predicted_runnable: bool | None
    observed_success: bool
    outcome: CompatibilityOutcome
    failure_reason: ExecutionFailureReason | None = None


class ComponentComparison(BaseModel):
    """One predicted component against the observation that stands for it.

    A total hides which term is wrong. On the recorded RTX 2060 cases the GPU
    weights land within 5% while the overhead heuristic is several times the
    compute buffer, and only the breakdown says so.
    """

    model_config = ConfigDict(frozen=True)

    component: str
    observed_label: str
    semantics: ComparisonSemantics
    semantics_reason: str | None = None
    metric: MetricComparison

    @model_validator(mode="after")
    def _proxy_is_explained(self) -> Self:
        if (
            self.semantics in {ComparisonSemantics.PROXY, ComparisonSemantics.PARTIAL}
            and not self.semantics_reason
        ):
            raise ValueError(
                "proxy and partial comparisons require a reason naming what the "
                "observation does not cover"
            )
        return self


class PredictionComparison(BaseModel):
    """Comparison between a Jaull prediction and one measured execution."""

    model_config = ConfigDict(frozen=True)

    ram: MetricComparison
    vram: MetricComparison
    compatibility: CompatibilityComparison
    vram_components: tuple[ComponentComparison, ...] = ()


__all__ = [
    "ComparisonSemantics",
    "CompatibilityComparison",
    "CompatibilityOutcome",
    "ComponentComparison",
    "MemoryObservationSource",
    "MetricComparison",
    "MetricComparisonAvailability",
    "PredictionComparison",
]
