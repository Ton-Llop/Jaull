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


class PredictionComparison(BaseModel):
    """Comparison between a Jaull prediction and one measured execution."""

    model_config = ConfigDict(frozen=True)

    ram: MetricComparison
    vram: MetricComparison
    compatibility: CompatibilityComparison


__all__ = [
    "CompatibilityComparison",
    "CompatibilityOutcome",
    "MetricComparison",
    "MetricComparisonAvailability",
    "PredictionComparison",
]
