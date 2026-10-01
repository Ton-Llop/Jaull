"""Published model evaluations, separate from memory and runtime observations."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from jaull.domain.recommendation import ExternalEvaluationEvidence

_NonBlankText = Annotated[str, Field(min_length=1, pattern=r"\S")]


class CapabilitySubject(BaseModel):
    """Exact evaluated representation; null means unknown, never a wildcard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    variant: _NonBlankText | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
    precision: _NonBlankText | None = None


class EvaluationProtocol(BaseModel):
    """Declared evaluation conditions; missing conditions stay explicit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_method: _NonBlankText | None = None
    few_shot: int | None = Field(default=None, ge=0, strict=True)
    reasoning_mode: _NonBlankText | None = None
    max_reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)
    max_generation_tokens: int | None = Field(default=None, gt=0, strict=True)
    tools: tuple[_NonBlankText, ...] | None = None
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)


class CapabilityEvaluation(ExternalEvaluationEvidence):
    """Strict catalog form of existing external evidence; not a quality verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_id: _NonBlankText
    repo_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    variant: _NonBlankText | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
    precision: _NonBlankText | None = None
    task: Literal["knowledge", "reasoning", "maths", "coding", "instruction_following"]
    benchmark: _NonBlankText
    benchmark_version: _NonBlankText | None = None
    metric: _NonBlankText
    unit: Literal["percent", "fraction", "points"]
    direction: Literal["higher_is_better", "lower_is_better"]
    value: float = Field(allow_inf_nan=False, strict=True)
    source: _NonBlankText
    source_kind: Literal["publisher_reported", "independent"]
    date: _NonBlankText
    protocol: EvaluationProtocol = Field(default_factory=EvaluationProtocol)

    @property
    def subject(self) -> CapabilitySubject:
        return CapabilitySubject(
            repo_id=self.repo_id, variant=self.variant,
            revision=self.revision, precision=self.precision,
        )

    @field_validator("source")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        url = HttpUrl(value)
        if url.username is not None or url.password is not None:
            raise ValueError("Evidence source URLs must not contain credentials")
        return value

    @field_validator("date")
    @classmethod
    def validate_assessment_date(cls, value: str) -> str:
        return date.fromisoformat(value).isoformat()

    @model_validator(mode="after")
    def validate_score_range(self) -> Self:
        if self.unit == "percent" and not 0 <= self.value <= 100:
            raise ValueError("A percent score must be between 0 and 100")
        if self.unit == "fraction" and not 0 <= self.value <= 1:
            raise ValueError("A fraction score must be between 0 and 1")
        return self


class CapabilityCatalog(BaseModel):
    """Versioned source data. It has no ranking weights or family bonuses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    catalog_version: _NonBlankText
    evaluations: tuple[CapabilityEvaluation, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Catalog schema version must be an integer")
        return value

    @model_validator(mode="after")
    def unique_evaluation_ids(self) -> Self:
        ids = [item.evaluation_id for item in self.evaluations]
        if len(ids) != len(set(ids)):
            raise ValueError("Catalog evaluation IDs must be unique")
        return self


def comparison_blockers(
    first: CapabilityEvaluation, second: CapabilityEvaluation
) -> tuple[str, ...]:
    """Check declared conditions only; no score aggregation or winning model.

    An empty result means the recorded protocol fields align, not independent
    verification that the evaluations were executed identically.
    """
    blockers: list[str] = []
    for key in ("task", "benchmark", "benchmark_version", "metric", "unit", "direction"):
        left, right = getattr(first, key), getattr(second, key)
        if left is None or right is None:
            blockers.append(f"{key} is unknown")
        elif left != right:
            blockers.append(f"{key} differs")
    for key in ("variant", "precision"):
        left, right = getattr(first.subject, key), getattr(second.subject, key)
        if left is None or right is None:
            blockers.append(f"subject {key} is unknown")
        elif left != right:
            blockers.append(f"subject {key} differs")
    if first.subject.revision is None or second.subject.revision is None:
        blockers.append("evaluated model revision is unknown")
    for key in EvaluationProtocol.model_fields:
        left, right = getattr(first.protocol, key), getattr(second.protocol, key)
        if left is None or right is None:
            blockers.append(f"protocol {key} is unknown")
        elif left != right:
            blockers.append(f"protocol {key} differs")
    if first.notes != second.notes:
        blockers.append("additional protocol notes differ")
    return tuple(blockers)
