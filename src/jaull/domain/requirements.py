"""User-facing requirement models shared by discovery, recommendation and workflow.

These are the pure-domain enums and Pydantic models. Progress-tracking state
(WorkflowStep, StepStatus, ProgressStep, WorkflowProgress) lives in
``workflow/models.py`` because it is orchestration concern, not domain.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UseCase(StrEnum):
    GENERAL_CHAT = "general_chat"
    CODING = "coding"
    DOCUMENT_QA = "document_qa"
    SUMMARIZATION_EXTRACTION = "summarization_extraction"
    REASONING = "reasoning"
    BATCH_PROCESSING = "batch_processing"  # Legacy input; normalized on load.
    WRITING_TRANSLATION = "writing_translation"


class RecommendationPriority(StrEnum):
    QUALITY = "quality"
    BALANCED = "balanced"
    SPEED = "speed"
    MEMORY = "memory"


class WorkloadMode(StrEnum):
    INTERACTIVE = "interactive"
    BATCH = "batch"
    SERVICE = "service"


class WorkloadProfile(BaseModel):
    """Requested workload, not a measurement or a deployment qualification."""

    model_config = ConfigDict(frozen=True)

    context_length: int = Field(gt=0)
    concurrent_users: int = Field(default=1, ge=1)
    mode: WorkloadMode = WorkloadMode.INTERACTIVE
    expected_input_tokens: int | None = Field(default=None, gt=0)
    expected_output_tokens: int | None = Field(default=None, gt=0)
    min_generation_tps: float | None = Field(default=None, gt=0)
    max_ttft_ms: float | None = Field(default=None, gt=0)


def _normalize_legacy_batch(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("use_case") not in (
        UseCase.BATCH_PROCESSING,
        UseCase.BATCH_PROCESSING.value,
    ):
        return value
    normalized = dict(value)
    normalized["use_case"] = UseCase.GENERAL_CHAT
    normalized.setdefault("workload_mode", WorkloadMode.BATCH)
    return normalized


class DocumentScale(StrEnum):
    """How much text the user expects to feed the model at once."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    COLLECTION = "collection"


class ConcurrencyLevel(StrEnum):
    SINGLE = "single"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class CommercialUse(StrEnum):
    """Answer to "must the model allow commercial use?"."""

    YES = "yes"
    NO = "no"
    NOT_SURE = "not_sure"
    NO_PREFERENCE = "no_preference"


class UserAnswers(BaseModel):
    """Raw wizard answers, exactly as the user gave them.

    Kept separate from :class:`UserRequirements` so the report can show both what
    was asked and what the tool inferred from it.
    """

    model_config = ConfigDict(frozen=True)

    use_case: UseCase
    workload_mode: WorkloadMode = WorkloadMode.INTERACTIVE
    priority: RecommendationPriority
    languages: list[str] = Field(default_factory=list)
    other_languages: list[str] = Field(default_factory=list)
    concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE
    document_scale: DocumentScale | None = None
    commercial_use: CommercialUse = CommercialUse.YES

    @model_validator(mode="before")
    @classmethod
    def _legacy_batch(cls, value: Any) -> Any:
        return _normalize_legacy_batch(value)


class UserRequirements(BaseModel):
    """Wizard answers normalised into something the search and ranker can use."""

    model_config = ConfigDict(frozen=True)

    use_case: UseCase
    workload_mode: WorkloadMode = WorkloadMode.INTERACTIVE
    priority: RecommendationPriority
    languages: list[str] = Field(default_factory=list)
    concurrent_users: int = Field(default=1, ge=1)
    concurrency_range: str = "One user"
    desired_context: int = Field(gt=0)
    expected_input_tokens: int | None = Field(default=None, gt=0)
    expected_output_tokens: int | None = Field(default=None, gt=0)
    min_generation_tps: float | None = Field(default=None, gt=0)
    max_ttft_ms: float | None = Field(default=None, gt=0)
    commercial_use_required: bool | None = None
    pipeline_tag: str
    preferred_formats: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _legacy_batch(cls, value: Any) -> Any:
        return _normalize_legacy_batch(value)

    @property
    def workload_profile(self) -> WorkloadProfile:
        return WorkloadProfile(
            context_length=self.desired_context,
            concurrent_users=self.concurrent_users,
            mode=self.workload_mode,
            expected_input_tokens=self.expected_input_tokens,
            expected_output_tokens=self.expected_output_tokens,
            min_generation_tps=self.min_generation_tps,
            max_ttft_ms=self.max_ttft_ms,
        )


__all__ = [
    "CommercialUse",
    "ConcurrencyLevel",
    "DocumentScale",
    "RecommendationPriority",
    "UseCase",
    "UserAnswers",
    "UserRequirements",
    "WorkloadMode",
    "WorkloadProfile",
]
