"""User-facing requirement models shared by discovery, recommendation and workflow.

These are the pure-domain enums and Pydantic models. Progress-tracking state
(WorkflowStep, StepStatus, ProgressStep, WorkflowProgress) lives in
``workflow/models.py`` because it is orchestration concern, not domain.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class UseCase(StrEnum):
    GENERAL_CHAT = "general_chat"
    CODING = "coding"
    DOCUMENT_QA = "document_qa"
    WRITING_TRANSLATION = "writing_translation"


class RecommendationPriority(StrEnum):
    QUALITY = "quality"
    BALANCED = "balanced"
    SPEED = "speed"
    MEMORY = "memory"


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


class UserAnswers(BaseModel):
    """Raw wizard answers, exactly as the user gave them.

    Kept separate from :class:`UserRequirements` so the report can show both what
    was asked and what the tool inferred from it.
    """

    model_config = ConfigDict(frozen=True)

    use_case: UseCase
    priority: RecommendationPriority
    languages: list[str] = Field(default_factory=list)
    other_languages: list[str] = Field(default_factory=list)
    concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE
    document_scale: DocumentScale | None = None
    commercial_use: CommercialUse = CommercialUse.YES


class UserRequirements(BaseModel):
    """Wizard answers normalised into something the search and ranker can use."""

    model_config = ConfigDict(frozen=True)

    use_case: UseCase
    priority: RecommendationPriority
    languages: list[str] = Field(default_factory=list)
    concurrent_users: int = Field(default=1, ge=1)
    concurrency_range: str = "One user"
    desired_context: int = Field(gt=0)
    commercial_use_required: bool | None = None
    pipeline_tag: str
    preferred_formats: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


__all__ = [
    "CommercialUse",
    "ConcurrencyLevel",
    "DocumentScale",
    "RecommendationPriority",
    "UseCase",
    "UserAnswers",
    "UserRequirements",
]
