"""Domain models for the guided workflow.

Deliberately free of Textual and Rich: the wizard answers, the normalised
requirements and the progress model are all plain pydantic so the orchestrator
and its tests never need a terminal.
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


class WorkflowStep(StrEnum):
    WELCOME = "welcome"
    HARDWARE_SCAN = "hardware_scan"
    REQUIREMENTS = "requirements"
    CANDIDATE_DISCOVERY = "candidate_discovery"
    CANDIDATE_EVALUATION = "candidate_evaluation"
    RANKING = "ranking"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProgressStep(BaseModel):
    """One line of the progress display."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    status: StepStatus = StepStatus.PENDING
    detail: str | None = None


class WorkflowProgress(BaseModel):
    """Ordered list of steps plus the currently active one."""

    model_config = ConfigDict(frozen=True)

    steps: list[ProgressStep] = Field(default_factory=list)
    current_key: str | None = None

    def with_step(
        self, key: str, status: StepStatus, detail: str | None = None
    ) -> WorkflowProgress:
        """Return a copy with ``key`` updated. Unknown keys are appended."""
        updated: list[ProgressStep] = []
        seen = False
        for step in self.steps:
            if step.key == key:
                seen = True
                updated.append(
                    step.model_copy(
                        update={
                            "status": status,
                            "detail": detail if detail is not None else step.detail,
                        }
                    )
                )
            else:
                updated.append(step)
        if not seen:
            updated.append(
                ProgressStep(key=key, label=key, status=status, detail=detail)
            )
        current = key if status is StepStatus.RUNNING else self.current_key
        return WorkflowProgress(steps=updated, current_key=current)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status is StepStatus.DONE)


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
    "ProgressStep",
    "RecommendationPriority",
    "StepStatus",
    "UseCase",
    "UserAnswers",
    "UserRequirements",
    "WorkflowProgress",
    "WorkflowStep",
]
