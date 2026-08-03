"""The state a guided run carries from welcome to results.

Pure pydantic with no Textual or Rich import: the TUI renders this, the
orchestrator produces it, and tests assert on it directly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jaull.discovery.models import EvaluatedCandidate, ModelCandidate
from jaull.domain.hardware import HardwareProfile
from jaull.recommendation.models import ModelRecommendation
from jaull.workflow.models import (
    UserAnswers,
    UserRequirements,
    WorkflowProgress,
    WorkflowStep,
)


class RecommendationWorkflowState(BaseModel):
    """Everything known about one guided run."""

    model_config = ConfigDict(frozen=True)

    current_step: WorkflowStep = WorkflowStep.WELCOME
    hardware: HardwareProfile | None = None
    answers: UserAnswers | None = None
    requirements: UserRequirements | None = None
    candidates: list[ModelCandidate] = Field(default_factory=list)
    evaluated_candidates: list[EvaluatedCandidate] = Field(default_factory=list)
    recommendations: list[ModelRecommendation] = Field(default_factory=list)
    progress: WorkflowProgress = Field(default_factory=WorkflowProgress)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # Query labels actually issued, so the report can show the search strategy.
    search_queries: list[str] = Field(default_factory=list)
    # Set when the run ended with nothing to recommend but no hard failure.
    no_results_reason: list[str] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.current_step is WorkflowStep.FAILED

    @property
    def completed(self) -> bool:
        return self.current_step is WorkflowStep.COMPLETED

    @property
    def primary(self) -> ModelRecommendation | None:
        return self.recommendations[0] if self.recommendations else None

    @property
    def alternatives(self) -> list[ModelRecommendation]:
        return self.recommendations[1:]

    def advance(self, step: WorkflowStep) -> RecommendationWorkflowState:
        return self.model_copy(update={"current_step": step})

    def with_error(self, message: str) -> RecommendationWorkflowState:
        return self.model_copy(
            update={
                "current_step": WorkflowStep.FAILED,
                "errors": [*self.errors, message],
            }
        )


__all__ = ["RecommendationWorkflowState"]
