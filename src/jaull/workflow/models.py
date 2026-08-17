"""Progress-tracking models for the guided workflow.

The user-facing requirement enums and pydantic models (UseCase, UserAnswers,
UserRequirements, ...) live in ``jaull.domain.requirements`` because they are
consumed by discovery and recommendation too. This module keeps only the
orchestration state that describes *how* the run is progressing.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
                next_detail = detail
                if next_detail is None and status is StepStatus.RUNNING:
                    next_detail = step.detail
                updated.append(
                    step.model_copy(
                        update={
                            "status": status,
                            "detail": next_detail,
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


__all__ = [
    "ProgressStep",
    "StepStatus",
    "WorkflowProgress",
    "WorkflowStep",
]
