"""Final output models: what the user actually sees."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from local_ai_check.discovery.models import EvaluatedCandidate
from local_ai_check.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from local_ai_check.recommendation.policies import LicenseCategory


class ScoreBreakdown(BaseModel):
    """The composite score with its parts, so a ranking can be argued with."""

    model_config = ConfigDict(frozen=True)

    total: float = Field(ge=0.0, le=1.0)
    hardware_fit: float = 0.0
    task_match: float = 0.0
    language_match: float = 0.0
    license: float = 0.0
    metadata_quality: float = 0.0
    popularity: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)

    @property
    def out_of_100(self) -> int:
        return round(self.total * 100)


class ModelRecommendation(BaseModel):
    """One recommended model, ready to render."""

    model_config = ConfigDict(frozen=True)

    rank: int
    evaluated: EvaluatedCandidate
    score: ScoreBreakdown
    status: CompatibilityStatus
    confidence: EstimationConfidence
    license_category: LicenseCategory
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Repos folded into this one because they are the same model in another format.
    related_repositories: list[str] = Field(default_factory=list)
    # Short label such as "Smaller and faster"; only set when it is actually true.
    alternative_label: str | None = None

    @property
    def repo_id(self) -> str:
        return self.evaluated.repo_id

    @property
    def is_primary(self) -> bool:
        return self.rank == 1


__all__ = ["ModelRecommendation", "ScoreBreakdown"]
