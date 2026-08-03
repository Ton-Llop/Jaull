"""Final output models: what the user actually sees."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jaull.discovery.models import EvaluatedCandidate
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from jaull.recommendation.policies import LicenseCategory


class ScoreBreakdown(BaseModel):
    """The composite score with its parts, so a ranking can be argued with."""

    model_config = ConfigDict(frozen=True)

    total: float = Field(ge=0.0, le=1.0)
    memory_fit: float = 0.0
    concurrency_fit: float = 0.0
    capability: float = 0.0
    task_match: float = 0.0
    language_match: float = 0.0
    license: float = 0.0
    metadata_quality: float = 0.0
    popularity: float = 0.0
    # ``hardware_fit`` equals ``min(memory_fit, concurrency_fit)`` and stays in
    # the JSON export so pre-Fase-5 consumers keep a familiar number.
    hardware_fit: float = 0.0
    # < 1 when a hard requirement is unmet. 1.0 means the ranker only had to
    # multiply by the confidence scaling.
    hard_penalty: float = 1.0
    unmet_requirements: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)

    @property
    def out_of_100(self) -> int:
        return round(self.total * 100)


class SeriesSibling(BaseModel):
    """Another size in the same model family, shown next to a recommendation."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    parameter_count: int | None = None
    parameter_label: str  # "0.5B", "7B", "70B"
    status: CompatibilityStatus
    total_bytes: int | None = None
    note: str  # "Smaller (1.5B, comfortable)", "Larger (7B, tight fit)", ...


class ModelRecommendation(BaseModel):
    """One recommended model, ready to render."""

    model_config = ConfigDict(frozen=True)

    rank: int
    evaluated: EvaluatedCandidate
    score: ScoreBreakdown
    status: CompatibilityStatus
    confidence: EstimationConfidence
    license_category: LicenseCategory
    # Fase 5: heading tier and same-series alternatives, both aditivos.
    tier: str = "recommended"
    series_alternatives: list[SeriesSibling] = Field(default_factory=list)
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


__all__ = ["ModelRecommendation", "ScoreBreakdown", "SeriesSibling"]
