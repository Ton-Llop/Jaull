"""Search-time and evaluation-time contracts for a candidate model.

``ModelCandidate`` is the cheap, search-time view — everything here comes from a
single ``list_models`` page, no per-repository API call. ``EvaluatedCandidate``
is the enriched view after inspection, estimation and scoring. Both are the
public contract that discovery, recommendation and workflow share, so they
live in ``domain/`` rather than inside any single feature package.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# Side-effect import: loading ``jaull.domain.runtime`` resolves the
# forward-ref on ``MemoryEstimate.runtime_recommendation`` so isolated unit
# tests can construct ``EvaluatedCandidate`` without Pydantic tripping on
# "class not fully defined".
from jaull.domain import runtime as _runtime  # noqa: F401
from jaull.domain.artifact_profile import ArtifactProfile
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis
from jaull.domain.parameters import ParameterCount
from jaull.domain.runtime import RuntimeAssessment


class SearchQuery(BaseModel):
    """One ``list_models`` call, described declaratively so it can be asserted on."""

    model_config = ConfigDict(frozen=True)

    label: str
    search: str | None = None
    pipeline_tag: str | None = None
    filter_tags: tuple[str, ...] = ()
    sort: str = "downloads"
    limit: int = 20

    def cache_key(self) -> tuple[str | None, str | None, tuple[str, ...], str, int]:
        return (self.search, self.pipeline_tag, self.filter_tags, self.sort, self.limit)


class ModelCandidate(BaseModel):
    """A repository worth considering, before any deep inspection."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    license: str | None = None
    languages: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    gated: bool | str = False
    private: bool = False
    repository_type: RepositoryType | None = None
    base_model_repo_id: str | None = None
    revision_hint: str | None = None
    last_modified: datetime | None = None
    source_queries: list[str] = Field(default_factory=list)
    metadata_confidence: EstimationConfidence = EstimationConfidence.MEDIUM
    # Set when preliminary filtering keeps a candidate but with reservations.
    penalties: list[str] = Field(default_factory=list)

    def merged_with(self, other: ModelCandidate) -> ModelCandidate:
        """Combine two sightings of the same repo, keeping the richer metadata."""
        queries = list(self.source_queries)
        for query in other.source_queries:
            if query not in queries:
                queries.append(query)
        return self.model_copy(
            update={
                "source_queries": queries,
                "license": self.license or other.license,
                "pipeline_tag": self.pipeline_tag or other.pipeline_tag,
                "library_name": self.library_name or other.library_name,
                "languages": self.languages or other.languages,
                "tags": self.tags or other.tags,
                "downloads": max(self.downloads, other.downloads),
                "likes": max(self.likes, other.likes),
                "revision_hint": self.revision_hint or other.revision_hint,
                "last_modified": self.last_modified or other.last_modified,
            }
        )


class EvaluatedCandidate(BaseModel):
    """A candidate after inspection, estimation and scoring."""

    model_config = ConfigDict(frozen=True)

    candidate: ModelCandidate
    analysis: ModelAnalysis | None = None
    selected_configuration: InferenceConfiguration | None = None
    memory_estimate: MemoryEstimate | None = None
    compatibility: CompatibilityAssessment | None = None
    configuration_reason: str | None = None
    alternatives_considered: list[str] = Field(default_factory=list)

    task_match_score: float = 0.0
    language_match_score: float = 0.0
    # ``hardware_fit_score`` is kept as ``min(memory_fit_score, concurrency_fit_score)``
    # for backwards compatibility. New code should read the two axes directly.
    hardware_fit_score: float = 0.0
    memory_fit_score: float = 0.0
    concurrency_fit_score: float = 0.0
    capability_score: float = 0.0
    artifact_realism_score: float = 1.0
    license_score: float = 0.0
    metadata_quality_score: float = 0.0
    popularity_score: float = 0.0

    warnings: list[str] = Field(default_factory=list)
    failed: bool = False
    # Populated by ``recommendation.requirements_gate`` at ranking time.
    # Kept as plain scalars/list so this module stays free of ``recommendation``
    # imports (which would be a circular dependency).
    requirement_penalty: float = 1.0
    unmet_requirement_labels: list[str] = Field(default_factory=list)

    # Optional evidence populated during evaluation. All three are ``None`` on
    # freshly-constructed candidates (search results); ``enrichment`` fills
    # them once the analysis and configuration are available.
    parameter_count_info: ParameterCount | None = None
    artifact_profile: ArtifactProfile | None = None
    runtime_assessment: RuntimeAssessment | None = None

    @property
    def repo_id(self) -> str:
        return self.candidate.repo_id


__all__ = ["EvaluatedCandidate", "ModelCandidate", "SearchQuery"]
