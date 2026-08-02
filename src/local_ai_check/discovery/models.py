"""Normalised view of a model found through Hugging Face search.

`ModelCandidate` is the cheap, search-time view: everything here comes from a
single `list_models` page, with no per-repository API call. The expensive
`EvaluatedCandidate` view is produced later by `discovery.enrichment`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from local_ai_check.domain.enums import RepositoryType
from local_ai_check.domain.estimation import (
    CompatibilityAssessment,
    EstimationConfidence,
    MemoryEstimate,
)
from local_ai_check.domain.inference import InferenceConfiguration
from local_ai_check.domain.model import ModelAnalysis


class SearchQuery(BaseModel):
    """One `list_models` call, described declaratively so it can be asserted on."""

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
    hardware_fit_score: float = 0.0
    license_score: float = 0.0
    metadata_quality_score: float = 0.0
    popularity_score: float = 0.0

    warnings: list[str] = Field(default_factory=list)
    failed: bool = False

    @property
    def repo_id(self) -> str:
        return self.candidate.repo_id


__all__ = ["EvaluatedCandidate", "ModelCandidate", "SearchQuery"]
