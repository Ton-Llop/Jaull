"""Evidence-based recommendation domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution_plans import ExecutionPlan


class RecommendationEvidenceCategory(StrEnum):
    IDENTITY = "identity"
    METADATA = "metadata"
    HARDWARE = "hardware"
    EXECUTION = "execution"
    LOCAL = "local"
    EXTERNAL = "external"
    INFERENCE = "inference"


class RecommendationEvidenceSource(StrEnum):
    SEARCH_METADATA = "search_metadata"
    MODEL_CARD_METADATA = "model_card_metadata"
    MODEL_ANALYSIS = "model_analysis"
    MEMORY_ESTIMATOR = "memory_estimator"
    RUNTIME_CAPABILITY = "runtime_capability"
    EXECUTION_READINESS = "execution_readiness"
    LOCAL_EXPERIMENT = "local_experiment"
    LOCAL_BENCHMARK = "local_benchmark"
    EXTERNAL_EVALUATION = "external_evaluation"
    HEURISTIC = "heuristic"
    POLICY = "policy"


class RecommendationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: RecommendationEvidenceCategory
    key: str
    value: str
    source: RecommendationEvidenceSource
    confidence: EstimationConfidence = EstimationConfidence.UNKNOWN
    observed: bool = False
    provenance: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ExternalEvaluationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark: str
    task: str
    value: float | str
    metric: str
    source: str
    confidence: EstimationConfidence = EstimationConfidence.UNKNOWN
    verified: bool | None = None
    revision: str | None = None
    date: str | None = None
    notes: list[str] = Field(default_factory=list)


class AssessmentLevel(StrEnum):
    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class RecommendationPosition(StrEnum):
    BEST_OVERALL = "best_overall"
    HIGHER_QUALITY = "higher_quality"
    FASTER = "faster"
    LIGHTER = "lighter"
    ALTERNATIVE = "alternative"


class HardConstraintCode(StrEnum):
    MEMORY_INSUFFICIENT = "memory_insufficient"
    LICENSE_INCOMPATIBLE = "license_incompatible"
    LANGUAGE_INCOMPATIBLE = "language_incompatible"
    RUNTIME_NOT_READY = "runtime_not_ready"
    ARTIFACT_RUNTIME_INCOMPATIBLE = "artifact_runtime_incompatible"
    UNSUPPORTED_TASK = "unsupported_task"


class HardConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: HardConstraintCode
    message: str
    evidence_key: str | None = None


class PlanAssessment(BaseModel):
    """Multidimensional assessment of one concrete execution plan.

    Deliberately no global numeric score: ranking policy consumes the dimensions,
    but explanations keep the axes visible.
    """

    model_config = ConfigDict(frozen=True)

    plan_id: str
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    suitability: AssessmentLevel = AssessmentLevel.UNKNOWN
    feasibility: AssessmentLevel = AssessmentLevel.UNKNOWN
    executability: AssessmentLevel = AssessmentLevel.UNKNOWN
    performance_evidence: AssessmentLevel = AssessmentLevel.UNKNOWN
    resource_evidence: AssessmentLevel = AssessmentLevel.UNKNOWN
    license_fit: AssessmentLevel = AssessmentLevel.UNKNOWN
    language_fit: AssessmentLevel = AssessmentLevel.UNKNOWN
    confidence: EstimationConfidence = EstimationConfidence.UNKNOWN
    evidence: list[RecommendationEvidence] = Field(default_factory=list)
    external_evaluations: list[ExternalEvaluationEvidence] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    measured_memory_bytes: int | None = Field(default=None, ge=0)
    estimated_memory_bytes: int | None = Field(default=None, ge=0)
    measured_generation_tokens_per_second: float | None = Field(default=None, ge=0.0)
    local_experiment_id: str | None = None
    local_benchmark_id: str | None = None

    @property
    def rejected(self) -> bool:
        return bool(self.hard_constraints)


class ExecutionPlanRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    plan: ExecutionPlan
    assessment: PlanAssessment
    reasons: list[str] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)
    confidence: EstimationConfidence = EstimationConfidence.UNKNOWN


__all__ = [
    "AssessmentLevel",
    "ExecutionPlanRecommendation",
    "ExternalEvaluationEvidence",
    "HardConstraint",
    "HardConstraintCode",
    "PlanAssessment",
    "RecommendationEvidence",
    "RecommendationEvidenceCategory",
    "RecommendationEvidenceSource",
    "RecommendationPosition",
]
