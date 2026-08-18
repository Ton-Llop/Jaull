"""Logical-model identities, artifact variants and execution plans."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import (
    ExecutionReadiness,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeName,
    RuntimeRecommendation,
)


class ModelIdentityEvidenceKind(StrEnum):
    REPOSITORY_ID = "repository_id"
    BASE_MODEL_METADATA = "base_model_metadata"
    CONFIG_ARCHITECTURE = "config_architecture"
    PARAMETER_METADATA = "parameter_metadata"
    NAME_HEURISTIC = "name_heuristic"


class ArtifactVariantFormat(StrEnum):
    TRANSFORMERS = "transformers"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class IdentityMatchStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class PlanCompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    NOT_COMPATIBLE = "not_compatible"
    UNKNOWN = "unknown"


class ModelIdentityEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ModelIdentityEvidenceKind
    source: str
    value: str
    confidence: EstimationConfidence


class ModelIdentity(BaseModel):
    """Identity of the logical model independent from a concrete artifact."""

    model_config = ConfigDict(frozen=True)

    canonical_repo_id: str | None = None
    family: str | None = None
    model_name: str
    parameter_count: int | None = Field(default=None, ge=0)
    variant: str | None = None
    architecture: str | None = None
    confidence: EstimationConfidence = EstimationConfidence.UNKNOWN
    evidence: list[ModelIdentityEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_evidence_for_non_unknown(self) -> Self:
        if self.confidence is not EstimationConfidence.UNKNOWN and not self.evidence:
            raise ValueError("ModelIdentity with confidence requires evidence")
        return self


_ARTIFACT_REPOSITORY_SUFFIXES = (
    "gguf",
    "awq",
    "gptq",
    "onnx",
    "hf",
    "quantized",
    "bnb",
    "exl2",
    "mlx",
)


def logical_model_repo_id(repo_id: str) -> str:
    """Return the repository identity without an artifact-format suffix.

    Repository casing is preserved for display and provenance. Use
    :func:`logical_model_repo_key` when comparing identities.
    """

    value = repo_id.strip()
    if "/" not in value:
        return _strip_artifact_repository_suffix(value)
    owner, name = value.split("/", 1)
    return f"{owner}/{_strip_artifact_repository_suffix(name)}"


def logical_model_repo_key(repo_id: str) -> str:
    """Stable comparison key for a logical model repository."""

    return logical_model_repo_id(repo_id).casefold()


def model_identity_key(identity: ModelIdentity) -> str:
    """Identify a logical model independently from artifacts and evidence.

    ``ModelIdentity`` is a rich evidence-bearing value object, so equality of
    the complete object is intentionally stricter than logical-model equality.
    Grouping recommendations must use this key instead.
    """

    if identity.canonical_repo_id:
        return f"repo:{logical_model_repo_key(identity.canonical_repo_id)}"
    family = (identity.family or "").strip().casefold()
    model_name = _strip_artifact_repository_suffix(identity.model_name.strip()).casefold()
    variant = (identity.variant or "").strip().casefold()
    architecture = (identity.architecture or "").strip().casefold()
    parameters = "" if identity.parameter_count is None else str(identity.parameter_count)
    return f"metadata:{family}|{model_name}|{parameters}|{variant}|{architecture}"


def _strip_artifact_repository_suffix(name: str) -> str:
    lowered = name.casefold()
    for suffix in _ARTIFACT_REPOSITORY_SUFFIXES:
        for separator in ("-", "_"):
            token = f"{separator}{suffix}"
            if lowered.endswith(token):
                return name[: -len(token)]
    return name


class ArtifactVariant(BaseModel):
    """Metadata for one available representation of a logical model.

    This is discovery metadata. It intentionally does not imply that the
    artifact has been downloaded or verified.
    """

    model_config = ConfigDict(frozen=True)

    model_identity: ModelIdentity
    repo_id: str
    revision: str | None = None
    format: ArtifactVariantFormat
    filename: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    quantization: str | None = None
    precision: str | None = None
    source: str = "huggingface"
    compatible_runtimes: list[RuntimeName] = Field(default_factory=list)
    identity_match: IdentityMatchStatus = IdentityMatchStatus.CONFIRMED
    confidence: EstimationConfidence = EstimationConfidence.UNKNOWN
    evidence: list[ModelIdentityEvidence] = Field(default_factory=list)

    @property
    def label(self) -> str:
        if self.quantization:
            return f"{self.format.value} {self.quantization}"
        if self.precision:
            return f"{self.format.value} {self.precision}"
        return self.format.value

    def to_model_artifact(self) -> ModelArtifact:
        filename = self.filename or self.repo_id
        return ModelArtifact(
            repo_id=self.repo_id,
            revision=self.revision or "main",
            filename=filename,
            format=self.format.value,
            quantization=self.quantization,
            size_bytes=self.size_bytes,
        )


class ExecutionPlan(BaseModel):
    """One concrete way Jaull could attempt to execute a model artifact."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    model_identity: ModelIdentity
    artifact: ArtifactVariant
    runtime: RuntimeRecommendation
    runtime_family: RuntimeName
    backend_selection: RuntimeBackendSelection | None = None
    runtime_capability: RuntimeCapability | None = None
    execution_readiness: ExecutionReadiness | None = None
    memory_prediction: MemoryEstimate | None = None
    hardware: HardwareProfile | None = None
    compatibility: PlanCompatibilityStatus = PlanCompatibilityStatus.UNKNOWN
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _runtime_matches_recommendation(self) -> Self:
        if self.runtime_family is not self.runtime.runtime:
            raise ValueError("ExecutionPlan runtime_family must match runtime.runtime")
        if self.artifact.model_identity != self.model_identity:
            raise ValueError("ExecutionPlan model_identity must match artifact identity")
        return self


class PreparedExecutionPlan(BaseModel):
    """Execution plan after on-demand preparation for a concrete action."""

    model_config = ConfigDict(frozen=True)

    plan: ExecutionPlan
    artifact: ModelArtifact


__all__ = [
    "ArtifactVariant",
    "ArtifactVariantFormat",
    "ExecutionPlan",
    "IdentityMatchStatus",
    "ModelIdentity",
    "ModelIdentityEvidence",
    "ModelIdentityEvidenceKind",
    "PlanCompatibilityStatus",
    "PreparedExecutionPlan",
    "logical_model_repo_id",
    "logical_model_repo_key",
    "model_identity_key",
]
