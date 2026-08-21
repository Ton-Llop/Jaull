"""Build logical-model identities, artifact variants and execution plans."""

from __future__ import annotations

import re

from jaull.domain.candidates import ModelCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ExecutionPlan,
    IdentityMatchStatus,
    ModelIdentity,
    ModelIdentityEvidence,
    ModelIdentityEvidenceKind,
    PlanCompatibilityStatus,
    logical_model_repo_id,
    logical_model_repo_key,
)
from jaull.domain.families import detect_family, resolve_parameter_count
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessStatus,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeRecommendation,
)

InspectModelFn = object


def resolve_model_identity(
    *,
    candidate: ModelCandidate,
    analysis: ModelAnalysis | None = None,
) -> ModelIdentity:
    """Resolve the logical model identity from structured metadata first."""

    raw_canonical = (
        candidate.base_model_repo_id
        if candidate.base_model_repo_id and _is_artifact_wrapper_repo(candidate)
        else candidate.repo_id
    )
    canonical = logical_model_repo_id(raw_canonical)
    evidence: list[ModelIdentityEvidence] = []
    confidence = candidate.metadata_confidence
    if candidate.base_model_repo_id:
        evidence.append(
            ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.BASE_MODEL_METADATA,
                source=candidate.repo_id,
                value=candidate.base_model_repo_id,
                confidence=EstimationConfidence.HIGH,
            )
        )
        confidence = EstimationConfidence.HIGH
    elif canonical != raw_canonical:
        evidence.append(
            ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.NAME_HEURISTIC,
                source=candidate.repo_id,
                value=canonical,
                confidence=EstimationConfidence.LOW,
            )
        )
    else:
        evidence.append(
            ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.REPOSITORY_ID,
                source=candidate.repo_id,
                value=candidate.repo_id,
                confidence=confidence,
            )
        )

    architecture = None
    if analysis is not None and analysis.config is not None:
        architecture = next(iter(analysis.config.architectures), None)
        if architecture is not None:
            evidence.append(
                ModelIdentityEvidence(
                    kind=ModelIdentityEvidenceKind.CONFIG_ARCHITECTURE,
                    source=analysis.repo.repo_id,
                    value=architecture,
                    confidence=EstimationConfidence.MEDIUM,
                )
            )

    parameter_info = resolve_parameter_count(candidate, analysis)
    if parameter_info.count is not None:
        evidence.append(
            ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.PARAMETER_METADATA,
                source=parameter_info.source.value,
                value=str(parameter_info.count),
                confidence=parameter_info.confidence,
            )
        )

    return ModelIdentity(
        canonical_repo_id=canonical,
        family=detect_family(candidate, analysis),
        model_name=_repo_name(canonical),
        parameter_count=parameter_info.count,
        variant=_variant_from_name(canonical),
        architecture=architecture,
        confidence=confidence,
        evidence=evidence,
    )


def _is_artifact_wrapper_repo(candidate: ModelCandidate) -> bool:
    if candidate.repository_type in {
        RepositoryType.GGUF,
        RepositoryType.ONNX,
        RepositoryType.ADAPTER,
    }:
        return True
    repo_key = logical_model_repo_key(candidate.repo_id)
    return repo_key != candidate.repo_id.casefold()


def build_execution_plan(
    *,
    model_identity: ModelIdentity,
    artifact: ArtifactVariant,
    runtime: RuntimeRecommendation,
    memory_prediction: MemoryEstimate | None = None,
    hardware: HardwareProfile | None = None,
    backend_selection: RuntimeBackendSelection | None = None,
    runtime_capability: RuntimeCapability | None = None,
    execution_readiness: ExecutionReadiness | None = None,
) -> ExecutionPlan:
    compatibility = _compatibility(artifact, runtime, execution_readiness)
    return ExecutionPlan(
        plan_id=_plan_id(artifact, runtime),
        model_identity=model_identity,
        artifact=artifact,
        runtime=runtime,
        runtime_family=runtime.runtime,
        backend_selection=backend_selection,
        runtime_capability=runtime_capability,
        execution_readiness=execution_readiness,
        memory_prediction=memory_prediction,
        hardware=hardware,
        compatibility=compatibility,
        evidence=_plan_evidence(artifact, execution_readiness),
        warnings=_plan_warnings(artifact, runtime, execution_readiness),
    )


def _compatibility(
    artifact: ArtifactVariant,
    runtime: RuntimeRecommendation,
    readiness: ExecutionReadiness | None,
) -> PlanCompatibilityStatus:
    if runtime.runtime not in artifact.compatible_runtimes:
        return PlanCompatibilityStatus.NOT_COMPATIBLE
    if readiness is None:
        return PlanCompatibilityStatus.UNKNOWN
    if readiness.status is ExecutionReadinessStatus.READY:
        return PlanCompatibilityStatus.COMPATIBLE
    if readiness.status is ExecutionReadinessStatus.NOT_READY:
        return PlanCompatibilityStatus.NOT_COMPATIBLE
    return PlanCompatibilityStatus.UNKNOWN


def _plan_evidence(
    artifact: ArtifactVariant,
    readiness: ExecutionReadiness | None,
) -> list[str]:
    evidence = [
        f"artifact:{artifact.format.value}",
        f"identity_match:{artifact.identity_match.value}",
    ]
    if artifact.quantization:
        evidence.append(f"quantization:{artifact.quantization}")
    if readiness is not None:
        evidence.append(f"readiness:{readiness.status.value}")
        evidence.append(f"readiness_reason:{readiness.reason.value}")
    return evidence


def _plan_warnings(
    artifact: ArtifactVariant,
    runtime: RuntimeRecommendation,
    readiness: ExecutionReadiness | None,
) -> list[str]:
    warnings = list(runtime.warnings)
    if artifact.identity_match is IdentityMatchStatus.UNCERTAIN:
        warnings.append("Artifact identity match is heuristic and not automatically safe.")
    if readiness is not None and readiness.status is not ExecutionReadinessStatus.READY:
        warnings.append(readiness.message or f"Runtime readiness is {readiness.status.value}.")
    return warnings


def _plan_id(artifact: ArtifactVariant, runtime: RuntimeRecommendation) -> str:
    parts = [
        _slug(artifact.repo_id),
        _slug(artifact.filename or "repo"),
        artifact.format.value,
        artifact.quantization or artifact.precision or "default",
        runtime.runtime.value,
    ]
    return "plan-" + "-".join(_slug(part) for part in parts if part)


def _repo_name(repo_id: str) -> str:
    return repo_id.split("/")[-1] if repo_id else "unknown"


def _variant_from_name(repo_id: str) -> str | None:
    name = _repo_name(repo_id).lower()
    for token in ("instruct", "chat", "base"):
        if token in name:
            return token
    return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "unknown"


__all__ = [
    "build_execution_plan",
    "resolve_model_identity",
]
