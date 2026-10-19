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
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.inference import TargetDevice, WeightPrecision
from jaull.domain.model import ModelAnalysis
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeCapability,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.domain.runtime_support import available_compute_backends, technical_support_issue

InspectModelFn = object


def resolve_model_identity(
    *,
    candidate: ModelCandidate,
    analysis: ModelAnalysis | None = None,
) -> ModelIdentity:
    """Resolve the logical model identity from structured metadata first."""

    declared_base = candidate.base_model_repo_id
    base_key = (declared_base or "").casefold()
    tags = [*candidate.tags, *(analysis.repo.tags if analysis is not None else [])]
    transformed = declared_base is not None and any(
        tag.casefold() in {
            f"base_model:finetune:{base_key}",
            f"base_model:adapter:{base_key}",
        }
        for tag in tags
    )
    raw_canonical = (
        candidate.base_model_repo_id
        if candidate.base_model_repo_id
        and not transformed
        and (
            _is_artifact_wrapper_repo(candidate)
            or (
                analysis is not None and analysis.classification.primary_type is RepositoryType.GGUF
            )
        )
        else candidate.repo_id
    )
    canonical = (
        raw_canonical
        if raw_canonical == candidate.base_model_repo_id
        else logical_model_repo_id(raw_canonical)
    )
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
        if canonical == candidate.repo_id:
            evidence.append(ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.REPOSITORY_ID,
                source=candidate.repo_id, value=candidate.repo_id,
                confidence=candidate.metadata_confidence,
            ))
    elif canonical != raw_canonical:
        evidence.append(
            ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.NAME_HEURISTIC,
                source=candidate.repo_id,
                value=canonical,
                confidence=EstimationConfidence.LOW,
            )
        )
        confidence = EstimationConfidence.LOW
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
    compatibility = _compatibility(artifact, runtime)
    selection = _execution_selection(runtime, hardware, backend_selection, memory_prediction)
    issue = None
    if hardware is not None and selection is not None:
        precision = (
            memory_prediction.inference_configuration.precision
            if memory_prediction is not None
            else None
        )
        if precision is None and artifact.precision in {item.value for item in WeightPrecision}:
            precision = WeightPrecision(artifact.precision)
        issue = technical_support_issue(
            runtime,
            backend=selection.selected_backend,
            hardware=hardware,
            quantization=artifact.quantization,
            precision=precision,
        )
        flags = {flag.name: flag.value for flag in runtime.flags}
        if (
            runtime.runtime is RuntimeName.LLAMA_CPP
            and flags.get("--n-gpu-layers") not in {None, "0"}
            and selection.selected_backend is ComputeBackend.CPU
        ):
            issue = "GPU layer offload requires an available hardware accelerator backend."
        if issue is not None:
            compatibility = PlanCompatibilityStatus.NOT_COMPATIBLE
    if (
        execution_readiness is not None
        and selection is not None
        and execution_readiness.selection.selected_backend is not selection.selected_backend
    ):
        execution_readiness = None
    return ExecutionPlan(
        plan_id=_plan_id(artifact, runtime),
        model_identity=model_identity,
        artifact=artifact,
        runtime=runtime,
        runtime_family=runtime.runtime,
        backend_selection=selection,
        runtime_capability=runtime_capability,
        execution_readiness=execution_readiness,
        memory_prediction=memory_prediction,
        hardware=hardware,
        compatibility=compatibility,
        evidence=_plan_evidence(artifact, execution_readiness),
        warnings=[
            *_plan_warnings(artifact, runtime, execution_readiness),
            *([issue] if issue else []),
        ],
    )


def _execution_selection(
    runtime: RuntimeRecommendation,
    hardware: HardwareProfile | None,
    selection: RuntimeBackendSelection | None,
    estimate: MemoryEstimate | None,
) -> RuntimeBackendSelection | None:
    """Align backend provenance with the structured placement, never a binary probe."""
    if hardware is None:
        return selection
    available = available_compute_backends(hardware)
    flags = {flag.name: flag.value for flag in runtime.flags}
    device = flags.get("device_map")
    ngl = flags.get("--n-gpu-layers")
    if device == "cpu" or ngl == "0":
        backend = ComputeBackend.CPU
    elif device == "cuda":
        backend = (
            selection.selected_backend
            if selection is not None
            and selection.selected_backend in {ComputeBackend.CUDA, ComputeBackend.HIP}
            else ComputeBackend.CUDA if ComputeBackend.CUDA in available
            else ComputeBackend.HIP if ComputeBackend.HIP in available
            else ComputeBackend.CUDA
        )
    elif (
        estimate is not None
        and estimate.assessment.effective_device is TargetDevice.CPU
        and ngl is None
    ):
        backend = ComputeBackend.CPU
    elif selection is not None:
        backend = selection.selected_backend
    else:
        supported = available - {ComputeBackend.CPU}
        if runtime.runtime is RuntimeName.TRANSFORMERS:
            supported.discard(ComputeBackend.VULKAN)
        backend = next(
            (
                item
                for item in (ComputeBackend.CUDA, ComputeBackend.HIP, ComputeBackend.VULKAN)
                if item in supported
            ),
            ComputeBackend.CPU,
        )
    if selection is not None and selection.selected_backend is backend:
        return selection
    if selection is not None:
        alternative = next(
            (item for item in selection.alternatives if item.backend is backend),
            None,
        )
        if alternative is not None:
            return RuntimeBackendSelection(
                selected_backend=backend,
                selected_accelerator=alternative.accelerator,
                backend_info=alternative.backend_info,
                reason=alternative.reason,
            )
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=(
            RuntimeBackendSelectionReason.CPU_FALLBACK
            if backend is ComputeBackend.CPU
            else RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE
        ),
    )


def _compatibility(
    artifact: ArtifactVariant,
    runtime: RuntimeRecommendation,
) -> PlanCompatibilityStatus:
    """Whether the artifact and the runtime fit each other.

    Deliberately blind to what is installed locally: a GGUF file is compatible
    with llama.cpp whether or not llama-cli exists on this machine. Whether the
    plan can be launched *now* is ``ExecutionPlan.execution_readiness``.
    """

    if runtime.runtime not in artifact.compatible_runtimes:
        return PlanCompatibilityStatus.NOT_COMPATIBLE
    return PlanCompatibilityStatus.COMPATIBLE


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
    if readiness is not None and readiness.status in {
        ExecutionReadinessStatus.NOT_READY,
        ExecutionReadinessStatus.UNKNOWN,
    }:
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
