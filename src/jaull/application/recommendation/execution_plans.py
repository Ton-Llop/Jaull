"""Build execution-plan artifacts from recommendation results."""

from __future__ import annotations

from jaull.domain.artifact_profile import ArtifactFormat
from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    IdentityMatchStatus,
    ModelIdentity,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import GgufVariant, ModelAnalysis
from jaull.domain.runtime import (
    ExecutionReadiness,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution_plans.quantization import packed_transformers_quantization_label
from jaull.execution_plans.service import build_execution_plan, resolve_model_identity
from jaull.recommendation.models import ModelRecommendation


def variant_from_recommendation(
    recommendation: ModelRecommendation,
    *,
    identity: ModelIdentity | None = None,
) -> ArtifactVariant:
    evaluated = recommendation.evaluated
    effective_identity = identity or resolve_model_identity(
        candidate=evaluated.candidate,
        analysis=evaluated.analysis,
    )
    config = evaluated.selected_configuration
    profile = evaluated.artifact_profile
    analysis = evaluated.analysis
    quantization = config.quantization if config is not None else None

    if quantization:
        gguf = _gguf_variant(analysis, quantization)
        return ArtifactVariant(
            model_identity=effective_identity,
            repo_id=recommendation.repo_id,
            revision="main",
            format=ArtifactVariantFormat.GGUF,
            filename=gguf.files[0].path if gguf and gguf.files else None,
            size_bytes=gguf.total_bytes if gguf is not None else None,
            quantization=quantization,
            source="recommendation",
            compatible_runtimes=[RuntimeName.LLAMA_CPP],
            identity_match=IdentityMatchStatus.CONFIRMED,
            confidence=EstimationConfidence.HIGH,
            evidence=list(effective_identity.evidence),
        )

    packed_quantization = packed_transformers_quantization_label(
        analysis.config if analysis is not None else None
    )
    precision = None
    if packed_quantization is None and config is not None and config.precision:
        precision = config.precision.value
    format_ = ArtifactVariantFormat.TRANSFORMERS
    if profile is not None and profile.format is ArtifactFormat.NATIVE:
        format_ = ArtifactVariantFormat.SAFETENSORS
    elif analysis is not None:
        filenames = {file.path for file in analysis.files}
        if any(name.endswith(".safetensors") for name in filenames):
            format_ = ArtifactVariantFormat.SAFETENSORS
    return ArtifactVariant(
        model_identity=effective_identity,
        repo_id=recommendation.repo_id,
        revision="main",
        format=format_,
        filename=recommendation.repo_id,
        size_bytes=analysis.total_size_bytes if analysis is not None else None,
        quantization=packed_quantization,
        precision=precision,
        source="recommendation",
        compatible_runtimes=[RuntimeName.TRANSFORMERS],
        identity_match=IdentityMatchStatus.CONFIRMED,
        confidence=recommendation.confidence,
        evidence=list(effective_identity.evidence),
    )


def execution_plan_for_recommendation(
    recommendation: ModelRecommendation,
    *,
    hardware: HardwareProfile | None = None,
    backend_selection: RuntimeBackendSelection | None = None,
    runtime_capability: RuntimeCapability | None = None,
    execution_readiness: ExecutionReadiness | None = None,
) -> ExecutionPlan:
    if (
        recommendation.plan is not None
        and hardware is None
        and backend_selection is None
        and runtime_capability is None
        and execution_readiness is None
    ):
        return recommendation.plan
    estimate: MemoryEstimate | None = recommendation.evaluated.memory_estimate
    runtime = estimate.runtime_recommendation if estimate is not None else None
    if runtime is None:
        runtime = RuntimeRecommendation(
            runtime=RuntimeName.UNKNOWN,
            confidence=EstimationConfidence.UNKNOWN,
            warnings=["No runtime recommendation was available."],
        )
    identity = resolve_model_identity(
        candidate=recommendation.evaluated.candidate,
        analysis=recommendation.evaluated.analysis,
    )
    variant = variant_from_recommendation(recommendation, identity=identity)
    return build_execution_plan(
        model_identity=identity,
        artifact=variant,
        runtime=runtime,
        memory_prediction=estimate,
        hardware=hardware,
        backend_selection=backend_selection,
        runtime_capability=runtime_capability,
        execution_readiness=execution_readiness,
    )


def _gguf_variant(analysis: ModelAnalysis | None, quantization: str) -> GgufVariant | None:
    if analysis is None:
        return None
    for variant in analysis.classification.gguf_variants:
        if variant.quantization == quantization:
            return variant
    return None


__all__ = ["execution_plan_for_recommendation", "variant_from_recommendation"]
