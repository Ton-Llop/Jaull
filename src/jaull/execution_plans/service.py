"""Build logical-model identities, artifact variants and execution plans."""

from __future__ import annotations

import re

from jaull.discovery.search_client import ModelSearchClient
from jaull.domain.artifact_profile import ArtifactFormat
from jaull.domain.candidates import ModelCandidate, SearchQuery
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    IdentityMatchStatus,
    ModelIdentity,
    ModelIdentityEvidence,
    ModelIdentityEvidenceKind,
    PlanCompatibilityStatus,
    logical_model_repo_id,
    logical_model_repo_key,
    model_identity_key,
)
from jaull.domain.families import detect_family, resolve_parameter_count
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import GgufVariant, ModelAnalysis
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessStatus,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.recommendation.models import ModelRecommendation
from jaull.workflow import policies
from jaull.workflow.model_analysis_cache import ModelAnalysisCacheProtocol
from jaull.workflow.telemetry import PerformanceTelemetry

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

    precision = config.precision.value if config is not None and config.precision else None
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
        precision=precision,
        source="recommendation",
        compatible_runtimes=[RuntimeName.TRANSFORMERS],
        identity_match=IdentityMatchStatus.CONFIRMED,
        confidence=recommendation.confidence,
        evidence=list(effective_identity.evidence),
    )


def discover_artifact_variants(
    *,
    identity: ModelIdentity,
    current: ArtifactVariant | None,
    search_client: ModelSearchClient,
    inspect_model: object,
    analysis_cache: ModelAnalysisCacheProtocol | None = None,
    limit: int = 20,
    include_uncertain: bool = False,
    max_deep_inspection: int = policies.MAX_VARIANT_DEEP_INSPECTION,
    telemetry: PerformanceTelemetry | None = None,
) -> list[ArtifactVariant]:
    """Discover metadata-only artifact variants without downloading weights."""

    variants: list[ArtifactVariant] = []
    if current is not None:
        variants.append(current)
    query = SearchQuery(
        label="artifact-variants",
        search=f"{identity.model_name} GGUF",
        limit=limit,
    )
    if telemetry is not None:
        telemetry.increment("variant_search_api_calls")
    search_results = search_client.search(query)
    if telemetry is not None:
        telemetry.increment("variant_candidates_searched", len(search_results))
    candidates = _prioritize_variant_candidates(
        identity,
        search_results,
        include_uncertain=include_uncertain,
    )
    inspected = 0
    for candidate in candidates:
        if inspected >= max_deep_inspection:
            break
        analysis = _inspect_candidate(
            inspect_model=inspect_model,
            candidate=candidate,
            analysis_cache=analysis_cache,
            telemetry=telemetry,
        )
        inspected += 1
        if telemetry is not None:
            telemetry.increment("variant_candidates_deeply_inspected")
        candidate_identity = resolve_model_identity(candidate=candidate, analysis=analysis)
        match = _identity_match(identity, candidate, candidate_identity)
        if match is IdentityMatchStatus.REJECTED:
            continue
        if match is IdentityMatchStatus.UNCERTAIN and not include_uncertain:
            continue
        variants.extend(
            _variants_from_candidate(
                identity=identity,
                candidate=candidate,
                analysis=analysis,
                match=match,
            )
        )
        if (
            match is IdentityMatchStatus.CONFIRMED
            and _confirmed_gguf_quantization_count(variants) >= 3
        ):
            break
    return _dedupe_variants(variants)


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
    estimate = recommendation.evaluated.memory_estimate
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


def _variants_from_candidate(
    *,
    identity: ModelIdentity,
    candidate: ModelCandidate,
    analysis: ModelAnalysis,
    match: IdentityMatchStatus,
) -> list[ArtifactVariant]:
    variants: list[ArtifactVariant] = []
    if analysis.classification.primary_type is RepositoryType.GGUF:
        for gguf in analysis.classification.gguf_variants:
            variants.append(
                ArtifactVariant(
                    model_identity=identity,
                    repo_id=candidate.repo_id,
                    revision="main",
                    format=ArtifactVariantFormat.GGUF,
                    filename=gguf.files[0].path if gguf.files else None,
                    size_bytes=gguf.total_bytes,
                    quantization=gguf.quantization,
                    compatible_runtimes=[RuntimeName.LLAMA_CPP],
                    identity_match=match,
                    confidence=_match_confidence(match),
                    evidence=list(identity.evidence),
                )
            )
    if analysis.classification.primary_type is RepositoryType.TRANSFORMERS:
        variants.append(
            ArtifactVariant(
                model_identity=identity,
                repo_id=candidate.repo_id,
                revision="main",
                format=ArtifactVariantFormat.SAFETENSORS,
                filename=candidate.repo_id,
                size_bytes=analysis.total_size_bytes,
                compatible_runtimes=[RuntimeName.TRANSFORMERS],
                identity_match=match,
                confidence=_match_confidence(match),
                evidence=list(identity.evidence),
            )
        )
    return variants


def _identity_match(
    target: ModelIdentity,
    candidate: ModelCandidate,
    candidate_identity: ModelIdentity,
) -> IdentityMatchStatus:
    if (
        target.canonical_repo_id
        and candidate.base_model_repo_id
        and logical_model_repo_key(candidate.base_model_repo_id)
        == logical_model_repo_key(target.canonical_repo_id)
    ):
        return IdentityMatchStatus.CONFIRMED
    if (
        target.canonical_repo_id
        and candidate_identity.canonical_repo_id
        and model_identity_key(candidate_identity) == model_identity_key(target)
    ):
        return IdentityMatchStatus.CONFIRMED
    if target.family and candidate_identity.family and target.family != candidate_identity.family:
        return IdentityMatchStatus.REJECTED
    if _normalised_model_name(target.model_name) == _normalised_model_name(
        candidate_identity.model_name
    ):
        return IdentityMatchStatus.UNCERTAIN
    return IdentityMatchStatus.REJECTED


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


def _inspect_candidate(
    *,
    inspect_model: object,
    candidate: ModelCandidate,
    analysis_cache: ModelAnalysisCacheProtocol | None,
    telemetry: PerformanceTelemetry | None,
) -> ModelAnalysis:
    if analysis_cache is not None:
        cached = analysis_cache.get(candidate)
        if cached is not None:
            if telemetry is not None:
                telemetry.increment("variant_persistent_cache_hits")
            return cached
        if telemetry is not None:
            telemetry.increment("variant_persistent_cache_misses")
    analysis = inspect_model(candidate.repo_id)  # type: ignore[operator]
    if analysis_cache is not None:
        analysis_cache.put(candidate, analysis)
    return analysis  # type: ignore[no-any-return]


def _prioritize_variant_candidates(
    identity: ModelIdentity,
    candidates: list[ModelCandidate],
    *,
    include_uncertain: bool,
) -> list[ModelCandidate]:
    indexed = list(enumerate(candidates))
    ranked: list[tuple[int, int, ModelCandidate]] = []
    for index, candidate in indexed:
        priority = _variant_candidate_priority(identity, candidate)
        if priority is None:
            continue
        if priority > 1 and not include_uncertain:
            continue
        ranked.append((priority, index, candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in ranked]


def _variant_candidate_priority(
    identity: ModelIdentity,
    candidate: ModelCandidate,
) -> int | None:
    target = identity.canonical_repo_id
    if target and candidate.base_model_repo_id:
        if logical_model_repo_key(candidate.base_model_repo_id) == logical_model_repo_key(
            target
        ):
            return 0
        return None
    if target and logical_model_repo_key(candidate.repo_id) == logical_model_repo_key(
        target
    ):
        return 0
    cheap_identity = resolve_model_identity(candidate=candidate, analysis=None)
    if (
        target
        and cheap_identity.canonical_repo_id
        and model_identity_key(cheap_identity) == model_identity_key(identity)
    ):
        return 1
    if _normalised_model_name(identity.model_name) == _normalised_model_name(
        cheap_identity.model_name
    ):
        return 2
    if "gguf" in {tag.casefold() for tag in candidate.tags}:
        return 3
    return None


def _confirmed_gguf_quantization_count(variants: list[ArtifactVariant]) -> int:
    return sum(
        1
        for variant in variants
        if variant.format is ArtifactVariantFormat.GGUF
        and variant.identity_match is IdentityMatchStatus.CONFIRMED
        and variant.quantization
    )


def _gguf_variant(analysis: ModelAnalysis | None, quantization: str) -> GgufVariant | None:
    if analysis is None:
        return None
    for variant in analysis.classification.gguf_variants:
        if variant.quantization == quantization:
            return variant
    return None


def _repo_name(repo_id: str) -> str:
    return repo_id.split("/")[-1] if repo_id else "unknown"


def _variant_from_name(repo_id: str) -> str | None:
    name = _repo_name(repo_id).lower()
    for token in ("instruct", "chat", "base"):
        if token in name:
            return token
    return None


def _normalised_model_name(value: str) -> str:
    lowered = value.casefold()
    lowered = re.sub(r"\b(gguf|q\d(?:_k_[msl])?|int\d|fp16|bf16)\b", "", lowered)
    lowered = re.sub(r"[^a-z0-9.]+", "-", lowered)
    return lowered.strip("-")


def _match_confidence(match: IdentityMatchStatus) -> EstimationConfidence:
    if match is IdentityMatchStatus.CONFIRMED:
        return EstimationConfidence.HIGH
    if match is IdentityMatchStatus.UNCERTAIN:
        return EstimationConfidence.LOW
    return EstimationConfidence.UNKNOWN


def _dedupe_variants(variants: list[ArtifactVariant]) -> list[ArtifactVariant]:
    deduped: dict[tuple[str, str | None, str, str | None], ArtifactVariant] = {}
    for variant in variants:
        key = (
            variant.repo_id,
            variant.filename,
            variant.format.value,
            variant.quantization or variant.precision,
        )
        deduped.setdefault(key, variant)
    return list(deduped.values())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "unknown"


__all__ = [
    "build_execution_plan",
    "discover_artifact_variants",
    "execution_plan_for_recommendation",
    "resolve_model_identity",
    "variant_from_recommendation",
]
