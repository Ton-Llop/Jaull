"""Discover artifact variants for an existing logical model."""

from __future__ import annotations

import re
from typing import Protocol

from jaull.application.recommendation import policies
from jaull.discovery.search_client import ModelSearchClient
from jaull.domain.candidates import ModelCandidate, SearchQuery
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    IdentityMatchStatus,
    ModelIdentity,
    logical_model_repo_key,
    model_identity_key,
)
from jaull.domain.model import ModelAnalysis
from jaull.domain.runtime import RuntimeName
from jaull.execution_plans.quantization import packed_transformers_quantization_label
from jaull.execution_plans.service import resolve_model_identity
from jaull.observability.telemetry import PerformanceTelemetry
from jaull.ports.cache import ModelAnalysisCacheProtocol


class InspectModelFn(Protocol):
    def __call__(self, repo_id: str) -> ModelAnalysis: ...


def discover_artifact_variants(
    *,
    identity: ModelIdentity,
    current: ArtifactVariant | None,
    search_client: ModelSearchClient,
    inspect_model: InspectModelFn,
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


def _inspect_candidate(
    *,
    inspect_model: InspectModelFn,
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
    analysis = inspect_model(candidate.repo_id)
    if analysis_cache is not None:
        analysis_cache.put(candidate, analysis)
    return analysis


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
        packed_quantization = packed_transformers_quantization_label(analysis.config)
        variants.append(
            ArtifactVariant(
                model_identity=identity,
                repo_id=candidate.repo_id,
                revision="main",
                format=ArtifactVariantFormat.SAFETENSORS,
                filename=candidate.repo_id,
                size_bytes=analysis.total_size_bytes,
                quantization=packed_quantization,
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


def _confirmed_gguf_quantization_count(variants: list[ArtifactVariant]) -> int:
    return sum(
        1
        for variant in variants
        if variant.format is ArtifactVariantFormat.GGUF
        and variant.identity_match is IdentityMatchStatus.CONFIRMED
        and variant.quantization
    )


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


__all__ = ["discover_artifact_variants"]
