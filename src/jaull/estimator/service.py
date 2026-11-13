"""End-to-end memory estimation service."""

from __future__ import annotations

import logging
import math

from jaull.domain.artifact_profile import is_packed_transformers_quantization_config
from jaull.domain.enrichment import EnrichmentResult
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    MemoryComponent,
    MemoryEstimate,
    WeightEstimate,
)
from jaull.domain.hardware import HardwareProfile, available_accelerator_memory_bytes
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import GgufVariant, ModelAnalysis, ModelConfig
from jaull.domain.runtime import RuntimeRecommendation  # noqa: F401 (rebuild)
from jaull.estimator import compatibility, kv_cache, overhead, policies, weights
from jaull.estimator.gguf_selection import VariantChoice, select_variant
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.huggingface.client import HfClientProtocol
from jaull.metadata import service as metadata_service
from jaull.metadata.range_reader import HeaderBudgetMemo, HttpRangeClient
from jaull.ports.cache import GgufHeaderCacheProtocol
from jaull.runtime import service as runtime_service

logger = logging.getLogger(__name__)


def estimate_memory(
    analysis: ModelAnalysis,
    hardware: HardwareProfile,
    inference_cfg: InferenceConfiguration,
    client: HfClientProtocol,
    *,
    resolve_base_model: bool = True,
    range_client: HttpRangeClient | None = None,
    gguf_header_cache: GgufHeaderCacheProtocol | None = None,
    header_budget: HeaderBudgetMemo | None = None,
    recommend_runtime: bool = True,
) -> MemoryEstimate:
    assumptions: list[str] = []
    warnings: list[str] = list(analysis.warnings)

    weight_estimate, variant_choice = _estimate_weights(
        analysis=analysis,
        inference_cfg=inference_cfg,
        client=client,
        warnings=warnings,
    )
    if variant_choice is not None:
        assumptions.append(variant_choice.reason)

    enrichment = _run_enrichment(
        analysis=analysis,
        variant_choice=variant_choice,
        client=client,
        resolve_base_model=resolve_base_model,
        range_client=range_client,
        gguf_header_cache=gguf_header_cache,
        header_budget=header_budget,
    )
    warnings.extend(enrichment.warnings)
    if enrichment.base_model_resolution.repo_id:
        assumptions.append(
            f"Base model resolved to {enrichment.base_model_resolution.repo_id} "
            f"via {enrichment.base_model_resolution.source.value}."
        )

    effective_config = _pick_effective_config(analysis.config, enrichment)
    weight_estimate = weights.add_transformer_block_decomposition(
        weight_estimate,
        effective_config,
    )

    kv_estimate = kv_cache.estimate_kv_cache(
        config=effective_config,
        context_length=inference_cfg.context_length,
        batch_size=inference_cfg.batch_size,
        kv_dtype=inference_cfg.kv_cache_dtype,
        concurrent_users=inference_cfg.concurrent_users,
    )
    warnings.extend(kv_estimate.notes)
    assumptions.append(
        f"KV cache dtype assumed to be {inference_cfg.kv_cache_dtype.value}."
    )

    is_gguf = analysis.classification.primary_type is RepositoryType.GGUF
    overhead_estimate = overhead.estimate_overhead(
        weight_estimate.component.bytes,
        base_bytes=(
            policies.GGUF_OVERHEAD_BASE_BYTES if is_gguf else policies.OVERHEAD_BASE_BYTES
        ),
        weight_fraction=(
            overhead.GGUF_OVERHEAD_WEIGHT_FRACTION
            if is_gguf
            else overhead.OVERHEAD_WEIGHT_FRACTION
        ),
    )
    cpu_overhead = (
        overhead.estimate_gguf_cpu_overhead(weight_estimate.component.bytes)
        if is_gguf else overhead_estimate
    )

    gpu_reserve_bytes = inference_cfg.device_reserve_bytes
    ram_reserve_bytes = (
        policies.RAM_RESERVE_DEFAULT_BYTES
        if inference_cfg.ram_reserve_bytes is None else inference_cfg.ram_reserve_bytes
    )
    margin_percent = inference_cfg.safety_margin_percent
    if margin_percent is None:
        margin_percent = policies.SAFETY_MARGIN_DEFAULT_PERCENT
    if is_gguf:
        gpu_reserve_bytes = max(gpu_reserve_bytes, policies.GGUF_DEVICE_HEADROOM_BYTES)
        if inference_cfg.safety_margin_percent is None:
            margin_percent = 0.0
    cpu_only = inference_cfg.target_device is TargetDevice.CPU or (
        inference_cfg.target_device is TargetDevice.AUTO
        and available_accelerator_memory_bytes(hardware) is None
    )
    if cpu_only:
        overhead_estimate = cpu_overhead
    device_reserve = _device_reserve_component(
        ram_reserve_bytes if cpu_only else gpu_reserve_bytes, cpu=cpu_only
    )
    cpu_subtotal = _sum_components([
        weight_estimate.component.bytes, kv_estimate.component.bytes,
        cpu_overhead.component.bytes, ram_reserve_bytes,
    ])
    cpu_safety_margin = _safety_margin_component(cpu_subtotal, margin_percent)

    subtotal = _sum_components(
        [
            weight_estimate.component.bytes,
            kv_estimate.component.bytes,
            overhead_estimate.component.bytes,
            device_reserve.bytes,
        ]
    )

    safety_margin = _safety_margin_component(
        subtotal=subtotal,
        percent=margin_percent,
    )

    total_bytes = _sum_components([subtotal, safety_margin.bytes if safety_margin else 0])

    assessment, fit = compatibility.assess_components_with_fit(
        weights_bytes=weight_estimate.component.bytes,
        kv_cache_bytes=kv_estimate.component.bytes,
        overhead_bytes=overhead_estimate.component.bytes,
        device_reserve_bytes=device_reserve.bytes or 0,
        safety_margin_bytes=(
            safety_margin.bytes if safety_margin is not None and safety_margin.bytes else 0
        ),
        total_bytes=total_bytes,
        total_transformer_blocks=kv_estimate.layers,
        weight_decomposition=weight_estimate.transformer_block_decomposition,
        kv_layer_bytes=kv_estimate.layer_bytes,
        hardware=hardware,
        device_target=inference_cfg.target_device,
        cpu_reserve_bytes=ram_reserve_bytes,
        cpu_safety_margin_bytes=cpu_safety_margin.bytes if cpu_safety_margin else 0,
        cpu_overhead_bytes=cpu_overhead.component.bytes,
    )

    # AUTO may fall back from a dedicated GPU to CPU. Report the reserve and
    # margin of that selected placement, not the rejected GPU's headroom.
    uses_cpu = cpu_only or assessment.effective_device is TargetDevice.CPU
    if uses_cpu:
        overhead_estimate = cpu_overhead
        device_reserve = _device_reserve_component(ram_reserve_bytes, cpu=True)
        safety_margin = cpu_safety_margin
        total_bytes = _sum_components([
            cpu_subtotal, safety_margin.bytes if safety_margin else 0,
        ])
        if is_gguf:
            warnings.append(
                f"CPU overhead uses a general {cpu_overhead.base_bytes / policies.MiB:g} MiB "
                f"+ {cpu_overhead.weight_fraction:.0%} of weights heuristic, "
                "not the GPU compute-buffer calibration. CPU repacking is not separately "
                "sized and can exceed this allowance."
            )
    if is_gguf:
        assumptions.append(
            f"GGUF CPU headroom: {ram_reserve_bytes / policies.MiB:g} MiB of RAM; "
            "the GPU's 1 GiB minimum does not apply."
            if uses_cpu else
            "GGUF GPU headroom: at least 1 GiB of VRAM."
        )
        assumptions.append(
            "No automatic percentage margin is added to GGUF headroom."
            if inference_cfg.safety_margin_percent is None else
            "The explicitly requested percentage margin is added to the selected reserve."
        )

    global_confidence = _combined_confidence(
        [
            weight_estimate.component.confidence,
            kv_estimate.component.confidence,
            overhead_estimate.component.confidence,
            assessment.confidence,
        ]
    )
    assessment = assessment.model_copy(update={"confidence": global_confidence})

    architecture = _resolve_architecture(analysis, effective_config)

    preliminary = MemoryEstimate(
        repository=analysis.repo,
        repository_type=analysis.classification.primary_type,
        inference_configuration=inference_cfg,
        weights=weight_estimate,
        kv_cache=kv_estimate,
        runtime_overhead=overhead_estimate,
        device_reserve=device_reserve,
        safety_margin=safety_margin,
        total_bytes=total_bytes,
        assessment=assessment,
        hardware_fit=fit,
        assumptions=assumptions,
        warnings=warnings,
        base_model_resolution=enrichment.base_model_resolution,
        configuration_sources=(
            dict(enrichment.enriched_config.sources)
            if enrichment.enriched_config is not None
            else {}
        ),
        architecture=architecture,
    )

    if not recommend_runtime:
        return preliminary

    runtime_rec = runtime_service.recommend(
        estimate=preliminary,
        hardware=hardware,
        repository_type=analysis.classification.primary_type,
    )
    return preliminary.model_copy(update={"runtime_recommendation": runtime_rec})


def _resolve_architecture(
    analysis: ModelAnalysis, effective_config: ModelConfig | None
) -> str | None:
    if effective_config and effective_config.model_type:
        return effective_config.model_type
    if analysis.config and analysis.config.model_type:
        return analysis.config.model_type
    return None


def _run_enrichment(
    analysis: ModelAnalysis,
    variant_choice: VariantChoice | None,
    client: HfClientProtocol,
    resolve_base_model: bool,
    range_client: HttpRangeClient | None,
    gguf_header_cache: GgufHeaderCacheProtocol | None,
    header_budget: HeaderBudgetMemo | None = None,
) -> EnrichmentResult:
    if analysis.classification.primary_type is not RepositoryType.GGUF:
        return metadata_service.unresolved_result()
    if not resolve_base_model:
        return metadata_service.unresolved_result(
            warnings=["Base-model resolution disabled by --no-resolve-base-model."]
        )
    variant: GgufVariant | None = variant_choice.variant if variant_choice else None
    return metadata_service.enrich(
        analysis=analysis,
        variant=variant,
        client=client,
        range_client=range_client,
        header_cache=gguf_header_cache,
        header_budget=header_budget,
    )


def _pick_effective_config(
    original: ModelConfig | None, enrichment: EnrichmentResult
) -> ModelConfig | None:
    if enrichment.enriched_config is None:
        return original
    return enrichment.enriched_config.config


def _estimate_weights(
    analysis: ModelAnalysis,
    inference_cfg: InferenceConfiguration,
    client: HfClientProtocol,
    warnings: list[str],
) -> tuple[WeightEstimate, VariantChoice | None]:
    classification = analysis.classification

    if classification.primary_type is RepositoryType.GGUF:
        choice = select_variant(
            variants=classification.gguf_variants,
            requested=inference_cfg.quantization,
        )
        return weights.estimate_weights_from_gguf(choice.variant), choice

    if inference_cfg.quantization is not None:
        warnings.append(
            f"--quantization {inference_cfg.quantization!r} ignored: repository is "
            f"{classification.primary_type.value}, not GGUF."
        )

    if classification.primary_type is RepositoryType.TRANSFORMERS:
        summary = None
        if not is_packed_transformers_quantization_config(analysis.config):
            try:
                summary = client.safetensors_summary(analysis.repo.repo_id)
            except HuggingFaceUnavailableError as exc:
                warnings.append(f"Could not read safetensors metadata: {exc}")
        else:
            warnings.append(
                "Packed quantized safetensors metadata ignored for weight sizing; "
                "using artifact file sizes instead."
            )
        if summary is not None:
            return (
                weights.estimate_weights_from_safetensors(
                    summary=summary,
                    requested_precision=inference_cfg.precision,
                    config=analysis.config,
                ),
                None,
            )
        return (
            weights.estimate_weights_from_files(
                analysis=analysis,
                requested_precision=(
                    None
                    if is_packed_transformers_quantization_config(analysis.config)
                    else inference_cfg.precision
                ),
                config=analysis.config,
            ),
            None,
        )

    return (
        weights.estimate_weights_from_files(
            analysis=analysis,
            requested_precision=inference_cfg.precision,
            config=analysis.config,
        ),
        None,
    )


def _device_reserve_component(reserve_bytes: int, *, cpu: bool = False) -> MemoryComponent:
    name = "RAM reserve" if cpu else "Device reserve"
    pool = "RAM" if cpu else "VRAM"
    option = "--ram-reserve-gib" if cpu else "--device-reserve-gib"
    if reserve_bytes <= 0:
        return MemoryComponent(
            name=name,
            bytes=0,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation=f"No {pool} reserve configured ({option} 0).",
        )
    return MemoryComponent(
        name=name,
        bytes=reserve_bytes,
        source=EstimateSource.ASSUMED,
        confidence=EstimationConfidence.LOW,
        explanation=f"Configured {pool} reserve of {reserve_bytes / (1024**3):.2f} GiB.",
    )


def _safety_margin_component(
    subtotal: int | None, percent: float
) -> MemoryComponent | None:
    if percent <= 0 or subtotal is None:
        return None
    margin = math.ceil(subtotal * (percent / 100.0))
    return MemoryComponent(
        name="Safety margin",
        bytes=margin,
        source=EstimateSource.ASSUMED,
        confidence=EstimationConfidence.MEDIUM,
        explanation=f"{percent:g}% of estimated subtotal.",
    )


def _sum_components(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    if not known:
        return None
    return sum(known)


_CONFIDENCE_ORDER = {
    EstimationConfidence.HIGH: 3,
    EstimationConfidence.MEDIUM: 2,
    EstimationConfidence.LOW: 1,
    EstimationConfidence.UNKNOWN: 0,
}


def _combined_confidence(
    values: list[EstimationConfidence],
) -> EstimationConfidence:
    if not values:
        return EstimationConfidence.UNKNOWN
    return min(values, key=lambda c: _CONFIDENCE_ORDER[c])


__all__ = ["estimate_memory"]
