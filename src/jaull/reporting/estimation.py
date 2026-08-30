"""JSON emitter for :class:`MemoryEstimate`.

The Rich rendering side of the estimation view stays in
``jaull.presentation.estimation_report`` — this module owns only the JSON
projection so ``recommendation.report`` (now ``reporting.recommendation``)
can consume it without depending on ``presentation``.
"""

from __future__ import annotations

from typing import Any

from jaull.domain.estimation import (
    HardwareFitOffloadCandidate,
    HardwareFitOffloadDiagnostics,
    HardwareFitResult,
    MemoryComponent,
    MemoryEstimate,
)
from jaull.reporting.serialization import (
    base_resolution_to_dict,
    component_to_dict,
    confidence_to_string,
    runtime_to_dict,
)

SCHEMA_VERSION = 2


def estimate_to_json_dict(estimate: MemoryEstimate) -> dict[str, Any]:
    cfg = estimate.inference_configuration
    assessment = estimate.assessment

    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "repo_id": estimate.repository.repo_id,
            "repository_type": estimate.repository_type.value,
            "author": estimate.repository.author,
            "pipeline_tag": estimate.repository.pipeline_tag,
            "library_name": estimate.repository.library_name,
            "license": estimate.repository.license,
            "gated": estimate.repository.gated,
        },
        "inference_configuration": {
            "context_length": cfg.context_length,
            "batch_size": cfg.batch_size,
            "target_device": cfg.target_device.value,
            "precision": cfg.precision.value if cfg.precision else None,
            "quantization": cfg.quantization,
            "kv_cache_dtype": cfg.kv_cache_dtype.value,
            "safety_margin_percent": cfg.safety_margin_percent,
            "device_reserve_bytes": cfg.device_reserve_bytes,
        },
        "memory": {
            "weights_bytes": estimate.weights.component.bytes,
            "kv_cache_bytes": estimate.kv_cache.component.bytes,
            "runtime_overhead_bytes": estimate.runtime_overhead.component.bytes,
            "device_reserve_bytes": estimate.device_reserve.bytes,
            "safety_margin_bytes": (
                estimate.safety_margin.bytes if estimate.safety_margin else 0
            ),
            "total_bytes": estimate.total_bytes,
            "components": [component_to_dict(c) for c in _components(estimate)],
        },
        "weights": {
            "num_parameters": estimate.weights.num_parameters,
            "bits_per_parameter": estimate.weights.bits_per_parameter,
            "precision": (
                estimate.weights.precision.value
                if estimate.weights.precision
                else None
            ),
            "gguf_variant": estimate.weights.gguf_variant,
        },
        "kv_cache": {
            "layers": estimate.kv_cache.layers,
            "kv_heads": estimate.kv_cache.kv_heads,
            "head_dim": estimate.kv_cache.head_dim,
            "context_length": estimate.kv_cache.context_length,
            "batch_size": estimate.kv_cache.batch_size,
            "dtype_bytes": estimate.kv_cache.dtype_bytes,
            "formula": estimate.kv_cache.formula,
            "notes": list(estimate.kv_cache.notes),
        },
        "hardware": {
            "available_vram_bytes": assessment.available_vram_bytes,
            "available_ram_bytes": assessment.available_ram_bytes,
        },
        "assessment": {
            "status": assessment.status.value,
            "confidence": confidence_to_string(assessment.confidence),
            "target_device": assessment.target_device.value,
            "effective_device": assessment.effective_device.value,
            "utilisation_ratio": assessment.ratio,
            "reasons": list(assessment.reasons),
            "warnings": list(assessment.warnings),
        },
        "hardware_fit": _hardware_fit_to_dict(estimate.hardware_fit),
        "assumptions": list(estimate.assumptions),
        "warnings": list(estimate.warnings),
        "base_model_resolution": base_resolution_to_dict(
            estimate.base_model_resolution
        ),
        "configuration_sources": {
            field: source.value
            for field, source in estimate.configuration_sources.items()
        },
        "architecture": estimate.architecture,
        "runtime_recommendation": runtime_to_dict(estimate.runtime_recommendation),
    }


def _hardware_fit_to_dict(fit: HardwareFitResult | None) -> dict[str, Any] | None:
    """Project the structured placement, keeping budget and allocation apart.

    ``gpu_required_bytes`` is the capacity budget the analyzer checked against
    VRAM; ``gpu_physical_bytes`` is the subset a process is expected to actually
    allocate, with the device reserve and the safety margin removed. Emitting
    both is what lets a consumer compare against a measurement without having to
    know which components are policy.
    """

    if fit is None:
        return None
    return {
        "mode": fit.mode.value,
        "memory_topology": fit.memory_topology.value,
        "placement_method": fit.placement_method.value,
        "gpu_transformer_blocks": fit.gpu_transformer_blocks,
        "total_transformer_blocks": fit.total_transformer_blocks,
        "offload_diagnostics": hardware_fit_offload_diagnostics_to_dict(
            fit.offload_diagnostics
        ),
        "gpu_required_bytes": fit.gpu_required_bytes,
        "gpu_physical_bytes": fit.gpu_physical_bytes,
        "gpu_weight_bytes": fit.gpu_weight_bytes,
        "gpu_overhead_bytes": fit.gpu_overhead_bytes,
        "gpu_safety_margin_bytes": fit.gpu_safety_margin_bytes,
        "ram_required_bytes": fit.ram_required_bytes,
        "ram_physical_bytes": fit.ram_physical_bytes,
        "ram_weight_bytes": fit.ram_weight_bytes,
        "ram_overhead_bytes": fit.ram_overhead_bytes,
        "ram_safety_margin_bytes": fit.ram_safety_margin_bytes,
        "device_reserve_bytes": fit.device_reserve_bytes,
        "available_vram_bytes": fit.available_vram_bytes,
        "available_ram_bytes": fit.available_ram_bytes,
        "reason": fit.reason,
        "warnings": list(fit.warnings),
    }


def hardware_fit_offload_diagnostics_to_dict(
    diagnostics: HardwareFitOffloadDiagnostics | None,
) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    return {
        "search_ceiling_transformer_blocks": (
            diagnostics.search_ceiling_transformer_blocks
        ),
        "selected": hardware_fit_offload_candidate_to_dict(diagnostics.selected),
        "first_rejected_higher": hardware_fit_offload_candidate_to_dict(
            diagnostics.first_rejected_higher
        ),
    }


def hardware_fit_offload_candidate_to_dict(
    candidate: HardwareFitOffloadCandidate | None,
) -> dict[str, int] | None:
    if candidate is None:
        return None
    return {
        "gpu_transformer_blocks": candidate.gpu_transformer_blocks,
        "gpu_required_bytes": candidate.gpu_required_bytes,
        "ram_required_bytes": candidate.ram_required_bytes,
        "available_vram_bytes": candidate.available_vram_bytes,
        "excess_bytes": candidate.excess_bytes,
        "headroom_bytes": candidate.headroom_bytes,
        "gpu_weight_bytes": candidate.gpu_weight_bytes,
        "ram_weight_bytes": candidate.ram_weight_bytes,
        "kv_cache_bytes": candidate.kv_cache_bytes,
        "device_reserve_bytes": candidate.device_reserve_bytes,
        "gpu_overhead_bytes": candidate.gpu_overhead_bytes,
        "gpu_safety_margin_bytes": candidate.gpu_safety_margin_bytes,
    }


def _components(estimate: MemoryEstimate) -> list[MemoryComponent]:
    parts = [
        estimate.weights.component,
        estimate.kv_cache.component,
        estimate.runtime_overhead.component,
        estimate.device_reserve,
    ]
    if estimate.safety_margin is not None:
        parts.append(estimate.safety_margin)
    return parts


__all__ = [
    "SCHEMA_VERSION",
    "estimate_to_json_dict",
    "hardware_fit_offload_candidate_to_dict",
    "hardware_fit_offload_diagnostics_to_dict",
]
