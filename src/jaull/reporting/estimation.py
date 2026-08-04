"""JSON emitter for :class:`MemoryEstimate`.

The Rich rendering side of the estimation view stays in
``jaull.presentation.estimation_report`` — this module owns only the JSON
projection so ``recommendation.report`` (now ``reporting.recommendation``)
can consume it without depending on ``presentation``.
"""

from __future__ import annotations

from typing import Any

from jaull.domain.estimation import MemoryComponent, MemoryEstimate
from jaull.reporting.serialization import (
    base_resolution_to_dict,
    component_to_dict,
    confidence_to_string,
    runtime_to_dict,
)

SCHEMA_VERSION = 1


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


__all__ = ["SCHEMA_VERSION", "estimate_to_json_dict"]
