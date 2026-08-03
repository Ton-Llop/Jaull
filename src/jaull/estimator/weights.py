"""Estimate the memory footprint of a model's weights."""

from __future__ import annotations

import math
from pathlib import PurePosixPath

from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    MemoryComponent,
    WeightEstimate,
)
from jaull.domain.inference import WeightPrecision
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelConfig,
    SafetensorsSummary,
)
from jaull.estimator.policies import BYTES_PER_PARAM, bytes_per_parameter

# Map safetensors dtype strings (F16, BF16, I8, ...) to our precision enum where
# it makes sense. Anything not in this map defaults to fp16 sizing.
_SAFETENSORS_DTYPE_TO_PRECISION: dict[str, WeightPrecision] = {
    "F32": WeightPrecision.FLOAT32,
    "F16": WeightPrecision.FLOAT16,
    "BF16": WeightPrecision.BFLOAT16,
    "I8": WeightPrecision.INT8,
    "U8": WeightPrecision.INT8,
}


def _torch_dtype_to_precision(name: str | None) -> WeightPrecision | None:
    if name is None:
        return None
    normalized = name.lower()
    mapping = {
        "float32": WeightPrecision.FLOAT32,
        "fp32": WeightPrecision.FLOAT32,
        "float16": WeightPrecision.FLOAT16,
        "fp16": WeightPrecision.FLOAT16,
        "half": WeightPrecision.FLOAT16,
        "bfloat16": WeightPrecision.BFLOAT16,
        "bf16": WeightPrecision.BFLOAT16,
        "int8": WeightPrecision.INT8,
        "int4": WeightPrecision.INT4,
    }
    return mapping.get(normalized)


def estimate_weights_from_gguf(variant: GgufVariant) -> WeightEstimate:
    return WeightEstimate(
        component=MemoryComponent(
            name="Weights",
            bytes=variant.total_bytes,
            source=EstimateSource.EXACT,
            confidence=EstimationConfidence.HIGH,
            explanation=(
                f"Exact remote file size of the {variant.quantization} variant."
            ),
        ),
        num_parameters=None,
        bits_per_parameter=None,
        precision=None,
        gguf_variant=variant.quantization,
    )


def estimate_weights_from_safetensors(
    summary: SafetensorsSummary,
    requested_precision: WeightPrecision | None,
    config: ModelConfig | None,
) -> WeightEstimate:
    precision = requested_precision or _dominant_precision(summary) or (
        _torch_dtype_to_precision(config.torch_dtype) if config else None
    ) or WeightPrecision.FLOAT16

    bpp = bytes_per_parameter(precision)
    weight_bytes = math.ceil(summary.total_parameters * bpp)
    explanation = (
        f"{summary.total_parameters:,} parameters x {bpp:g} bytes "
        f"({precision.value}) from safetensors metadata."
    )
    if requested_precision and requested_precision != _dominant_precision(summary):
        explanation += " Requested precision differs from the stored precision."

    return WeightEstimate(
        component=MemoryComponent(
            name="Weights",
            bytes=weight_bytes,
            source=EstimateSource.METADATA,
            confidence=EstimationConfidence.HIGH,
            explanation=explanation,
        ),
        num_parameters=summary.total_parameters,
        bits_per_parameter=bpp * 8,
        precision=precision,
    )


def estimate_weights_from_files(
    analysis: ModelAnalysis,
    requested_precision: WeightPrecision | None,
    config: ModelConfig | None,
) -> WeightEstimate:
    weight_files_total = _weight_file_bytes(analysis)
    if weight_files_total == 0:
        return _unknown_weights("No weight files (safetensors or bin) found in the repository.")

    stored_precision = (
        _torch_dtype_to_precision(config.torch_dtype) if config else None
    ) or WeightPrecision.FLOAT16
    effective_precision = requested_precision or stored_precision

    if effective_precision == stored_precision:
        bytes_estimate = weight_files_total
        explanation = (
            f"Sum of weight file sizes ({_format(weight_files_total)}) using stored "
            f"precision {stored_precision.value}."
        )
    else:
        ratio = bytes_per_parameter(effective_precision) / bytes_per_parameter(stored_precision)
        bytes_estimate = math.ceil(weight_files_total * ratio)
        explanation = (
            f"Weight files total {_format(weight_files_total)} at {stored_precision.value}; "
            f"scaled to {effective_precision.value} (ratio {ratio:.3f})."
        )

    return WeightEstimate(
        component=MemoryComponent(
            name="Weights",
            bytes=bytes_estimate,
            source=EstimateSource.DERIVED,
            confidence=EstimationConfidence.MEDIUM,
            explanation=explanation,
        ),
        num_parameters=None,
        bits_per_parameter=bytes_per_parameter(effective_precision) * 8,
        precision=effective_precision,
    )


def _unknown_weights(reason: str) -> WeightEstimate:
    return WeightEstimate(
        component=MemoryComponent(
            name="Weights",
            bytes=None,
            source=EstimateSource.UNKNOWN,
            confidence=EstimationConfidence.UNKNOWN,
            explanation=reason,
        ),
    )


def _weight_file_bytes(analysis: ModelAnalysis) -> int:
    total = 0
    for f in analysis.files:
        name = PurePosixPath(f.path).name.lower()
        if not f.size_bytes:
            continue
        is_safetensors = name.endswith(".safetensors")
        is_pytorch_bin = name.endswith(".bin") and "pytorch_model" in name
        if is_safetensors or is_pytorch_bin:
            total += f.size_bytes
    return total


def _dominant_precision(summary: SafetensorsSummary) -> WeightPrecision | None:
    if not summary.parameters_by_dtype:
        return None
    dominant = max(summary.parameters_by_dtype.items(), key=lambda kv: kv[1])[0]
    return _SAFETENSORS_DTYPE_TO_PRECISION.get(dominant)


def _format(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


__all__ = [
    "BYTES_PER_PARAM",
    "estimate_weights_from_files",
    "estimate_weights_from_gguf",
    "estimate_weights_from_safetensors",
]
