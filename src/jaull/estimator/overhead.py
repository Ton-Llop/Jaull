"""Runtime overhead heuristic (allocator, buffers, kernels)."""

from __future__ import annotations

from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    MemoryComponent,
    RuntimeOverheadEstimate,
)
from jaull.estimator.policies import (
    CPU_COMPUTE_ALLOWANCE_BYTES,
    GGUF_OVERHEAD_BASE_BYTES,
    GGUF_OVERHEAD_WEIGHT_FRACTION,
    OVERHEAD_BASE_BYTES,
    OVERHEAD_MIN_BYTES,
    OVERHEAD_WEIGHT_FRACTION,
)


def estimate_overhead(
    weights_bytes: int | None,
    *,
    base_bytes: int = OVERHEAD_BASE_BYTES,
    weight_fraction: float = OVERHEAD_WEIGHT_FRACTION,
) -> RuntimeOverheadEstimate:
    if weights_bytes is None:
        bytes_estimate = max(OVERHEAD_MIN_BYTES, base_bytes)
        explanation = (
            f"Weight size unknown; using base overhead only ({_gib(base_bytes)} GiB)."
        )
    else:
        raw = base_bytes + int(weight_fraction * weights_bytes)
        bytes_estimate = max(OVERHEAD_MIN_BYTES, raw)
        if weight_fraction > 0:
            explanation = (
                f"Base {_gib(base_bytes)} GiB + "
                f"{weight_fraction * 100:.0f}% of weights "
                f"({_gib(int(weight_fraction * weights_bytes))} GiB)."
            )
        else:
            explanation = (
                f"Base {_gib(base_bytes)} GiB, not scaled with the weights "
                "(llama.cpp compute buffers measured at 0.1-0.2 GiB)."
            )

    return RuntimeOverheadEstimate(
        component=MemoryComponent(
            name="Runtime overhead",
            bytes=bytes_estimate,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation=explanation,
        ),
        base_bytes=base_bytes,
        weight_fraction=weight_fraction,
        minimum_bytes=OVERHEAD_MIN_BYTES,
    )


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f}"


def estimate_gguf_cpu_overhead(weights_bytes: int | None) -> RuntimeOverheadEstimate:
    """CPU fallback heuristic, not the GPU compute-buffer calibration.

    Reuse the general runtime allowance (512 MiB + 10% of weights) until
    tensor/kernel-specific repacking is known. A single observed CPU_REPACK
    allocation must not become a universal constant or an RSS correction.
    """
    result = estimate_overhead(weights_bytes)
    compute = CPU_COMPUTE_ALLOWANCE_BYTES
    assert result.component.bytes is not None
    return result.model_copy(update={
        "component": result.component.model_copy(update={
            "explanation": result.component.explanation + (
                " CPU allowance, not GPU calibration. CPU repacking is runtime-dependent "
                "and is not separately sized; this heuristic is not an upper bound."
            ),
        }),
        "components": [
            MemoryComponent(
                name="CPU compute allowance", bytes=compute,
                source=EstimateSource.ASSUMED, confidence=EstimationConfidence.LOW,
                explanation=f"{_gib(compute)} GiB heuristic; actual compute buffers "
                "depend on token batching.",
            ),
            MemoryComponent(
                name="Other CPU runtime allowance", bytes=result.component.bytes - compute,
                source=EstimateSource.ASSUMED, confidence=EstimationConfidence.LOW,
                explanation="Remaining general runtime heuristic; not a measured allocation.",
            ),
            MemoryComponent(
                name="CPU repacking", bytes=None,
                source=EstimateSource.UNKNOWN, confidence=EstimationConfidence.UNKNOWN,
                explanation="Depends on tensor types, CPU kernels and runtime build. "
                "Not separately added to the heuristic total.",
            ),
        ],
        "unmodelled_components": ["CPU repacking"],
    })


__all__ = [
    "GGUF_OVERHEAD_BASE_BYTES",
    "GGUF_OVERHEAD_WEIGHT_FRACTION",
    "OVERHEAD_WEIGHT_FRACTION",
    "estimate_overhead",
]
