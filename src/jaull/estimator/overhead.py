"""Runtime overhead heuristic (allocator, buffers, kernels)."""

from __future__ import annotations

from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    MemoryComponent,
    RuntimeOverheadEstimate,
)
from jaull.estimator.policies import (
    OVERHEAD_BASE_BYTES,
    OVERHEAD_MIN_BYTES,
    OVERHEAD_WEIGHT_FRACTION,
)


def estimate_overhead(weights_bytes: int | None) -> RuntimeOverheadEstimate:
    if weights_bytes is None:
        bytes_estimate = OVERHEAD_BASE_BYTES
        explanation = (
            f"Weight size unknown; using base overhead only "
            f"({_gib(OVERHEAD_BASE_BYTES)} GiB)."
        )
    else:
        raw = OVERHEAD_BASE_BYTES + int(OVERHEAD_WEIGHT_FRACTION * weights_bytes)
        bytes_estimate = max(OVERHEAD_MIN_BYTES, raw)
        explanation = (
            f"Base {_gib(OVERHEAD_BASE_BYTES)} GiB + "
            f"{OVERHEAD_WEIGHT_FRACTION * 100:.0f}% of weights "
            f"({_gib(int(OVERHEAD_WEIGHT_FRACTION * weights_bytes))} GiB)."
        )

    return RuntimeOverheadEstimate(
        component=MemoryComponent(
            name="Runtime overhead",
            bytes=bytes_estimate,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation=explanation,
        ),
        base_bytes=OVERHEAD_BASE_BYTES,
        weight_fraction=OVERHEAD_WEIGHT_FRACTION,
        minimum_bytes=OVERHEAD_MIN_BYTES,
    )


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f}"


__all__ = ["estimate_overhead"]
