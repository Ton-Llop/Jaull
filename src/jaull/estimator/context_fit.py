"""How much memory a model needs at each context, and the largest context that still fits.

The sweep asks the estimator the same question at several contexts. It adds no arithmetic of its
own: the KV cache, the overhead and the verdict all come from whatever ``estimate_fn`` returns, so
there is a single authority on what fits (the same pattern as
:mod:`jaull.estimator.configuration`, which walks a quantization ladder).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis
from jaull.estimator import policies
from jaull.estimator.compatibility import RESIDENT_FIT_STATUSES

EstimateFn = Callable[[ModelAnalysis, InferenceConfiguration], MemoryEstimate]


@dataclass(frozen=True)
class ContextPoint:
    """One context and what the estimator said about it."""

    context_length: int
    total_bytes: int | None
    kv_cache_bytes: int | None
    status: CompatibilityStatus
    confidence: EstimationConfidence


@dataclass(frozen=True)
class ContextSweep:
    points: tuple[ContextPoint, ...]
    # The last rung of the unbroken run of fits that starts at the smallest context, or ``None``.
    largest_fitting: int | None
    native_limit: int | None = None
    notes: list[str] = field(default_factory=list)


def sweep_context(
    analysis: ModelAnalysis,
    config: InferenceConfiguration,
    estimate_fn: EstimateFn,
    *,
    ladder: Sequence[int] = policies.CONTEXT_LADDER,
    native_limit: int | None = None,
) -> ContextSweep:
    """Estimate ``config`` at every context of ``ladder`` and find where it stops fitting.

    ``native_limit`` is the context the model was trained for, when it is known: rungs above it are
    not estimated, because a prediction there describes a run nobody should make.
    """
    contexts = _rungs(ladder, native_limit)
    points = tuple(_point(analysis, config, estimate_fn, context) for context in contexts)
    largest = _largest_contiguous_fit(points)

    notes: list[str] = []
    if native_limit is not None:
        notes.append(f"Capped at the model's native context of {native_limit} tokens.")
    if largest is None:
        notes.append(
            f"Not even the smallest context in the sweep ({contexts[0]} tokens) fits resident."
        )
    elif native_limit is None and largest == contexts[-1]:
        notes.append(
            f"The sweep stops at {largest} tokens and it all fits, so the model may fit more."
        )
    return ContextSweep(
        points=points, largest_fitting=largest, native_limit=native_limit, notes=notes
    )


def _rungs(ladder: Sequence[int], native_limit: int | None) -> list[int]:
    rungs = sorted(set(ladder))
    if native_limit is None:
        return rungs
    capped = [rung for rung in rungs if rung <= native_limit]
    if native_limit not in capped:
        capped.append(native_limit)
    return capped


def _point(
    analysis: ModelAnalysis,
    config: InferenceConfiguration,
    estimate_fn: EstimateFn,
    context: int,
) -> ContextPoint:
    estimate = estimate_fn(analysis, config.model_copy(update={"context_length": context}))
    return ContextPoint(
        context_length=context,
        total_bytes=estimate.total_bytes,
        kv_cache_bytes=estimate.kv_cache.component.bytes,
        status=estimate.assessment.status,
        confidence=estimate.assessment.confidence,
    )


def _largest_contiguous_fit(points: Sequence[ContextPoint]) -> int | None:
    """Memory only grows with the context, so a fit above a failure is not believed."""
    largest: int | None = None
    for point in points:
        if point.status not in RESIDENT_FIT_STATUSES:
            break
        largest = point.context_length
    return largest


__all__ = ["ContextPoint", "ContextSweep", "EstimateFn", "sweep_context"]
