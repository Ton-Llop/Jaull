"""Compare a memory prediction with one real execution observation."""

from __future__ import annotations

from jaull.domain.comparison import (
    CompatibilityComparison,
    CompatibilityOutcome,
    MetricComparison,
    MetricComparisonAvailability,
    PredictionComparison,
)
from jaull.domain.estimation import CompatibilityStatus, MemoryEstimate
from jaull.domain.execution import ExecutionObservation
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation

_RAM_GPU_UNAVAILABLE_REASON = (
    "MemoryEstimate does not preserve a host/device breakdown for the selected "
    "GPU-offloaded runtime, so process RSS is not comparable."
)
_VRAM_UNAVAILABLE_REASON = (
    "MemoryEstimate does not preserve process-attributed VRAM for the selected "
    "runtime configuration."
)


class PredictionRuntimeMismatchError(ValueError):
    """The runtime being validated does not match the prediction configuration."""


def compare_prediction(
    *,
    estimate: MemoryEstimate,
    observation: ExecutionObservation,
    runtime: RuntimeRecommendation | None = None,
) -> PredictionComparison:
    """Compare Jaull's prediction with a single measured execution.

    RAM is compared only when the selected runtime is CPU-only/no-offload. For
    GPU offload Jaull currently lacks a host/device memory split, so returning a
    number would compare different quantities.
    """

    ram_predicted, ram_availability, ram_reason = _predicted_ram(
        estimate=estimate,
        runtime=runtime,
    )
    ram = _metric_comparison(
        predicted_bytes=ram_predicted,
        measured_bytes=observation.peak_ram_bytes,
        unavailable_availability=ram_availability,
        unavailable_reason=ram_reason,
    )

    vram = _metric_comparison(
        predicted_bytes=None,
        measured_bytes=observation.peak_vram_bytes,
        unavailable_availability=(
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
        ),
        unavailable_reason=_VRAM_UNAVAILABLE_REASON,
    )

    predicted_runnable = _predicted_runnable(
        estimate.assessment.status,
        runtime=runtime,
    )
    compatibility = CompatibilityComparison(
        predicted_status=estimate.assessment.status,
        predicted_runnable=predicted_runnable,
        observed_success=observation.success,
        outcome=_compatibility_outcome(
            predicted_runnable=predicted_runnable,
            observed_success=observation.success,
        ),
        failure_reason=observation.failure_reason,
    )

    return PredictionComparison(
        ram=ram,
        vram=vram,
        compatibility=compatibility,
    )


def assert_prediction_runtime_matches(
    *,
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation,
) -> None:
    """Validate the lightweight provenance needed for one comparison.

    ``ExecutionObservation`` intentionally records what happened at the process
    boundary, not the requested model configuration. The caller must therefore
    compare with the same ``RuntimeRecommendation`` it passed to the runner. This
    guard catches the mismatches Jaull can detect today without introducing a
    larger experiment-record model.
    """

    if runtime.runtime is not RuntimeName.LLAMA_CPP:
        raise PredictionRuntimeMismatchError(
            f"Expected {RuntimeName.LLAMA_CPP.value!r} runtime, got "
            f"{runtime.runtime.value!r}."
        )

    ctx_size = _int_runtime_flag(runtime, "--ctx-size")
    if (
        ctx_size is not None
        and ctx_size != estimate.inference_configuration.context_length
    ):
        raise PredictionRuntimeMismatchError(
            "Runtime --ctx-size does not match MemoryEstimate context length: "
            f"{ctx_size} != {estimate.inference_configuration.context_length}."
        )

    n_gpu_layers = _n_gpu_layers(runtime)
    if (
        estimate.inference_configuration.target_device is TargetDevice.CPU
        and n_gpu_layers not in {None, 0}
    ):
        raise PredictionRuntimeMismatchError(
            "CPU-only MemoryEstimate cannot be compared with a GPU-offloaded "
            f"runtime (--n-gpu-layers {n_gpu_layers})."
        )


def _metric_comparison(
    *,
    predicted_bytes: int | None,
    measured_bytes: int | None,
    unavailable_availability: MetricComparisonAvailability | None = None,
    unavailable_reason: str | None = None,
) -> MetricComparison:
    if unavailable_availability is not None:
        return MetricComparison(
            predicted_bytes=predicted_bytes,
            measured_bytes=measured_bytes,
            error_bytes=None,
            absolute_error_bytes=None,
            error_percent=None,
            availability=unavailable_availability,
            unavailable_reason=unavailable_reason,
        )

    if predicted_bytes is None:
        return MetricComparison(
            predicted_bytes=None,
            measured_bytes=measured_bytes,
            error_bytes=None,
            absolute_error_bytes=None,
            error_percent=None,
            availability=MetricComparisonAvailability.PREDICTION_UNAVAILABLE,
            unavailable_reason="Prediction is unavailable.",
        )
    if measured_bytes is None:
        return MetricComparison(
            predicted_bytes=predicted_bytes,
            measured_bytes=None,
            error_bytes=None,
            absolute_error_bytes=None,
            error_percent=None,
            availability=MetricComparisonAvailability.MEASUREMENT_UNAVAILABLE,
            unavailable_reason="Measurement is unavailable.",
        )

    error = measured_bytes - predicted_bytes
    percent = (error / predicted_bytes * 100.0) if predicted_bytes > 0 else None
    return MetricComparison(
        predicted_bytes=predicted_bytes,
        measured_bytes=measured_bytes,
        error_bytes=error,
        absolute_error_bytes=abs(error),
        error_percent=percent,
        availability=MetricComparisonAvailability.AVAILABLE,
        unavailable_reason=None,
    )


def _predicted_ram(
    *,
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation | None,
) -> tuple[int | None, MetricComparisonAvailability | None, str | None]:
    if _uses_gpu_memory(estimate, runtime):
        return (
            None,
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
            _RAM_GPU_UNAVAILABLE_REASON,
        )

    parts = [
        estimate.weights.component.bytes,
        estimate.kv_cache.component.bytes,
        estimate.runtime_overhead.component.bytes,
    ]
    if any(part is None for part in parts):
        return (None, None, None)
    return (sum(part for part in parts if part is not None), None, None)


def _uses_gpu_memory(
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation | None,
) -> bool:
    n_gpu_layers = _n_gpu_layers(runtime)
    if n_gpu_layers is not None:
        return n_gpu_layers != 0
    return estimate.assessment.effective_device is TargetDevice.GPU


def _n_gpu_layers(runtime: RuntimeRecommendation | None) -> int | None:
    if runtime is None or runtime.runtime is not RuntimeName.LLAMA_CPP:
        return None
    for flag in runtime.flags:
        if flag.name == "--n-gpu-layers":
            try:
                return int(flag.value)
            except ValueError:
                return None
    return None


def _int_runtime_flag(runtime: RuntimeRecommendation, name: str) -> int | None:
    for flag in runtime.flags:
        if flag.name == name:
            try:
                return int(flag.value)
            except ValueError:
                return None
    return None


def _predicted_runnable(
    status: CompatibilityStatus,
    *,
    runtime: RuntimeRecommendation | None,
) -> bool | None:
    if status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }:
        return True
    if status is CompatibilityStatus.INSUFFICIENT:
        return False
    if status is CompatibilityStatus.UNKNOWN:
        return None
    if status is CompatibilityStatus.OFFLOADING_REQUIRED:
        if _runtime_matches_required_offload(runtime):
            return True
        return None
    return None


def _runtime_matches_required_offload(
    runtime: RuntimeRecommendation | None,
) -> bool:
    if runtime is None or runtime.runtime is RuntimeName.UNKNOWN:
        return False
    n_gpu_layers = _n_gpu_layers(runtime)
    return n_gpu_layers is not None and n_gpu_layers != 0


def _compatibility_outcome(
    *,
    predicted_runnable: bool | None,
    observed_success: bool,
) -> CompatibilityOutcome:
    if predicted_runnable is None:
        return CompatibilityOutcome.UNKNOWN
    if predicted_runnable and observed_success:
        return CompatibilityOutcome.CORRECT_SUCCESS
    if predicted_runnable and not observed_success:
        return CompatibilityOutcome.FALSE_POSITIVE
    if not predicted_runnable and observed_success:
        return CompatibilityOutcome.FALSE_NEGATIVE
    return CompatibilityOutcome.CORRECT_FAILURE


__all__ = [
    "PredictionRuntimeMismatchError",
    "assert_prediction_runtime_matches",
    "compare_prediction",
]
