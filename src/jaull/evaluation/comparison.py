"""Compare a memory prediction with one real execution observation."""

from __future__ import annotations

from jaull.domain.comparison import (
    ComparisonSemantics,
    CompatibilityComparison,
    CompatibilityOutcome,
    ComponentComparison,
    MemoryObservationSource,
    MetricComparison,
    MetricComparisonAvailability,
    PredictionComparison,
)
from jaull.domain.estimation import (
    CompatibilityStatus,
    HardwareFitMode,
    HardwareFitResult,
    MemoryEstimate,
)
from jaull.domain.execution import (
    ExecutionObservation,
    RuntimeBufferCategory,
    RuntimeReportedAllocation,
)
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation

_RAM_GPU_UNAVAILABLE_REASON = (
    "Peak process RSS does not measure host memory demand under GPU offload: "
    "llama.cpp maps the whole model file, and the pages it reads to upload "
    "weights to the device stay resident, so RSS reflects the artifact size "
    "rather than the host share of the placement."
)
_VRAM_NO_FIT_REASON = (
    "MemoryEstimate carries no structured hardware fit, so there is no "
    "device-specific VRAM prediction to compare."
)
_VRAM_NO_GPU_PLACEMENT_REASON = (
    "The predicted placement puts no weights on the GPU, so Jaull predicts no "
    "process-attributed VRAM for this configuration."
)
_VRAM_UNVERIFIED_PLACEMENT_REASON = (
    "The executed placement cannot be verified against the prediction: without "
    "a llama.cpp --n-gpu-layers value there is no evidence the run split the "
    "model the way the estimate assumed, and a difference could not be "
    "attributed to the memory model rather than to a different placement."
)
_VRAM_RUNTIME_MAPPING_UNAVAILABLE_REASON = (
    "The Hardware Fit prediction is expressed in runtime-agnostic transformer "
    "blocks, while llama.cpp --n-gpu-layers uses backend-specific offload "
    "units. Jaull has no validated mapping between the two yet, so this run "
    "cannot be held to the transformer-block placement."
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

    RAM is compared only when the selected runtime is CPU-only/no-offload. Under
    offload the blocker is the measurement, not a missing host/device split --
    ``HardwareFitResult`` carries one; see :func:`_predicted_ram` for why peak
    RSS does not follow it.

    VRAM is compared when the estimate carries a hardware fit that places
    weights on the GPU *and* the run can be shown to have used that placement.
    Both sides are then process-attributed allocations; see
    :func:`_predicted_vram` for which components qualify and why.
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

    vram_predicted, vram_availability, vram_reason = _predicted_vram(
        estimate=estimate,
        runtime=runtime,
    )
    vram_measured, vram_source = _observed_vram(observation)
    vram = _metric_comparison(
        predicted_bytes=vram_predicted,
        measured_bytes=vram_measured,
        unavailable_availability=vram_availability,
        unavailable_reason=vram_reason,
        source=vram_source,
        allocation=observation.runtime_allocation,
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
        vram_components=_vram_components(
            estimate=estimate,
            observation=observation,
            runtime=runtime,
        ),
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

    if runtime.runtime not in {RuntimeName.LLAMA_CPP, RuntimeName.TRANSFORMERS}:
        raise PredictionRuntimeMismatchError(
            "Expected a comparable runtime "
            f"({RuntimeName.LLAMA_CPP.value!r} or {RuntimeName.TRANSFORMERS.value!r}), got "
            f"{runtime.runtime.value!r}."
        )
    if runtime.runtime is RuntimeName.TRANSFORMERS:
        return

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


_OVERHEAD_PROXY_REASON = (
    "Jaull's runtime overhead is a heuristic covering the allocator, activation "
    "buffers and the CUDA context. llama.cpp's compute buffer is only the graph "
    "allocation it enumerates; it does not report the context or anything the "
    "allocator took on its own. The difference between the two therefore does "
    "not measure how wrong the heuristic is, and must not be used to calibrate "
    "it until a driver-attributed figure bounds the unreported part."
)
_Component = tuple[str, str, RuntimeBufferCategory, ComparisonSemantics, str | None]
_COMPONENTS: tuple[_Component, ...] = (
    (
        "gpu_weight_bytes",
        "CUDA model buffer",
        RuntimeBufferCategory.MODEL,
        ComparisonSemantics.DIRECT,
        None,
    ),
    (
        "gpu_kv_cache_bytes",
        "CUDA KV buffer",
        RuntimeBufferCategory.KV,
        ComparisonSemantics.DIRECT,
        None,
    ),
    (
        "gpu_overhead_bytes",
        "CUDA compute buffer",
        RuntimeBufferCategory.COMPUTE,
        ComparisonSemantics.PROXY,
        _OVERHEAD_PROXY_REASON,
    ),
)


def _vram_components(
    *,
    estimate: MemoryEstimate,
    observation: ExecutionObservation,
    runtime: RuntimeRecommendation | None,
) -> tuple[ComponentComparison, ...]:
    """Break the VRAM comparison down by component, when a runtime reported one.

    A total says the prediction is off; only the breakdown says which term is
    responsible. Nothing here is derived — every observed figure is a buffer the
    runtime enumerated, and a component the runtime did not report stays
    unavailable rather than being inferred from the others.
    """

    allocation = observation.runtime_allocation
    fit = estimate.hardware_fit
    if allocation is None or fit is None or not fit.places_weights_on_gpu:
        return ()

    # The same gate the total uses: comparing a predicted placement against a
    # run that placed the model differently attributes the difference to the
    # memory model when it belongs to the placement.
    mismatch = _placement_mismatch(fit, runtime)

    components: list[ComponentComparison] = []
    for field, label, category, semantics, reason in _COMPONENTS:
        predicted = getattr(fit, field, None)
        observed = allocation.device_bytes_for(category)
        if mismatch is not None:
            components.append(
                ComponentComparison(
                    component=field,
                    observed_label=label,
                    semantics=ComparisonSemantics.UNAVAILABLE,
                    semantics_reason=mismatch,
                    metric=_metric_comparison(
                        predicted_bytes=predicted,
                        measured_bytes=observed,
                        unavailable_availability=(
                            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
                        ),
                        unavailable_reason=mismatch,
                        source=MemoryObservationSource.RUNTIME_REPORTED_ALLOCATION,
                        allocation=allocation,
                    ),
                )
            )
            continue
        components.append(
            ComponentComparison(
                component=field,
                observed_label=label,
                semantics=semantics,
                semantics_reason=reason,
                metric=_metric_comparison(
                    predicted_bytes=predicted,
                    measured_bytes=observed,
                    source=MemoryObservationSource.RUNTIME_REPORTED_ALLOCATION,
                    allocation=allocation,
                ),
            )
        )
    return tuple(components)


def _observed_vram(
    observation: ExecutionObservation,
) -> tuple[int | None, MemoryObservationSource | None]:
    """Pick the VRAM measurement, preferring the driver's own attribution.

    NVML says what the driver charged to the process. A runtime's report says
    what the runtime asked for and enumerated, which is strictly less — it
    cannot see the CUDA context or anything the allocator took on its own. The
    driver figure wins wherever it exists; on a GPU in WDDM mode it never does,
    and the runtime report is the only observation available.
    """

    if observation.peak_vram_bytes is not None:
        return observation.peak_vram_bytes, MemoryObservationSource.NVML_PROCESS_ALLOCATION
    allocation = observation.runtime_allocation
    if allocation is not None and allocation.buffers:
        return (
            allocation.total_device_bytes,
            MemoryObservationSource.RUNTIME_REPORTED_ALLOCATION,
        )
    return None, None


def _metric_comparison(
    *,
    predicted_bytes: int | None,
    measured_bytes: int | None,
    unavailable_availability: MetricComparisonAvailability | None = None,
    unavailable_reason: str | None = None,
    source: MemoryObservationSource | None = None,
    allocation: RuntimeReportedAllocation | None = None,
) -> MetricComparison:
    provenance: dict[str, object] = {"source": source}
    if source is MemoryObservationSource.RUNTIME_REPORTED_ALLOCATION:
        provenance["driver_confirmed"] = False
        provenance["runtime"] = allocation.runtime if allocation else None
    elif source is MemoryObservationSource.NVML_PROCESS_ALLOCATION:
        provenance["driver_confirmed"] = True

    if unavailable_availability is not None:
        return MetricComparison(
            predicted_bytes=predicted_bytes,
            measured_bytes=measured_bytes,
            error_bytes=None,
            absolute_error_bytes=None,
            error_percent=None,
            availability=unavailable_availability,
            unavailable_reason=unavailable_reason,
            **provenance,  # type: ignore[arg-type]
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
            **provenance,  # type: ignore[arg-type]
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
            **provenance,  # type: ignore[arg-type]
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
        **provenance,  # type: ignore[arg-type]
    )


def _predicted_ram(
    *,
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation | None,
) -> tuple[int | None, MetricComparisonAvailability | None, str | None]:
    """The host figure comparable with peak RSS, when there is one.

    On CPU-only runs there is: the prediction is every weight byte plus the KV
    cache and the overhead, and RSS covers exactly that, mapped model included.

    Under offload there is not, and a host/device split would not rescue it.
    ``HardwareFitResult`` does carry ``ram_weight_bytes`` and friends, so the
    breakdown exists — but RSS does not follow it. The B001-R4 baseline measured
    the same 4723 MiB peak RSS with 24 of 29 units offloaded and with all of
    them, against a predicted host share of 1742 MiB; the total artifact is
    4466 MiB. RSS tracked the mapped file, not the placement, and did not move
    when the placement did. Comparing the two would report a 171% error that
    says nothing about the memory model.
    """

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


def _predicted_vram(
    *,
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation | None,
) -> tuple[int | None, MetricComparisonAvailability | None, str | None]:
    """The VRAM figure that is comparable with a process-attributed measurement.

    ``peak_vram_bytes`` is what NVML attributed to the inference process, so the
    prediction has to be the bytes that process is expected to allocate — not
    the capacity budget the analyzer checked against the card. ``device_reserve``
    is memory deliberately left free for *other* processes and ``safety_margin``
    is policy padding; including either would compare a budget with an
    allocation and report an error that is really just the size of the policy.
    ``HardwareFitResult.gpu_physical_bytes`` is that budget with both removed.

    Runtime overhead stays in: it models allocator, compute and activation
    buffers, which the process really does allocate. It is a coarse heuristic,
    but being wrong about a real quantity is a calibration result — which is
    exactly what this comparison exists to surface.
    """

    fit = estimate.hardware_fit
    if fit is None:
        return (
            None,
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
            _VRAM_NO_FIT_REASON,
        )
    if not fit.places_weights_on_gpu:
        return (
            None,
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
            _VRAM_NO_GPU_PLACEMENT_REASON,
        )

    if (
        fit.non_block_placement_bounds is not None
        and fit.non_block_placement_bounds.non_block_weight_bytes > 0
    ):
        return (
            None,
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
            "Non-block weight placement is bounded, not a device-specific point prediction.",
        )
    predicted = fit.gpu_physical_bytes
    if predicted is None:
        return (
            None,
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
            _VRAM_NO_FIT_REASON,
        )

    mismatch = _placement_mismatch(fit, runtime)
    if mismatch is not None:
        return (
            None,
            MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
            mismatch,
        )
    return (predicted, None, None)


def _placement_mismatch(
    fit: HardwareFitResult,
    runtime: RuntimeRecommendation | None,
) -> str | None:
    """Explain why the run cannot be held to this prediction, or ``None``.

    A VRAM figure only tests the memory model if the execution placed the model
    the way the estimate assumed. Hardware Fit expresses offload placement in
    runtime-agnostic transformer blocks. llama.cpp exposes backend-specific
    ``--n-gpu-layers`` units, which the Qwen2.5 validation showed are not the
    same vocabulary, so partial offload is reported as unverifiable until a
    backend mapping exists. Transformers decides device placement internally and
    exposes no equivalent, so there the prediction is also reported as
    unverifiable rather than compared on trust.
    """

    if runtime is None or runtime.runtime is not RuntimeName.LLAMA_CPP:
        return _VRAM_UNVERIFIED_PLACEMENT_REASON

    requested = _n_gpu_layers(runtime)
    if requested is None:
        return _VRAM_UNVERIFIED_PLACEMENT_REASON

    # llama.cpp spells "every layer" as a negative value rather than a count.
    if requested < 0:
        if fit.mode is HardwareFitMode.GPU_RESIDENT:
            return None
        return (
            f"Run placed every layer on the GPU (--n-gpu-layers {requested}) but "
            f"the prediction is a {fit.mode.value} placement, so the two describe "
            "different splits."
        )

    if fit.gpu_transformer_blocks is None:
        return (
            "The predicted placement was sized in bytes rather than transformer "
            "blocks, so "
            f"it cannot be checked against --n-gpu-layers {requested}."
        )

    return _VRAM_RUNTIME_MAPPING_UNAVAILABLE_REASON


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
