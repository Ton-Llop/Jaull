from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from rich.console import Console

from jaull.domain.comparison import (
    CompatibilityOutcome,
    MetricComparison,
    MetricComparisonAvailability,
)
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimateSource,
    EstimationConfidence,
    HardwareFitMode,
    HardwareFitPlacementMethod,
    HardwareFitResult,
    HardwareMemoryTopology,
    KvCacheEstimate,
    MemoryComponent,
    MemoryEstimate,
    RuntimeOverheadEstimate,
    WeightEstimate,
)
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionMeasurementMetadata,
    ExecutionObservation,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import ModelRepositoryInfo
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.comparison import (
    PredictionRuntimeMismatchError,
    assert_prediction_runtime_matches,
    compare_prediction,
)
from jaull.presentation.comparison_report import render_prediction_comparison


def test_ram_exact_error_convention_underestimation() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=1100),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.ram.availability is MetricComparisonAvailability.AVAILABLE
    assert comparison.ram.predicted_bytes == 1000
    assert comparison.ram.measured_bytes == 1100
    assert comparison.ram.error_bytes == 100
    assert comparison.ram.absolute_error_bytes == 100
    assert comparison.ram.error_percent == pytest.approx(10.0)


def test_ram_error_convention_overestimation() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=800),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.ram.error_bytes == -200
    assert comparison.ram.absolute_error_bytes == 200
    assert comparison.ram.error_percent == pytest.approx(-20.0)


def test_ram_exact_match() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=1000),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.ram.error_bytes == 0
    assert comparison.ram.absolute_error_bytes == 0
    assert comparison.ram.error_percent == pytest.approx(0.0)


def test_measurement_unavailable_does_not_treat_as_zero() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=None),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert (
        comparison.ram.availability
        is MetricComparisonAvailability.MEASUREMENT_UNAVAILABLE
    )
    assert comparison.ram.measured_bytes is None
    assert comparison.ram.error_bytes is None
    assert comparison.ram.error_percent is None


def test_prediction_unavailable_when_component_missing() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=None, overhead=200),
        observation=_observation(ram=1000),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert (
        comparison.ram.availability
        is MetricComparisonAvailability.PREDICTION_UNAVAILABLE
    )
    assert comparison.ram.predicted_bytes is None
    assert comparison.ram.error_bytes is None


def test_prediction_zero_has_no_percent_division() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=0, kv=0, overhead=0),
        observation=_observation(ram=100),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.ram.availability is MetricComparisonAvailability.AVAILABLE
    assert comparison.ram.error_bytes == 100
    assert comparison.ram.absolute_error_bytes == 100
    assert comparison.ram.error_percent is None


def test_gpu_offload_ram_comparison_is_methodologically_unavailable() -> None:
    comparison = compare_prediction(
        estimate=_estimate(
            weights=500,
            kv=300,
            overhead=200,
            status=CompatibilityStatus.COMPATIBLE,
            effective_device=TargetDevice.GPU,
        ),
        observation=_observation(ram=700),
        runtime=_runtime(n_gpu_layers=99),
    )

    assert (
        comparison.ram.availability
        is MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert comparison.ram.predicted_bytes is None
    assert comparison.ram.measured_bytes == 700
    assert comparison.ram.error_bytes is None
    assert "host/device breakdown" in comparison.ram.unavailable_reason


def test_n_gpu_layers_zero_is_not_offload_for_ram_comparison() -> None:
    comparison = compare_prediction(
        estimate=_estimate(
            weights=500,
            kv=300,
            overhead=200,
            target_device=TargetDevice.CPU,
            effective_device=TargetDevice.CPU,
        ),
        observation=_observation(ram=1000),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.ram.availability is MetricComparisonAvailability.AVAILABLE
    assert comparison.ram.predicted_bytes == 1000


def test_predicted_process_ram_excludes_reserve_and_safety_margin() -> None:
    estimate = _estimate(
        weights=500,
        kv=300,
        overhead=200,
        target_device=TargetDevice.CPU,
        effective_device=TargetDevice.CPU,
    )

    comparison = compare_prediction(
        estimate=estimate,
        observation=_observation(ram=1000),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert estimate.device_reserve.bytes == 256
    assert estimate.safety_margin is not None
    assert estimate.safety_margin.bytes == 128
    assert estimate.total_bytes == 1384
    assert comparison.ram.predicted_bytes == 1000


def test_vram_prediction_is_not_invented() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=1000, vram=None),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert (
        comparison.vram.availability
        is MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert comparison.vram.predicted_bytes is None
    assert comparison.vram.measured_bytes is None
    assert comparison.vram.error_percent is None


@pytest.mark.parametrize(
    ("predicted", "success", "outcome"),
    [
        (CompatibilityStatus.COMPATIBLE, True, CompatibilityOutcome.CORRECT_SUCCESS),
        (CompatibilityStatus.COMPATIBLE, False, CompatibilityOutcome.FALSE_POSITIVE),
        (CompatibilityStatus.INSUFFICIENT, True, CompatibilityOutcome.FALSE_NEGATIVE),
        (
            CompatibilityStatus.INSUFFICIENT,
            False,
            CompatibilityOutcome.CORRECT_FAILURE,
        ),
        (CompatibilityStatus.UNKNOWN, True, CompatibilityOutcome.UNKNOWN),
    ],
)
def test_compatibility_outcomes(
    predicted: CompatibilityStatus,
    success: bool,
    outcome: CompatibilityOutcome,
) -> None:
    comparison = compare_prediction(
        estimate=_estimate(
            weights=500,
            kv=300,
            overhead=200,
            status=predicted,
            effective_device=TargetDevice.CPU,
        ),
        observation=_observation(ram=1000, success=success),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.compatibility.outcome is outcome


def test_offloading_required_is_runnable_only_with_matching_offload_runtime() -> None:
    estimate = _estimate(
        weights=500,
        kv=300,
        overhead=200,
        status=CompatibilityStatus.OFFLOADING_REQUIRED,
        effective_device=TargetDevice.GPU,
    )

    with_offload = compare_prediction(
        estimate=estimate,
        observation=_observation(ram=1000),
        runtime=_runtime(n_gpu_layers=8),
    )
    without_offload = compare_prediction(
        estimate=estimate,
        observation=_observation(ram=1000),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert with_offload.compatibility.predicted_runnable is True
    assert with_offload.compatibility.outcome is CompatibilityOutcome.CORRECT_SUCCESS
    assert without_offload.compatibility.predicted_runnable is None
    assert without_offload.compatibility.outcome is CompatibilityOutcome.UNKNOWN


def test_runtime_match_guard_accepts_cpu_only_runtime() -> None:
    estimate = _estimate(
        weights=500,
        kv=300,
        overhead=200,
        target_device=TargetDevice.CPU,
        effective_device=TargetDevice.CPU,
    )

    assert_prediction_runtime_matches(
        estimate=estimate,
        runtime=_runtime(n_gpu_layers=0, ctx_size=4096),
    )


def test_runtime_match_guard_rejects_context_mismatch() -> None:
    estimate = _estimate(
        weights=500,
        kv=300,
        overhead=200,
        target_device=TargetDevice.CPU,
        effective_device=TargetDevice.CPU,
    )

    with pytest.raises(PredictionRuntimeMismatchError, match="ctx-size"):
        assert_prediction_runtime_matches(
            estimate=estimate,
            runtime=_runtime(n_gpu_layers=0, ctx_size=2048),
        )


def test_runtime_match_guard_rejects_cpu_prediction_with_offload_runtime() -> None:
    estimate = _estimate(
        weights=500,
        kv=300,
        overhead=200,
        target_device=TargetDevice.CPU,
        effective_device=TargetDevice.CPU,
    )

    with pytest.raises(PredictionRuntimeMismatchError, match="CPU-only"):
        assert_prediction_runtime_matches(
            estimate=estimate,
            runtime=_runtime(n_gpu_layers=2, ctx_size=4096),
        )


def test_fake_tinyllama_cpu_e2e_estimate_observation_comparison() -> None:
    estimate = _estimate(
        weights=668_788_096,
        kv=92_274_688,
        overhead=603_749_721,
        status=CompatibilityStatus.COMPATIBLE,
        target_device=TargetDevice.CPU,
        effective_device=TargetDevice.CPU,
    )
    runtime = _runtime(n_gpu_layers=0, ctx_size=4096)
    observation = _observation(ram=1_250_000_000, success=True)

    assert_prediction_runtime_matches(estimate=estimate, runtime=runtime)
    comparison = compare_prediction(
        estimate=estimate,
        observation=observation,
        runtime=runtime,
    )

    assert comparison.ram.availability is MetricComparisonAvailability.AVAILABLE
    assert comparison.ram.predicted_bytes == 1_364_812_505
    assert comparison.ram.measured_bytes == 1_250_000_000
    assert comparison.ram.error_bytes == -114_812_505
    assert comparison.compatibility.outcome is CompatibilityOutcome.CORRECT_SUCCESS


def test_failure_reason_is_preserved_without_oom_inference() -> None:
    comparison = compare_prediction(
        estimate=_estimate(
            weights=500,
            kv=300,
            overhead=200,
            status=CompatibilityStatus.COMPATIBLE,
        ),
        observation=_observation(
            ram=1200,
            success=False,
            exit_code=1,
            reason=ExecutionFailureReason.NON_ZERO_EXIT,
        ),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.compatibility.failure_reason is ExecutionFailureReason.NON_ZERO_EXIT
    assert comparison.compatibility.outcome is CompatibilityOutcome.FALSE_POSITIVE


def test_prediction_comparison_is_json_serializable() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=1100),
        runtime=_runtime(n_gpu_layers=0),
    )

    payload = comparison.model_dump(mode="json")
    dumped = json.dumps(payload)

    assert json.loads(dumped) == payload
    assert payload["ram"]["error_bytes"] == 100
    assert payload["compatibility"]["outcome"] == "correct_success"


def test_metric_comparison_rejects_contradictory_states() -> None:
    with pytest.raises(ValidationError):
        MetricComparison(
            predicted_bytes=1000,
            measured_bytes=1100,
            error_bytes=99,
            absolute_error_bytes=99,
            error_percent=9.9,
            availability=MetricComparisonAvailability.AVAILABLE,
        )


def test_prediction_comparison_rich_renderer() -> None:
    comparison = compare_prediction(
        estimate=_estimate(weights=500, kv=300, overhead=200),
        observation=_observation(ram=1100),
        runtime=_runtime(n_gpu_layers=0),
    )
    console = Console(record=True, width=100)

    render_prediction_comparison(console, comparison)
    output = console.export_text()

    assert "Prediction validation" in output
    assert "RAM" in output
    assert "+10.0%" in output
    assert "correct_success" in output


def _estimate(
    *,
    weights: int | None,
    kv: int | None,
    overhead: int | None,
    status: CompatibilityStatus = CompatibilityStatus.COMPATIBLE,
    target_device: TargetDevice = TargetDevice.AUTO,
    effective_device: TargetDevice = TargetDevice.CPU,
    hardware_fit: HardwareFitResult | None = None,
) -> MemoryEstimate:
    total = _sum_optional([weights, kv, overhead])
    return MemoryEstimate(
        hardware_fit=hardware_fit,
        repository=ModelRepositoryInfo(repo_id="org/model"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=target_device,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=_component("Weights", weights),
            gguf_variant="Q4_K_M",
        ),
        kv_cache=KvCacheEstimate(
            component=_component("KV cache", kv),
            layers=32,
            kv_heads=8,
            head_dim=128,
            context_length=4096,
            batch_size=1,
            dtype_bytes=2,
            formula="test",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=_component("Runtime overhead", overhead),
            base_bytes=0,
            weight_fraction=0.0,
            minimum_bytes=0,
        ),
        device_reserve=_component("Device reserve", 256),
        safety_margin=_component("Safety margin", 128),
        total_bytes=total + 256 + 128 if total is not None else None,
        assessment=CompatibilityAssessment(
            status=status,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.AUTO,
            effective_device=effective_device,
            available_vram_bytes=4096,
            available_ram_bytes=8192,
            ratio=0.5,
        ),
    )


def _component(name: str, value: int | None) -> MemoryComponent:
    return MemoryComponent(
        name=name,
        bytes=value,
        source=EstimateSource.DERIVED if value is not None else EstimateSource.UNKNOWN,
        confidence=(
            EstimationConfidence.HIGH
            if value is not None
            else EstimationConfidence.UNKNOWN
        ),
        explanation="test",
    )


def _sum_optional(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _observation(
    *,
    ram: int | None,
    vram: int | None = None,
    success: bool = True,
    exit_code: int | None = 0,
    reason: ExecutionFailureReason | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=1.0,
        peak_ram_bytes=ram,
        peak_vram_bytes=vram,
        exit_code=exit_code,
        failure_reason=reason,
        measurement=ExecutionMeasurementMetadata(sample_interval_seconds=0.05),
    )


def _runtime(n_gpu_layers: int, *, ctx_size: int | None = None) -> RuntimeRecommendation:
    flags = []
    if ctx_size is not None:
        flags.append(
            RuntimeFlag(
                name="--ctx-size",
                value=str(ctx_size),
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            )
        )
    flags.append(
        RuntimeFlag(
            name="--n-gpu-layers",
            value=str(n_gpu_layers),
            source=RuntimeFlagSource.HARDWARE,
            explanation="test",
        )
    )
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        confidence=EstimationConfidence.HIGH,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# VRAM: budget is not allocation
# ---------------------------------------------------------------------------
# ``peak_vram_bytes`` is what NVML attributed to the process, so the prediction
# has to be the bytes that process allocates. ``gpu_required_bytes`` is a
# capacity budget: it also holds the device reserve (memory left free for
# *other* processes) and the safety margin (policy padding). Comparing the
# budget would report the size of the policy as if it were estimator error.


def _fit(
    *,
    mode: HardwareFitMode = HardwareFitMode.GPU_RESIDENT,
    gpu_weight: int = 600,
    kv: int = 300,
    gpu_overhead: int = 100,
    device_reserve: int = 250,
    gpu_margin: int = 50,
    gpu_transformer_blocks: int | None = 32,
    total_transformer_blocks: int | None = 32,
    placement: HardwareFitPlacementMethod = (
        HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
    ),
    ram_weight: int = 0,
) -> HardwareFitResult:
    """A placement in round numbers so the arithmetic reads by hand.

    Physical = 600 + 300 + 100 = 1000. Budget = physical + 250 + 50 = 1300.
    """

    gpu_required = gpu_weight + kv + gpu_overhead + device_reserve + gpu_margin
    return HardwareFitResult(
        mode=mode,
        memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
        weights_bytes=gpu_weight + ram_weight,
        kv_cache_bytes=kv,
        overhead_bytes=gpu_overhead,
        device_reserve_bytes=device_reserve,
        safety_margin_bytes=gpu_margin,
        available_vram_bytes=8000,
        available_ram_bytes=16000,
        gpu_required_bytes=gpu_required,
        gpu_weight_bytes=gpu_weight,
        gpu_overhead_bytes=gpu_overhead,
        gpu_safety_margin_bytes=gpu_margin,
        ram_required_bytes=ram_weight,
        ram_weight_bytes=ram_weight,
        gpu_transformer_blocks=gpu_transformer_blocks,
        total_transformer_blocks=total_transformer_blocks,
        placement_method=placement,
        reason="test placement",
    )


def _gpu_estimate(fit: HardwareFitResult | None) -> MemoryEstimate:
    return _estimate(
        weights=600,
        kv=300,
        overhead=100,
        effective_device=TargetDevice.GPU,
        hardware_fit=fit,
    )


def test_vram_compares_the_allocation_not_the_budget() -> None:
    """The headline case: reserve and margin must not count as prediction."""

    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is MetricComparisonAvailability.AVAILABLE
    # 1000, not the 1300 budget.
    assert comparison.vram.predicted_bytes == 1000
    assert comparison.vram.error_bytes == 0
    assert comparison.vram.error_percent == pytest.approx(0.0)


def test_vram_overprediction_is_a_negative_error() -> None:
    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=None, vram=800),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is MetricComparisonAvailability.AVAILABLE
    assert comparison.vram.error_bytes == -200
    assert comparison.vram.absolute_error_bytes == 200
    assert comparison.vram.error_percent == pytest.approx(-20.0)


def test_vram_underprediction_is_a_positive_error() -> None:
    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=None, vram=1250),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is MetricComparisonAvailability.AVAILABLE
    assert comparison.vram.error_bytes == 250
    assert comparison.vram.error_percent == pytest.approx(25.0)


def test_vram_without_a_measurement_reports_the_measurement_missing() -> None:
    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=None, vram=None),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.MEASUREMENT_UNAVAILABLE
    )
    assert comparison.vram.predicted_bytes == 1000
    assert comparison.vram.error_bytes is None


def test_vram_without_a_fit_stays_methodologically_unavailable() -> None:
    """An estimate from before this contract must not silently compare."""

    comparison = compare_prediction(
        estimate=_gpu_estimate(None),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert comparison.vram.predicted_bytes is None
    assert "no structured hardware fit" in (comparison.vram.unavailable_reason or "")


def test_partial_offload_is_not_compared_without_a_runtime_mapping() -> None:
    fit = _fit(
        mode=HardwareFitMode.GPU_OFFLOAD,
        gpu_weight=600,
        ram_weight=400,
        gpu_transformer_blocks=24,
        total_transformer_blocks=32,
    )

    comparison = compare_prediction(
        estimate=_gpu_estimate(fit),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=24),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert comparison.vram.predicted_bytes is None
    reason = comparison.vram.unavailable_reason or ""
    assert "transformer blocks" in reason
    assert "--n-gpu-layers" in reason


# ---------------------------------------------------------------------------
# VRAM: refusing to compare different things
# ---------------------------------------------------------------------------
def test_a_cpu_placement_predicts_no_process_vram() -> None:
    """``gpu_required_bytes`` on a rejected GPU is hypothetical, not a forecast."""

    fit = _fit(mode=HardwareFitMode.CPU_RAM, gpu_weight=0, gpu_transformer_blocks=0)

    comparison = compare_prediction(
        estimate=_gpu_estimate(fit),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=0),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert "no weights on the GPU" in (comparison.vram.unavailable_reason or "")


def test_a_backend_specific_layer_count_is_not_a_transformer_block_count() -> None:
    """Do not treat llama.cpp offload units as Hardware Fit transformer blocks."""

    fit = _fit(
        mode=HardwareFitMode.GPU_OFFLOAD,
        gpu_transformer_blocks=24,
        total_transformer_blocks=32,
    )

    comparison = compare_prediction(
        estimate=_gpu_estimate(fit),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=30),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    reason = comparison.vram.unavailable_reason or ""
    assert "transformer blocks" in reason
    assert "--n-gpu-layers" in reason


def test_running_every_runtime_layer_on_gpu_contradicts_offload_prediction() -> None:
    fit = _fit(
        mode=HardwareFitMode.GPU_OFFLOAD,
        gpu_transformer_blocks=24,
        total_transformer_blocks=32,
    )

    comparison = compare_prediction(
        estimate=_gpu_estimate(fit),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert "every layer" in (comparison.vram.unavailable_reason or "")


def test_a_byte_sized_placement_cannot_be_checked_against_a_layer_flag() -> None:
    fit = _fit(
        mode=HardwareFitMode.GPU_OFFLOAD,
        gpu_transformer_blocks=None,
        total_transformer_blocks=None,
        placement=HardwareFitPlacementMethod.ESTIMATED_BYTES,
    )

    comparison = compare_prediction(
        estimate=_gpu_estimate(fit),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=24),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert "sized in bytes" in (comparison.vram.unavailable_reason or "")


def test_without_a_runtime_the_placement_cannot_be_verified() -> None:
    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=None, vram=1000),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert "cannot be verified" in (comparison.vram.unavailable_reason or "")


def test_transformers_placement_is_reported_as_unverifiable() -> None:
    """Transformers decides device placement itself and exposes no equivalent flag."""

    runtime = RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        confidence=EstimationConfidence.HIGH,
        flags=[],
    )

    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=None, vram=1000),
        runtime=runtime,
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )


def test_resident_prediction_accepts_runtime_full_offload() -> None:
    """``-1`` is the runtime-specific spelling of full GPU residency."""

    comparison = compare_prediction(
        estimate=_gpu_estimate(
            _fit(gpu_transformer_blocks=32, total_transformer_blocks=32)
        ),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.vram.availability is MetricComparisonAvailability.AVAILABLE


def test_resident_prediction_rejects_explicit_runtime_count_without_mapping() -> None:
    comparison = compare_prediction(
        estimate=_gpu_estimate(
            _fit(gpu_transformer_blocks=32, total_transformer_blocks=32)
        ),
        observation=_observation(ram=None, vram=1000),
        runtime=_runtime(n_gpu_layers=32),
    )

    assert comparison.vram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert "transformer blocks" in (comparison.vram.unavailable_reason or "")


def test_wiring_vram_did_not_change_the_ram_verdict() -> None:
    """The RAM path is untouched by this milestone, including its refusal."""

    comparison = compare_prediction(
        estimate=_gpu_estimate(_fit()),
        observation=_observation(ram=900, vram=1000),
        runtime=_runtime(n_gpu_layers=-1),
    )

    assert comparison.ram.availability is (
        MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    )
    assert "host/device breakdown" in (comparison.ram.unavailable_reason or "")
