"""Conditional placement bounds, never a runtime-specific tensor assignment."""

import pytest

from jaull.domain.comparison import MetricComparisonAvailability
from jaull.domain.estimation import (
    CompatibilityStatus,
    HardwareFitMode,
    HardwareFitResult,
    TransformerBlockWeightDecomposition,
)
from jaull.domain.estimation import (
    TransformerBlockWeightDecompositionMethod as Method,
)
from jaull.domain.execution import ExecutionObservation
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import GgufVariant, ModelConfig, ModelFile
from jaull.estimator import service
from jaull.estimator.hardware_fit import _calculate_offload_placement, analyze_components
from jaull.evaluation.comparison import compare_prediction
from jaull.reporting.estimation import _hardware_fit_to_dict, estimate_to_json_dict
from tests.test_hardware_fit_analyzer import _hardware
from tests.test_memory_estimator import _FakeClient, _gguf_analysis


def _decomposition(total=20, blocks=13, count=4, method=Method.CONFIG_PARAMETER_DECOMPOSITION):
    return TransformerBlockWeightDecomposition(
        total_weight_bytes=total,
        estimated_transformer_block_weight_bytes=blocks,
        estimated_non_block_weight_bytes=total - blocks,
        estimated_bytes_per_transformer_block=(blocks + count - 1) // count,
        total_transformer_blocks=count,
        method=method,
    )


def test_bounds_search_matches_exhaustive_all_non_block_splits() -> None:
    common = {
        "weights_bytes": 20, "kv_cache_bytes": 5, "overhead_bytes": 3,
        "device_reserve_bytes": 1, "safety_margin_bytes": 2, "total_transformer_blocks": 4,
    }
    by_count = {}
    for n in range(1, 4):
        block_gpu = (13 * n + 3) // 4
        placements = [
            _calculate_offload_placement(
                **common, gpu_transformer_blocks=n, gpu_weight_bytes=block_gpu + x,
            )
            for x in range(8)
        ]
        assert all(p is not None for p in placements)
        by_count[n] = [p for p in placements if p is not None]
    for vram in range(1, 31):
        for ram in range(1, 31):
            result = analyze_components(
                **common, hardware=_hardware(ram=ram, vram=vram),
                weight_decomposition=_decomposition(),
            )
            viable = [
                n for n, placements in by_count.items()
                if all(p.gpu_required_bytes <= vram and p.ram_required_bytes <= ram
                       for p in placements)
            ]
            if viable:
                assert result.mode is HardwareFitMode.GPU_OFFLOAD
                assert result.gpu_transformer_blocks == max(viable)
                d = result.offload_diagnostics
                assert d is not None
                assert d.search_ceiling_transformer_blocks >= max(viable)
                assert result.gpu_required_bytes == max(
                    p.gpu_required_bytes for p in by_count[max(viable)]
                )
                bounds = result.non_block_placement_bounds
                assert bounds is not None
                assert bounds.ram_required_max_bytes == max(
                    p.ram_required_bytes for p in by_count[max(viable)]
                )
                _assert_bounds(result)
                if d.search_ceiling_transformer_blocks > max(viable):
                    assert d.first_rejected_higher is not None
                    assert d.first_rejected_higher.gpu_transformer_blocks == max(viable) + 1
                    assert d.first_rejected_higher.gpu_required_bytes > vram
            else:
                assert result.mode in {HardwareFitMode.CPU_RAM, HardwareFitMode.TOO_LARGE}
                assert result.non_block_placement_bounds is None


def _assert_bounds(result) -> None:
    b = result.non_block_placement_bounds
    assert b is not None
    assert b.gpu_transformer_block_weight_bytes + b.ram_transformer_block_weight_bytes + (
        b.non_block_weight_bytes
    ) == result.weights_bytes
    assert result.gpu_weight_bytes == (
        b.gpu_transformer_block_weight_bytes + b.non_block_weight_bytes
    )
    assert result.ram_weight_bytes == b.ram_transformer_block_weight_bytes
    assert result.gpu_weight_bytes + result.ram_weight_bytes == result.weights_bytes
    assert result.gpu_required_bytes == (
        result.gpu_weight_bytes + result.gpu_kv_cache_bytes + result.device_reserve_bytes
        + result.gpu_overhead_bytes + result.gpu_safety_margin_bytes
    )
    assert result.ram_required_bytes == (
        result.ram_weight_bytes + result.ram_kv_cache_bytes
        + result.ram_overhead_bytes + result.ram_safety_margin_bytes
    )
    assert b.ram_required_max_bytes == (
        b.ram_transformer_block_weight_bytes + b.non_block_weight_bytes
        + result.ram_kv_cache_bytes + b.ram_overhead_max_bytes + b.ram_safety_margin_max_bytes
    )
    assert b.gpu_required_min_bytes == (
        b.gpu_transformer_block_weight_bytes + result.gpu_kv_cache_bytes
        + result.device_reserve_bytes + result.overhead_bytes - b.ram_overhead_max_bytes
        + result.safety_margin_bytes - b.ram_safety_margin_max_bytes
    )
    assert result.gpu_physical_bytes is None
    assert result.ram_physical_bytes is None


@pytest.mark.parametrize("ram,vram,unified,mode", [
    (40, 40, False, HardwareFitMode.GPU_RESIDENT),
    (40, 1, False, HardwareFitMode.CPU_RAM),
    (40, None, False, HardwareFitMode.CPU_RAM),
    (40, None, True, HardwareFitMode.CPU_RAM),
    (1, 1, False, HardwareFitMode.TOO_LARGE),
    (1, None, True, HardwareFitMode.TOO_LARGE),
])
def test_non_partial_modes_keep_all_weights_in_the_existing_pool(ram, vram, unified, mode):
    result = analyze_components(
        weights_bytes=20, kv_cache_bytes=5, overhead_bytes=3,
        hardware=_hardware(ram=ram, vram=vram, unified=unified),
        weight_decomposition=_decomposition(),
    )
    assert result.mode is mode
    assert result.non_block_placement_bounds is None
    assert result.gpu_weight_bytes + result.ram_weight_bytes == 20
    if mode is HardwareFitMode.GPU_RESIDENT:
        assert result.gpu_transformer_blocks == 4
        assert result.gpu_weight_bytes == 20
    elif vram is not None:
        assert result.gpu_transformer_blocks == 0


def test_uniform_fallback_retains_previous_decision_and_payload_reads() -> None:
    common = {"weights_bytes": 20, "kv_cache_bytes": 5, "overhead_bytes": 3,
              "total_transformer_blocks": 4, "hardware": _hardware(ram=40, vram=20)}
    before = analyze_components(**common)
    after = analyze_components(
        **common,
        weight_decomposition=_decomposition(blocks=20, method=Method.UNIFORM_WEIGHT_FALLBACK),
    )
    assert before == after
    payload = before.model_dump()
    payload.pop("non_block_placement_bounds")
    assert HardwareFitResult.model_validate(payload) == before


def test_qwen_bounds_are_not_calibrated_to_observed_runtime_units() -> None:
    common = {
        "weights_bytes": 4_683_074_240, "kv_cache_bytes": 234_881_024,
        "overhead_bytes": 1_005_178_336, "device_reserve_bytes": 536_870_912,
        "safety_margin_bytes": 646_000_452, "total_transformer_blocks": 28,
        "hardware": _hardware(ram=7_593_828_352, vram=4_985_380_864),
    }
    before = analyze_components(**common)
    after = analyze_components(
        **common, weight_decomposition=_decomposition(4_683_074_240, 4_012_773_975, 28),
    )
    assert before.gpu_transformer_blocks == 18
    assert after.mode is HardwareFitMode.GPU_OFFLOAD
    d = after.offload_diagnostics
    assert d is not None and d.first_rejected_higher is not None
    # These are outputs of the endpoint formulas, not a target for calibration.
    assert after.gpu_transformer_blocks == 17
    assert d.first_rejected_higher.gpu_transformer_blocks == 18
    assert d.search_ceiling_transformer_blocks == 24
    assert d.selected.headroom_bytes == 90_145_728
    assert d.first_rejected_higher.excess_bytes == 110_833_348
    _assert_bounds(after)
    payload = _hardware_fit_to_dict(after)
    assert payload is not None
    assert payload["non_block_placement_bounds"] == after.non_block_placement_bounds.model_dump()
    assert HardwareFitResult.model_validate_json(after.model_dump_json()) == after


def test_inconsistent_decomposition_is_rejected_not_silently_used() -> None:
    with pytest.raises(ValueError, match="estimated weights"):
        analyze_components(weights_bytes=21, kv_cache_bytes=0, overhead_bytes=0,
                           hardware=_hardware(ram=40), weight_decomposition=_decomposition())
    with pytest.raises(ValueError, match="block count"):
        analyze_components(weights_bytes=20, kv_cache_bytes=0, overhead_bytes=0,
                           total_transformer_blocks=8, hardware=_hardware(ram=40),
                           weight_decomposition=_decomposition())


def test_production_estimator_transports_decomposition_and_bounds_to_reporting() -> None:
    total = 4_683_074_240
    analysis = _gguf_analysis([GgufVariant(
        quantization="Q4_K_M", files=[ModelFile(path="model.gguf", size_bytes=total)],
        total_bytes=total,
    )]).model_copy(update={"config": ModelConfig(
        model_type="qwen2", num_hidden_layers=28, hidden_size=3584,
        num_attention_heads=28, num_key_value_heads=4, intermediate_size=18944,
        vocab_size=152064, tie_word_embeddings=False,
    )})
    estimate = service.estimate_memory(
        analysis, _hardware(ram=7_593_828_352, vram=4_985_380_864),
        InferenceConfiguration(context_length=4096, quantization="Q4_K_M",
                               device_reserve_bytes=536_870_912),
        _FakeClient(), resolve_base_model=False, recommend_runtime=False,
    )
    decomposition = estimate.weights.transformer_block_decomposition
    assert decomposition is not None
    assert decomposition.method is Method.CONFIG_PARAMETER_DECOMPOSITION
    assert decomposition.estimated_transformer_block_weight_bytes == 4_012_773_975
    fit = estimate.hardware_fit
    assert fit is not None and fit.non_block_placement_bounds is not None
    assert fit.non_block_placement_bounds.non_block_weight_bytes == 670_300_265
    assert fit.gpu_transformer_blocks == 17
    assert estimate.assessment.status is CompatibilityStatus.OFFLOADING_REQUIRED
    payload = estimate_to_json_dict(estimate)
    assert payload["hardware_fit"]["non_block_placement_bounds"]["non_block_weight_bytes"] == (
        decomposition.estimated_non_block_weight_bytes
    )
    assert payload["hardware_fit"]["gpu_physical_bytes"] is None
    comparison = compare_prediction(
        estimate=estimate,
        observation=ExecutionObservation(
            success=True, exit_code=0, duration_seconds=1, peak_vram_bytes=123,
        ),
    )
    assert comparison.vram.availability is MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE
    assert "Non-block" in (comparison.vram.unavailable_reason or "")
    assert comparison.vram.predicted_bytes is None
