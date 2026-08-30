from __future__ import annotations

import math

import pytest

from jaull.domain.estimation import (
    HardwareFitMode,
    HardwareFitPlacementMethod,
    HardwareMemoryTopology,
)
from jaull.domain.hardware import (
    AcceleratorProfile,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.estimator.hardware_fit import (
    _split_kv_cache_by_transformer_blocks,
    analyze_components,
)

GIB = 1024**3


def _hardware(
    *,
    ram: int,
    vram: int | None = None,
    unified: bool = False,
) -> HardwareProfile:
    gpus = (
        [
            GpuInfo(
                name="Test GPU",
                vram_total_bytes=vram,
                vram_available_bytes=vram,
                driver_version="1.0",
                cuda_version="12.0",
            )
        ]
        if vram is not None and not unified
        else []
    )
    accelerators = (
        [
            AcceleratorProfile(
                name="Unified accelerator",
                shared_memory=True,
                available_memory_bytes=ram,
            )
        ]
        if unified
        else []
    )
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=ram * 2, available_bytes=ram),
        storage=[],
        gpus=gpus,
        accelerators=accelerators,
        warnings=[],
    )


def test_gpu_resident_when_full_requirement_fits_vram() -> None:
    result = analyze_components(
        weights_bytes=5 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=32 * GIB, vram=8 * GIB),
        total_transformer_blocks=20,
    )

    assert result.mode is HardwareFitMode.GPU_RESIDENT
    assert result.gpu_weight_bytes == 5 * GIB
    assert result.ram_weight_bytes == 0
    assert result.gpu_transformer_blocks == 20
    assert result.offload_diagnostics is None


def test_gpu_offload_places_some_weights_on_gpu_and_rest_in_ram() -> None:
    result = analyze_components(
        weights_bytes=8 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=16 * GIB, vram=6 * GIB),
        total_transformer_blocks=8,
    )

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    assert result.placement_method is HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
    assert result.gpu_weight_bytes > 0
    assert result.ram_weight_bytes > 0
    assert result.gpu_transformer_blocks == 4
    diagnostics = result.offload_diagnostics
    assert diagnostics is not None
    assert diagnostics.search_ceiling_transformer_blocks > result.gpu_transformer_blocks


def test_offload_splits_overhead_and_safety_margin_by_pool() -> None:
    overhead = int(2.5 * GIB)
    safety_margin = int(2.4 * GIB)
    result = analyze_components(
        weights_bytes=20 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=overhead,
        device_reserve_bytes=int(0.5 * GIB),
        safety_margin_bytes=safety_margin,
        hardware=_hardware(ram=32 * GIB, vram=8 * GIB),
        total_transformer_blocks=20,
    )

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    assert result.gpu_transformer_blocks is not None
    assert result.gpu_transformer_blocks > 1
    assert 0 < result.gpu_overhead_bytes < overhead
    assert 0 < result.ram_overhead_bytes < overhead
    assert result.gpu_overhead_bytes + result.ram_overhead_bytes == overhead
    assert 0 < result.gpu_safety_margin_bytes < safety_margin
    assert 0 < result.ram_safety_margin_bytes < safety_margin
    assert (
        result.gpu_safety_margin_bytes + result.ram_safety_margin_bytes
        == safety_margin
    )


def test_cpu_ram_when_gpu_placement_is_not_viable_but_ram_fits() -> None:
    """Not even one block fits, so nothing lands on the GPU — cache included."""

    result = analyze_components(
        weights_bytes=4 * GIB,
        kv_cache_bytes=2 * GIB,
        overhead_bytes=2 * GIB,
        hardware=_hardware(ram=12 * GIB, vram=3 * GIB),
        total_transformer_blocks=2,
    )

    assert result.mode is HardwareFitMode.CPU_RAM
    assert result.gpu_weight_bytes == 0
    assert result.gpu_kv_cache_bytes == 0
    assert result.ram_kv_cache_bytes == 2 * GIB
    assert result.ram_required_bytes == 8 * GIB
    assert result.offload_diagnostics is None


def test_too_large_when_no_supported_placement_fits() -> None:
    result = analyze_components(
        weights_bytes=12 * GIB,
        kv_cache_bytes=4 * GIB,
        overhead_bytes=2 * GIB,
        hardware=_hardware(ram=10 * GIB, vram=4 * GIB),
        total_transformer_blocks=12,
    )

    assert result.mode is HardwareFitMode.TOO_LARGE
    assert "No supported placement fits" in result.reason
    assert result.offload_diagnostics is None


def test_offload_does_not_use_vram_plus_ram_as_single_pool() -> None:
    result = analyze_components(
        weights_bytes=10 * GIB,
        kv_cache_bytes=5 * GIB,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=10 * GIB, vram=6 * GIB),
        total_transformer_blocks=10,
    )

    assert result.weights_bytes + result.kv_cache_bytes + result.overhead_bytes == (
        result.available_ram_bytes + result.available_vram_bytes
    )
    assert result.mode is HardwareFitMode.TOO_LARGE
    assert result.gpu_weight_bytes == 0


def test_exact_boundary_counts_as_fit() -> None:
    result = analyze_components(
        weights_bytes=6 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=32 * GIB, vram=8 * GIB),
    )

    assert result.mode is HardwareFitMode.GPU_RESIDENT
    assert result.gpu_required_bytes == 8 * GIB


def test_kv_cache_can_change_gpu_resident_to_offload() -> None:
    hardware = _hardware(ram=16 * GIB, vram=6 * GIB)

    short_context = analyze_components(
        weights_bytes=5 * GIB,
        kv_cache_bytes=512 * 1024**2,
        overhead_bytes=512 * 1024**2,
        hardware=hardware,
        total_transformer_blocks=10,
    )
    long_context = analyze_components(
        weights_bytes=5 * GIB,
        kv_cache_bytes=2 * GIB,
        overhead_bytes=512 * 1024**2,
        hardware=hardware,
        total_transformer_blocks=10,
    )

    assert short_context.mode is HardwareFitMode.GPU_RESIDENT
    assert long_context.mode is HardwareFitMode.GPU_OFFLOAD


def test_concurrency_scaled_kv_can_change_fit() -> None:
    hardware = _hardware(ram=16 * GIB, vram=6 * GIB)

    one_user = analyze_components(
        weights_bytes=5 * GIB,
        kv_cache_bytes=512 * 1024**2,
        overhead_bytes=512 * 1024**2,
        hardware=hardware,
        total_transformer_blocks=10,
    )
    four_users = analyze_components(
        weights_bytes=5 * GIB,
        kv_cache_bytes=2 * GIB,
        overhead_bytes=512 * 1024**2,
        hardware=hardware,
        total_transformer_blocks=10,
    )

    assert one_user.mode is HardwareFitMode.GPU_RESIDENT
    assert four_users.mode is HardwareFitMode.GPU_OFFLOAD


def test_transformer_block_metadata_controls_placement_method() -> None:
    hardware = _hardware(ram=16 * GIB, vram=6 * GIB)

    block_aware = analyze_components(
        weights_bytes=8 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=hardware,
        total_transformer_blocks=8,
    )
    byte_fallback = analyze_components(
        weights_bytes=8 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=hardware,
        total_transformer_blocks=None,
    )

    assert block_aware.mode is HardwareFitMode.GPU_OFFLOAD
    assert block_aware.placement_method is (
        HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
    )
    assert byte_fallback.mode is HardwareFitMode.GPU_OFFLOAD
    assert byte_fallback.placement_method is HardwareFitPlacementMethod.ESTIMATED_BYTES
    assert byte_fallback.gpu_transformer_blocks is None
    assert block_aware.offload_diagnostics is not None
    assert byte_fallback.offload_diagnostics is None


def test_offload_diagnostics_record_when_search_ceiling_was_the_selected_block() -> None:
    result = analyze_components(
        weights_bytes=8 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=0,
        hardware=_hardware(ram=16 * GIB, vram=5 * GIB),
        total_transformer_blocks=8,
    )

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    assert result.gpu_transformer_blocks == 4
    diagnostics = result.offload_diagnostics
    assert diagnostics is not None
    assert diagnostics.search_ceiling_transformer_blocks == 4
    assert diagnostics.selected.gpu_transformer_blocks == 4
    assert diagnostics.first_rejected_higher is None
    assert diagnostics.selected.excess_bytes == 0
    assert diagnostics.selected.headroom_bytes == (
        diagnostics.selected.available_vram_bytes
        - diagnostics.selected.gpu_required_bytes
    )


def test_total_transformer_blocks_candidate_is_not_recorded_as_partial_rejection() -> None:
    result = analyze_components(
        weights_bytes=8 * GIB,
        kv_cache_bytes=0,
        overhead_bytes=2 * GIB,
        hardware=_hardware(ram=16 * GIB, vram=9 * GIB),
        total_transformer_blocks=8,
    )

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    assert result.gpu_transformer_blocks == 7
    diagnostics = result.offload_diagnostics
    assert diagnostics is not None
    assert diagnostics.search_ceiling_transformer_blocks == 8
    assert diagnostics.selected.gpu_transformer_blocks == 7
    assert diagnostics.first_rejected_higher is None


def test_qwen_7b_rtx2060_regression_reports_transformer_blocks_not_runtime_layers() -> None:
    result = analyze_components(
        weights_bytes=4_683_074_240,
        kv_cache_bytes=234_881_024,
        overhead_bytes=1_005_178_336,
        device_reserve_bytes=536_870_912,
        safety_margin_bytes=646_000_452,
        hardware=_hardware(ram=7_593_828_352, vram=4_985_380_864),
        total_transformer_blocks=28,
    )

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    assert result.placement_method is HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
    assert result.gpu_transformer_blocks == 18
    assert result.total_transformer_blocks == 28

    diagnostics = result.offload_diagnostics
    assert diagnostics is not None
    assert diagnostics.search_ceiling_transformer_blocks == 25
    selected = diagnostics.selected
    rejected = diagnostics.first_rejected_higher

    assert selected.gpu_transformer_blocks == 18
    assert selected.gpu_required_bytes == result.gpu_required_bytes
    assert selected.gpu_required_bytes <= selected.available_vram_bytes
    assert selected.excess_bytes == 0
    assert selected.headroom_bytes == (
        selected.available_vram_bytes - selected.gpu_required_bytes
    )
    assert selected.headroom_bytes > 0

    assert rejected is not None
    assert rejected.gpu_transformer_blocks == 19
    assert rejected.available_vram_bytes == result.available_vram_bytes
    assert rejected.gpu_required_bytes > rejected.available_vram_bytes
    assert rejected.excess_bytes == (
        rejected.gpu_required_bytes - rejected.available_vram_bytes
    )
    assert rejected.headroom_bytes == 0
    assert rejected.ram_required_bytes <= selected.ram_required_bytes
    assert rejected.gpu_required_bytes == (
        rejected.gpu_weight_bytes
        + rejected.gpu_kv_cache_bytes
        + rejected.device_reserve_bytes
        + rejected.gpu_overhead_bytes
        + rejected.gpu_safety_margin_bytes
    )

    # The KV cache follows the blocks: 18 of 28 on the GPU, 19 of 28 for the
    # candidate above it. Both shares round up on the GPU side and leave RAM
    # the remainder, so neither creates nor loses a byte.
    assert selected.gpu_kv_cache_bytes == math.ceil(234_881_024 * 18 / 28)
    assert rejected.gpu_kv_cache_bytes == math.ceil(234_881_024 * 19 / 28)
    for candidate in (selected, rejected):
        assert (
            candidate.gpu_kv_cache_bytes + candidate.ram_kv_cache_bytes
            == candidate.kv_cache_bytes
            == 234_881_024
        )


def test_qwen_7b_rtx2060_kv_placement_moves_the_boundary_without_moving_the_choice()  -> None:
    """What the KV split changes, and what it does not, on the measured fixture.

    Charging the whole cache to VRAM overstated every partial offload. Fixing it
    shrinks the gap at block 19 by most of its size — and still does not close
    it, so the analyzer keeps selecting 18. The correction is to the model, not
    to the answer; this test exists so a future change to either is visible.
    """

    result = analyze_components(
        weights_bytes=4_683_074_240,
        kv_cache_bytes=234_881_024,
        overhead_bytes=1_005_178_336,
        device_reserve_bytes=536_870_912,
        safety_margin_bytes=646_000_452,
        hardware=_hardware(ram=7_593_828_352, vram=4_985_380_864),
        total_transformer_blocks=28,
    )

    diagnostics = result.offload_diagnostics
    assert diagnostics is not None
    rejected = diagnostics.first_rejected_higher
    assert rejected is not None

    assert result.gpu_transformer_blocks == 18
    assert rejected.gpu_transformer_blocks == 19

    # Block 19 still misses, but by far less than the 126_394_504 bytes it
    # missed by while the whole cache was charged to VRAM.
    assert 0 < rejected.excess_bytes < 126_394_504 // 4

    # RAM now carries the share of the cache that the GPU does not.
    assert result.ram_required_bytes == (
        result.ram_weight_bytes
        + result.ram_kv_cache_bytes
        + result.ram_overhead_bytes
        + result.ram_safety_margin_bytes
    )
    assert result.ram_kv_cache_bytes > 0


# ---------------------------------------------------------------------------
# KV cache placement follows transformer-block placement
# ---------------------------------------------------------------------------
def _offload(
    *,
    blocks: int,
    kv: int = 2 * GIB,
    weights: int = 8 * GIB,
    vram: int = 6 * GIB,
    ram: int = 64 * GIB,
) -> object:
    return analyze_components(
        weights_bytes=weights,
        kv_cache_bytes=kv,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=ram, vram=vram),
        total_transformer_blocks=blocks,
    )


@pytest.mark.parametrize("blocks", [2, 3, 7, 8, 16, 28, 32, 40, 80])
@pytest.mark.parametrize("kv", [1, 3, 7, 1024, 234_881_024, 3 * GIB])
def test_kv_cache_split_conserves_every_byte(blocks: int, kv: int) -> None:
    """No byte is created or lost, whatever the ratio does to the rounding."""

    result = _offload(blocks=blocks, kv=kv)

    assert result.gpu_kv_cache_bytes + result.ram_kv_cache_bytes == kv
    assert result.gpu_kv_cache_bytes >= 0
    assert result.ram_kv_cache_bytes >= 0


def test_gpu_resident_keeps_the_whole_kv_cache_on_the_gpu() -> None:
    result = analyze_components(
        weights_bytes=3 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=512 * 1024**2,
        hardware=_hardware(ram=32 * GIB, vram=16 * GIB),
        total_transformer_blocks=20,
    )

    assert result.mode is HardwareFitMode.GPU_RESIDENT
    assert result.gpu_kv_cache_bytes == 1 * GIB
    assert result.ram_kv_cache_bytes == 0


def test_too_large_leaves_the_whole_kv_cache_in_ram() -> None:
    result = analyze_components(
        weights_bytes=12 * GIB,
        kv_cache_bytes=4 * GIB,
        overhead_bytes=2 * GIB,
        hardware=_hardware(ram=10 * GIB, vram=4 * GIB),
        total_transformer_blocks=2,
    )

    assert result.mode is HardwareFitMode.TOO_LARGE
    assert result.gpu_kv_cache_bytes == 0
    assert result.ram_kv_cache_bytes == 4 * GIB


def test_partial_offload_splits_the_kv_cache_by_transformer_blocks() -> None:
    result = _offload(blocks=8, kv=2 * GIB)

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    placed = result.gpu_transformer_blocks
    assert placed is not None
    assert 0 < placed < 8
    assert result.gpu_kv_cache_bytes == math.ceil(2 * GIB * placed / 8)
    assert result.ram_kv_cache_bytes == 2 * GIB - result.gpu_kv_cache_bytes


def test_the_kv_share_follows_the_block_ratio_at_every_context_size() -> None:
    """A larger context grows the cache and may move the placement.

    What must not change is the rule: whatever placement the analyzer lands on,
    the cache follows it in the same proportion. A bigger context legitimately
    pushes blocks off the GPU — that is the fit reacting, not the split
    drifting.
    """

    observed = []
    for kv in (1 * GIB, 2 * GIB, 4 * GIB, 8 * GIB):
        result = analyze_components(
            weights_bytes=8 * GIB,
            kv_cache_bytes=kv,
            overhead_bytes=1 * GIB,
            hardware=_hardware(ram=64 * GIB, vram=12 * GIB),
            total_transformer_blocks=16,
        )
        placed = result.gpu_transformer_blocks
        assert placed is not None
        assert result.gpu_kv_cache_bytes + result.ram_kv_cache_bytes == kv
        assert result.gpu_kv_cache_bytes == math.ceil(kv * placed / 16)
        observed.append((kv, placed))

    # A growing cache never buys more blocks on the GPU.
    placements = [placed for _, placed in observed]
    assert placements == sorted(placements, reverse=True), observed


def test_no_blocks_on_the_gpu_puts_no_kv_cache_there() -> None:
    gpu_kv, ram_kv = _split_kv_cache_by_transformer_blocks(
        1000, gpu_transformer_blocks=0, total_transformer_blocks=8
    )

    assert (gpu_kv, ram_kv) == (0, 1000)


def test_every_block_on_the_gpu_puts_the_whole_kv_cache_there() -> None:
    gpu_kv, ram_kv = _split_kv_cache_by_transformer_blocks(
        1000, gpu_transformer_blocks=8, total_transformer_blocks=8
    )

    assert (gpu_kv, ram_kv) == (1000, 0)


def test_the_gpu_share_rounds_up_so_the_scarcer_pool_is_never_understated() -> None:
    """Same convention as the overhead and margin splits: GPU up, RAM remainder."""

    gpu_kv, ram_kv = _split_kv_cache_by_transformer_blocks(
        100, gpu_transformer_blocks=1, total_transformer_blocks=3
    )

    assert gpu_kv == 34  # ceil(100 / 3), not 33
    assert ram_kv == 66
    assert gpu_kv + ram_kv == 100


def test_byte_estimated_fallback_does_not_invent_a_block_aware_split() -> None:
    """Without a block count there is no placement for the cache to follow."""

    result = analyze_components(
        weights_bytes=8 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=32 * GIB, vram=6 * GIB),
        total_transformer_blocks=None,
    )

    assert result.mode is HardwareFitMode.GPU_OFFLOAD
    assert result.placement_method is HardwareFitPlacementMethod.ESTIMATED_BYTES
    assert result.gpu_transformer_blocks is None
    # Conservative: the whole cache stays charged to VRAM, as before.
    assert result.gpu_kv_cache_bytes == 1 * GIB
    assert result.ram_kv_cache_bytes == 0


@pytest.mark.parametrize("blocks", [4, 8, 28, 40])
def test_both_pools_reconstruct_exactly_from_their_components(blocks: int) -> None:
    result = _offload(blocks=blocks)

    assert result.gpu_required_bytes == (
        result.gpu_weight_bytes
        + result.gpu_kv_cache_bytes
        + result.device_reserve_bytes
        + result.gpu_overhead_bytes
        + result.gpu_safety_margin_bytes
    )
    assert result.ram_required_bytes == (
        result.ram_weight_bytes
        + result.ram_kv_cache_bytes
        + result.ram_overhead_bytes
        + result.ram_safety_margin_bytes
    )


@pytest.mark.parametrize("blocks", [4, 8, 28, 40])
def test_diagnostics_candidates_reconstruct_both_pools(blocks: int) -> None:
    diagnostics = _offload(blocks=blocks).offload_diagnostics
    assert diagnostics is not None

    candidates = [diagnostics.selected]
    if diagnostics.first_rejected_higher is not None:
        candidates.append(diagnostics.first_rejected_higher)

    for candidate in candidates:
        assert candidate.gpu_required_bytes == (
            candidate.gpu_weight_bytes
            + candidate.gpu_kv_cache_bytes
            + candidate.device_reserve_bytes
            + candidate.gpu_overhead_bytes
            + candidate.gpu_safety_margin_bytes
        )
        assert candidate.ram_required_bytes == (
            candidate.ram_weight_bytes
            + candidate.ram_kv_cache_bytes
            + candidate.ram_overhead_bytes
            + candidate.ram_safety_margin_bytes
        )
        assert (
            candidate.gpu_kv_cache_bytes + candidate.ram_kv_cache_bytes
            == candidate.kv_cache_bytes
        )


def test_the_kv_share_contributes_to_the_step_between_consecutive_candidates() -> None:
    """One more block costs its KV share too, not only its weights.

    While the whole cache was charged to VRAM the KV term was identical at every
    candidate, so the step from one block count to the next was weights plus the
    re-weighted padding and nothing else. Once the cache follows the placement
    that stops being true, and the step has a KV component. Pinned here because
    it is easy to keep describing the boundary the old way.
    """

    # Tight enough that the ceiling candidate is rejected, so there is a step to
    # measure rather than an immediately feasible first try.
    result = _offload(blocks=8, vram=5 * GIB)
    diagnostics = result.offload_diagnostics
    assert diagnostics is not None
    selected = diagnostics.selected
    rejected = diagnostics.first_rejected_higher
    assert rejected is not None
    assert rejected.gpu_transformer_blocks == selected.gpu_transformer_blocks + 1

    step = rejected.gpu_required_bytes - selected.gpu_required_bytes
    parts = {
        name: getattr(rejected, name) - getattr(selected, name)
        for name in (
            "gpu_weight_bytes",
            "gpu_kv_cache_bytes",
            "device_reserve_bytes",
            "gpu_overhead_bytes",
            "gpu_safety_margin_bytes",
        )
    }

    assert sum(parts.values()) == step
    assert parts["gpu_kv_cache_bytes"] > 0, parts
    assert parts["device_reserve_bytes"] == 0, "the reserve is per-device, not per-block"

    # The step is exactly the room that was left over plus the room that was
    # missing, which is what makes the boundary explainable at all.
    assert selected.headroom_bytes + rejected.excess_bytes == step


def test_overhead_and_margin_are_split_on_the_placement_that_includes_kv() -> None:
    """The heuristics must be weighted by the placement actually described.

    Subtracting the KV share from VRAM after weighting overhead and margin by a
    GPU-carries-everything basis would leave the two padding terms describing a
    placement that no longer exists.
    """

    result = _offload(blocks=8)

    gpu_basis = result.gpu_weight_bytes + result.gpu_kv_cache_bytes
    ram_basis = result.ram_weight_bytes + result.ram_kv_cache_bytes
    expected_gpu_overhead = math.ceil(
        result.overhead_bytes * (gpu_basis / (gpu_basis + ram_basis))
    )

    assert result.gpu_overhead_bytes == expected_gpu_overhead
    assert result.gpu_overhead_bytes + result.ram_overhead_bytes == (
        result.overhead_bytes
    )


def test_legacy_total_layers_alias_cannot_conflict_with_transformer_blocks() -> None:
    with pytest.raises(ValueError, match="legacy alias"):
        analyze_components(
            weights_bytes=5 * GIB,
            kv_cache_bytes=1 * GIB,
            overhead_bytes=1 * GIB,
            hardware=_hardware(ram=32 * GIB, vram=8 * GIB),
            total_transformer_blocks=28,
            total_layers=29,
        )


def test_unified_memory_is_not_treated_as_ram_plus_vram() -> None:
    result = analyze_components(
        weights_bytes=6 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        hardware=_hardware(ram=8 * GIB, unified=True),
    )

    assert result.memory_topology is HardwareMemoryTopology.UNIFIED_MEMORY
    assert result.mode is HardwareFitMode.CPU_RAM
    assert result.available_vram_bytes is None
