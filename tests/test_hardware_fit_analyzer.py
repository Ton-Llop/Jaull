from __future__ import annotations

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
from jaull.estimator.hardware_fit import analyze_components

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
    result = analyze_components(
        weights_bytes=4 * GIB,
        kv_cache_bytes=2 * GIB,
        overhead_bytes=2 * GIB,
        hardware=_hardware(ram=12 * GIB, vram=3 * GIB),
        total_transformer_blocks=8,
    )

    assert result.mode is HardwareFitMode.CPU_RAM
    assert result.gpu_weight_bytes == 0
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
    assert diagnostics.selected.headroom_bytes == 0
    assert diagnostics.selected.excess_bytes == 0


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
    assert selected.headroom_bytes == 95_186_565

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
        + rejected.kv_cache_bytes
        + rejected.device_reserve_bytes
        + rejected.gpu_overhead_bytes
        + rejected.gpu_safety_margin_bytes
    )
    assert rejected.gpu_required_bytes == 5_111_775_368


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
