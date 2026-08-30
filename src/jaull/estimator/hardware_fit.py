"""Placement-aware hardware fit analysis.

This module consumes memory estimates produced elsewhere. It does not estimate
weights, KV cache, overhead, or ranking preference; it only checks whether the
estimated memory can be placed on the detected hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jaull.domain.estimation import (
    HardwareFitMode,
    HardwareFitOffloadCandidate,
    HardwareFitOffloadDiagnostics,
    HardwareFitPlacementMethod,
    HardwareFitResult,
    HardwareMemoryTopology,
    MemoryEstimate,
)
from jaull.domain.hardware import HardwareProfile


@dataclass(frozen=True)
class _OffloadPlacement:
    total_transformer_blocks: int | None
    gpu_transformer_blocks: int | None
    gpu_required_bytes: int
    ram_required_bytes: int
    gpu_weight_bytes: int
    ram_weight_bytes: int
    gpu_overhead_bytes: int
    ram_overhead_bytes: int
    gpu_safety_margin_bytes: int
    ram_safety_margin_bytes: int


def analyze_estimate(
    estimate: MemoryEstimate,
    hardware: HardwareProfile,
) -> HardwareFitResult | None:
    """Analyze placement for an already-built memory estimate."""

    weights_bytes = estimate.weights.component.bytes
    kv_cache_bytes = estimate.kv_cache.component.bytes
    overhead_bytes = estimate.runtime_overhead.component.bytes
    if weights_bytes is None or kv_cache_bytes is None or overhead_bytes is None:
        return None
    safety_margin_bytes = (
        estimate.safety_margin.bytes if estimate.safety_margin is not None else 0
    )
    return analyze_components(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=estimate.device_reserve.bytes or 0,
        safety_margin_bytes=safety_margin_bytes or 0,
        hardware=hardware,
        total_transformer_blocks=estimate.kv_cache.layers,
    )


def analyze_components(
    *,
    weights_bytes: int,
    kv_cache_bytes: int,
    overhead_bytes: int,
    hardware: HardwareProfile,
    device_reserve_bytes: int = 0,
    safety_margin_bytes: int = 0,
    total_transformer_blocks: int | None = None,
    total_layers: int | None = None,
) -> HardwareFitResult:
    """Classify model placement without collapsing RAM and VRAM into one pool."""

    if (
        total_transformer_blocks is not None
        and total_layers is not None
        and total_transformer_blocks != total_layers
    ):
        raise ValueError(
            "total_layers is a legacy alias for total_transformer_blocks; "
            "both values were provided and they differ."
        )
    if total_transformer_blocks is None:
        total_transformer_blocks = total_layers
    topology = _memory_topology(hardware)
    available_vram = _available_vram(hardware)
    available_ram = hardware.memory.available_bytes

    if topology is HardwareMemoryTopology.UNIFIED_MEMORY:
        return _analyze_unified_memory(
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            overhead_bytes=overhead_bytes,
            device_reserve_bytes=device_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            available_ram=available_ram,
            total_transformer_blocks=total_transformer_blocks,
        )

    gpu_fixed_bytes = kv_cache_bytes + overhead_bytes + device_reserve_bytes
    gpu_required_if_resident = gpu_fixed_bytes + weights_bytes + safety_margin_bytes

    if available_vram is not None and gpu_required_if_resident <= available_vram:
        return HardwareFitResult(
            mode=HardwareFitMode.GPU_RESIDENT,
            memory_topology=topology,
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            overhead_bytes=overhead_bytes,
            device_reserve_bytes=device_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            available_vram_bytes=available_vram,
            available_ram_bytes=available_ram,
            gpu_required_bytes=gpu_required_if_resident,
            gpu_weight_bytes=weights_bytes,
            gpu_overhead_bytes=overhead_bytes,
            gpu_safety_margin_bytes=safety_margin_bytes,
            ram_required_bytes=0,
            ram_weight_bytes=0,
            gpu_transformer_blocks=total_transformer_blocks,
            total_transformer_blocks=total_transformer_blocks,
            placement_method=(
                HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
                if _valid_transformer_block_count(total_transformer_blocks)
                else HardwareFitPlacementMethod.ESTIMATED_BYTES
            ),
            reason="Weights, KV cache, overhead, reserve, and margin fit in VRAM.",
        )

    if available_vram is not None:
        offload = _try_gpu_offload(
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            overhead_bytes=overhead_bytes,
            device_reserve_bytes=device_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            available_vram=available_vram,
            available_ram=available_ram,
            total_transformer_blocks=total_transformer_blocks,
            topology=topology,
        )
        if offload is not None:
            return offload

    cpu_required = weights_bytes + kv_cache_bytes + overhead_bytes + safety_margin_bytes
    if cpu_required <= available_ram:
        prefix = (
            "No viable GPU placement found"
            if available_vram is not None
            else "No GPU detected"
        )
        return HardwareFitResult(
            mode=HardwareFitMode.CPU_RAM,
            memory_topology=topology,
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            overhead_bytes=overhead_bytes,
            device_reserve_bytes=device_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            available_vram_bytes=available_vram,
            available_ram_bytes=available_ram,
            gpu_required_bytes=(
                gpu_required_if_resident if available_vram is not None else None
            ),
            gpu_weight_bytes=0,
            ram_required_bytes=cpu_required,
            ram_weight_bytes=weights_bytes,
            ram_overhead_bytes=overhead_bytes,
            ram_safety_margin_bytes=safety_margin_bytes,
            gpu_transformer_blocks=_gpu_transformer_blocks_when_unused(available_vram),
            total_transformer_blocks=total_transformer_blocks,
            placement_method=HardwareFitPlacementMethod.NONE,
            reason=f"{prefix}; full model fits in system RAM.",
        )

    return HardwareFitResult(
        mode=HardwareFitMode.TOO_LARGE,
        memory_topology=topology,
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=device_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        available_vram_bytes=available_vram,
        available_ram_bytes=available_ram,
        gpu_required_bytes=(
            gpu_required_if_resident if available_vram is not None else None
        ),
        gpu_weight_bytes=0,
        ram_required_bytes=cpu_required,
        ram_weight_bytes=weights_bytes,
        ram_overhead_bytes=overhead_bytes,
        ram_safety_margin_bytes=safety_margin_bytes,
        gpu_transformer_blocks=_gpu_transformer_blocks_when_unused(available_vram),
        total_transformer_blocks=total_transformer_blocks,
        placement_method=HardwareFitPlacementMethod.NONE,
        reason=(
            "No supported placement fits: VRAM cannot hold a viable GPU placement "
            "and system RAM is insufficient for CPU execution."
        ),
    )


def _try_gpu_offload(
    *,
    weights_bytes: int,
    kv_cache_bytes: int,
    overhead_bytes: int,
    device_reserve_bytes: int,
    safety_margin_bytes: int,
    available_vram: int,
    available_ram: int,
    total_transformer_blocks: int | None,
    topology: HardwareMemoryTopology,
) -> HardwareFitResult | None:
    gpu_fixed_bytes = kv_cache_bytes + device_reserve_bytes
    available_for_gpu_weights = available_vram - gpu_fixed_bytes
    if available_for_gpu_weights <= 0:
        return None

    method: HardwareFitPlacementMethod
    if _valid_transformer_block_count(total_transformer_blocks):
        assert total_transformer_blocks is not None
        estimated_bytes_per_transformer_block = math.ceil(
            weights_bytes / total_transformer_blocks
        )
        if estimated_bytes_per_transformer_block <= 0:
            return None
        max_gpu_transformer_blocks = min(
            total_transformer_blocks,
            available_for_gpu_weights // estimated_bytes_per_transformer_block,
        )
        search_ceiling_transformer_blocks = int(max_gpu_transformer_blocks)
        last_rejected: HardwareFitOffloadCandidate | None = None
        for gpu_transformer_blocks in range(
            search_ceiling_transformer_blocks, 0, -1
        ):
            gpu_weight_bytes = min(
                weights_bytes,
                gpu_transformer_blocks * estimated_bytes_per_transformer_block,
            )
            placement = _calculate_offload_placement(
                weights_bytes=weights_bytes,
                kv_cache_bytes=kv_cache_bytes,
                overhead_bytes=overhead_bytes,
                device_reserve_bytes=device_reserve_bytes,
                safety_margin_bytes=safety_margin_bytes,
                total_transformer_blocks=total_transformer_blocks,
                gpu_transformer_blocks=gpu_transformer_blocks,
                gpu_weight_bytes=gpu_weight_bytes,
            )
            if placement is None:
                continue
            if (
                placement.gpu_required_bytes <= available_vram
                and placement.ram_required_bytes <= available_ram
            ):
                result = _offload_result_from_placement(
                    weights_bytes=weights_bytes,
                    kv_cache_bytes=kv_cache_bytes,
                    overhead_bytes=overhead_bytes,
                    device_reserve_bytes=device_reserve_bytes,
                    safety_margin_bytes=safety_margin_bytes,
                    available_vram=available_vram,
                    available_ram=available_ram,
                    placement=placement,
                    placement_method=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
                    topology=topology,
                    warnings=[],
                    offload_diagnostics=HardwareFitOffloadDiagnostics(
                        search_ceiling_transformer_blocks=(
                            search_ceiling_transformer_blocks
                        ),
                        selected=_offload_candidate(
                            placement,
                            kv_cache_bytes=kv_cache_bytes,
                            device_reserve_bytes=device_reserve_bytes,
                            available_vram=available_vram,
                        ),
                        first_rejected_higher=last_rejected,
                    ),
                )
                return result
            last_rejected = _offload_candidate(
                placement,
                kv_cache_bytes=kv_cache_bytes,
                device_reserve_bytes=device_reserve_bytes,
                available_vram=available_vram,
            )
        return None

    method = HardwareFitPlacementMethod.ESTIMATED_BYTES
    warnings = [
        "Transformer block count unavailable; GPU offload placement is "
        "estimated by bytes."
    ]
    low = 1
    high = min(weights_bytes - 1, available_for_gpu_weights)
    best: HardwareFitResult | None = None
    while low <= high:
        gpu_weight_bytes = (low + high) // 2
        byte_result = _build_offload_result(
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            overhead_bytes=overhead_bytes,
            device_reserve_bytes=device_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            available_vram=available_vram,
            available_ram=available_ram,
            total_transformer_blocks=total_transformer_blocks,
            gpu_transformer_blocks=None,
            gpu_weight_bytes=gpu_weight_bytes,
            placement_method=method,
            topology=topology,
            warnings=warnings,
        )
        if (
            byte_result is not None
            and byte_result.gpu_required_bytes is not None
            and byte_result.gpu_required_bytes <= available_vram
        ):
            best = byte_result
            low = gpu_weight_bytes + 1
        else:
            high = gpu_weight_bytes - 1
    if best is None or best.ram_required_bytes is None:
        return None
    if best.ram_required_bytes > available_ram:
        return None
    return best


def _build_offload_result(
    *,
    weights_bytes: int,
    kv_cache_bytes: int,
    overhead_bytes: int,
    device_reserve_bytes: int,
    safety_margin_bytes: int,
    available_vram: int,
    available_ram: int,
    total_transformer_blocks: int | None,
    gpu_transformer_blocks: int | None,
    gpu_weight_bytes: int,
    placement_method: HardwareFitPlacementMethod,
    topology: HardwareMemoryTopology,
    warnings: list[str],
) -> HardwareFitResult | None:
    placement = _calculate_offload_placement(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=device_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        total_transformer_blocks=total_transformer_blocks,
        gpu_transformer_blocks=gpu_transformer_blocks,
        gpu_weight_bytes=gpu_weight_bytes,
    )
    if placement is None:
        return None
    if (
        placement.gpu_required_bytes > available_vram
        or placement.ram_required_bytes > available_ram
    ):
        return None
    return _offload_result_from_placement(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=device_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        available_vram=available_vram,
        available_ram=available_ram,
        placement=placement,
        placement_method=placement_method,
        topology=topology,
        warnings=warnings,
    )


def _calculate_offload_placement(
    *,
    weights_bytes: int,
    kv_cache_bytes: int,
    overhead_bytes: int,
    device_reserve_bytes: int,
    safety_margin_bytes: int,
    total_transformer_blocks: int | None,
    gpu_transformer_blocks: int | None,
    gpu_weight_bytes: int,
) -> _OffloadPlacement | None:
    if gpu_weight_bytes <= 0 or gpu_weight_bytes >= weights_bytes:
        return None
    ram_weight_bytes = weights_bytes - gpu_weight_bytes

    gpu_physical_bytes = gpu_weight_bytes + kv_cache_bytes + device_reserve_bytes
    ram_physical_bytes = ram_weight_bytes
    gpu_overhead_bytes, ram_overhead_bytes = _split_heuristic_bytes(
        overhead_bytes,
        gpu_basis_bytes=gpu_weight_bytes + kv_cache_bytes,
        ram_basis_bytes=ram_weight_bytes,
    )
    gpu_margin_basis = gpu_physical_bytes + gpu_overhead_bytes
    ram_margin_basis = ram_physical_bytes + ram_overhead_bytes
    gpu_safety_margin_bytes, ram_safety_margin_bytes = _split_heuristic_bytes(
        safety_margin_bytes,
        gpu_basis_bytes=gpu_margin_basis,
        ram_basis_bytes=ram_margin_basis,
    )

    gpu_required = gpu_physical_bytes + gpu_overhead_bytes + gpu_safety_margin_bytes
    ram_required = ram_physical_bytes + ram_overhead_bytes + ram_safety_margin_bytes

    return _OffloadPlacement(
        total_transformer_blocks=total_transformer_blocks,
        gpu_transformer_blocks=gpu_transformer_blocks,
        gpu_required_bytes=gpu_required,
        ram_required_bytes=ram_required,
        gpu_weight_bytes=gpu_weight_bytes,
        ram_weight_bytes=ram_weight_bytes,
        gpu_overhead_bytes=gpu_overhead_bytes,
        ram_overhead_bytes=ram_overhead_bytes,
        gpu_safety_margin_bytes=gpu_safety_margin_bytes,
        ram_safety_margin_bytes=ram_safety_margin_bytes,
    )


def _offload_result_from_placement(
    *,
    weights_bytes: int,
    kv_cache_bytes: int,
    overhead_bytes: int,
    device_reserve_bytes: int,
    safety_margin_bytes: int,
    available_vram: int,
    available_ram: int,
    placement: _OffloadPlacement,
    placement_method: HardwareFitPlacementMethod,
    topology: HardwareMemoryTopology,
    warnings: list[str],
    offload_diagnostics: HardwareFitOffloadDiagnostics | None = None,
) -> HardwareFitResult:
    return HardwareFitResult(
        mode=HardwareFitMode.GPU_OFFLOAD,
        memory_topology=topology,
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=device_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        available_vram_bytes=available_vram,
        available_ram_bytes=available_ram,
        gpu_required_bytes=placement.gpu_required_bytes,
        gpu_weight_bytes=placement.gpu_weight_bytes,
        gpu_overhead_bytes=placement.gpu_overhead_bytes,
        gpu_safety_margin_bytes=placement.gpu_safety_margin_bytes,
        ram_required_bytes=placement.ram_required_bytes,
        ram_weight_bytes=placement.ram_weight_bytes,
        ram_overhead_bytes=placement.ram_overhead_bytes,
        ram_safety_margin_bytes=placement.ram_safety_margin_bytes,
        gpu_transformer_blocks=placement.gpu_transformer_blocks,
        total_transformer_blocks=placement.total_transformer_blocks,
        placement_method=placement_method,
        offload_diagnostics=offload_diagnostics,
        reason=(
            "Model does not fit fully in VRAM, but a valid GPU/RAM weight "
            "placement fits without treating RAM and VRAM as one pool."
        ),
        warnings=warnings,
    )


def _offload_candidate(
    placement: _OffloadPlacement,
    *,
    kv_cache_bytes: int,
    device_reserve_bytes: int,
    available_vram: int,
) -> HardwareFitOffloadCandidate:
    assert placement.gpu_transformer_blocks is not None
    return HardwareFitOffloadCandidate(
        gpu_transformer_blocks=placement.gpu_transformer_blocks,
        gpu_required_bytes=placement.gpu_required_bytes,
        ram_required_bytes=placement.ram_required_bytes,
        available_vram_bytes=available_vram,
        excess_bytes=max(0, placement.gpu_required_bytes - available_vram),
        headroom_bytes=max(0, available_vram - placement.gpu_required_bytes),
        gpu_weight_bytes=placement.gpu_weight_bytes,
        ram_weight_bytes=placement.ram_weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        device_reserve_bytes=device_reserve_bytes,
        gpu_overhead_bytes=placement.gpu_overhead_bytes,
        gpu_safety_margin_bytes=placement.gpu_safety_margin_bytes,
    )


def _analyze_unified_memory(
    *,
    weights_bytes: int,
    kv_cache_bytes: int,
    overhead_bytes: int,
    device_reserve_bytes: int,
    safety_margin_bytes: int,
    available_ram: int,
    total_transformer_blocks: int | None,
) -> HardwareFitResult:
    """Fit against the single shared pool, reserve included.

    A device reserve is memory held back for the accelerator. On discrete
    hardware it is VRAM, which is why a ``CPU_RAM`` placement there does not
    pay for it. Here the accelerator draws from the same pool as the CPU, so
    the reserve genuinely consumes the budget and is counted. The caller's
    value is never silently discarded: whatever it passes shows up in
    ``ram_required_bytes`` and is echoed back in ``device_reserve_bytes``.
    """

    required = (
        weights_bytes
        + kv_cache_bytes
        + overhead_bytes
        + device_reserve_bytes
        + safety_margin_bytes
    )
    if required <= available_ram:
        return HardwareFitResult(
            mode=HardwareFitMode.CPU_RAM,
            memory_topology=HardwareMemoryTopology.UNIFIED_MEMORY,
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            overhead_bytes=overhead_bytes,
            device_reserve_bytes=device_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            available_ram_bytes=available_ram,
            ram_required_bytes=required,
            ram_weight_bytes=weights_bytes,
            ram_overhead_bytes=overhead_bytes,
            ram_safety_margin_bytes=safety_margin_bytes,
            gpu_transformer_blocks=None,
            total_transformer_blocks=total_transformer_blocks,
            placement_method=HardwareFitPlacementMethod.NONE,
            reason=(
                "Unified-memory hardware uses one shared pool; fit was checked "
                "against available system memory without summing VRAM and RAM. "
                "The device reserve is charged to that shared pool."
            ),
        )
    return HardwareFitResult(
        mode=HardwareFitMode.TOO_LARGE,
        memory_topology=HardwareMemoryTopology.UNIFIED_MEMORY,
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=device_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        available_ram_bytes=available_ram,
        ram_required_bytes=required,
        ram_weight_bytes=weights_bytes,
        ram_overhead_bytes=overhead_bytes,
        ram_safety_margin_bytes=safety_margin_bytes,
        gpu_transformer_blocks=None,
        total_transformer_blocks=total_transformer_blocks,
        placement_method=HardwareFitPlacementMethod.NONE,
        reason="Unified-memory pool is insufficient for the estimated requirement.",
    )


def _memory_topology(hardware: HardwareProfile) -> HardwareMemoryTopology:
    if any(accelerator.shared_memory for accelerator in hardware.accelerators):
        return HardwareMemoryTopology.UNIFIED_MEMORY
    return HardwareMemoryTopology.DISCRETE_MEMORY


def _split_heuristic_bytes(
    total_bytes: int,
    *,
    gpu_basis_bytes: int,
    ram_basis_bytes: int,
) -> tuple[int, int]:
    if total_bytes <= 0:
        return 0, 0
    if gpu_basis_bytes <= 0:
        return 0, total_bytes
    if ram_basis_bytes <= 0:
        return total_bytes, 0

    basis_total = gpu_basis_bytes + ram_basis_bytes
    gpu_bytes = math.ceil(total_bytes * (gpu_basis_bytes / basis_total))
    return gpu_bytes, total_bytes - gpu_bytes


def _available_vram(hardware: HardwareProfile) -> int | None:
    if not hardware.gpus:
        return None
    return max(gpu.vram_available_bytes for gpu in hardware.gpus)


def _valid_transformer_block_count(total_transformer_blocks: int | None) -> bool:
    return total_transformer_blocks is not None and total_transformer_blocks > 0


def _gpu_transformer_blocks_when_unused(available_vram: int | None) -> int | None:
    """How many transformer blocks sit on a GPU this placement does not use.

    ``0`` and ``None`` are not interchangeable here, and consumers downstream
    will read them as different facts: ``0`` means a GPU exists and none of the
    transformer blocks were placed on it, while ``None`` means the question
    does not apply because there is no GPU at all.
    """

    return 0 if available_vram is not None else None


__all__ = ["analyze_components", "analyze_estimate"]
