from __future__ import annotations

from jaull.discovery import candidate_filter
from jaull.domain.estimation import HardwareFitMode
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
    available_accelerator_memory_bytes,
    planning_accelerator_memory_bytes,
)
from jaull.domain.inference import TargetDevice
from jaull.estimator.compatibility import assess
from jaull.estimator.hardware_fit import analyze_components
from jaull.workflow.orchestrator import _memory_budget

GIB = 1024**3


def _amd_hardware(*, available_vram: int | None) -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=24 * GIB),
        accelerators=[
            AcceleratorProfile(
                name="AMD Radeon RX 7600",
                vendor=AcceleratorVendor.AMD,
                type=AcceleratorType.DEDICATED,
                dedicated_memory_bytes=8 * GIB,
                available_memory_bytes=available_vram,
                backends=[
                    ComputeBackendInfo(
                        backend=ComputeBackend.VULKAN,
                        availability=BackendAvailability.AVAILABLE,
                    )
                ],
            )
        ],
    )


def test_discrete_accelerator_memory_is_available_to_planning_and_hfa() -> None:
    hardware = _amd_hardware(available_vram=6 * GIB)

    assert planning_accelerator_memory_bytes(hardware) == 8 * GIB
    assert available_accelerator_memory_bytes(hardware) == 6 * GIB
    assert candidate_filter._planning_vram_bytes(hardware) == 8 * GIB
    assert _memory_budget(hardware) == 8 * GIB

    fit = analyze_components(
        weights_bytes=4 * GIB,
        kv_cache_bytes=0,
        overhead_bytes=0,
        hardware=hardware,
    )

    assert fit.mode is HardwareFitMode.GPU_RESIDENT
    assert fit.available_vram_bytes == 6 * GIB

    compatibility = assess(4 * GIB, hardware, TargetDevice.GPU)

    assert compatibility.available_vram_bytes == 6 * GIB


def test_capacity_without_observed_availability_is_not_used_as_run_now_vram() -> None:
    hardware = _amd_hardware(available_vram=None)

    assert planning_accelerator_memory_bytes(hardware) == 8 * GIB
    assert available_accelerator_memory_bytes(hardware) is None

    fit = analyze_components(
        weights_bytes=4 * GIB,
        kv_cache_bytes=0,
        overhead_bytes=0,
        hardware=hardware,
    )

    assert fit.mode is HardwareFitMode.CPU_RAM
    assert fit.available_vram_bytes is None


def test_shared_accelerator_memory_is_not_a_second_vram_pool() -> None:
    hardware = _amd_hardware(available_vram=6 * GIB).model_copy(
        update={
            "accelerators": [
                AcceleratorProfile(
                    name="AMD integrated graphics",
                    vendor=AcceleratorVendor.AMD,
                    type=AcceleratorType.INTEGRATED,
                    dedicated_memory_bytes=6 * GIB,
                    available_memory_bytes=6 * GIB,
                    shared_memory=True,
                )
            ]
        }
    )

    assert planning_accelerator_memory_bytes(hardware) is None
    assert available_accelerator_memory_bytes(hardware) is None
