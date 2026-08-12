from __future__ import annotations

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    BackendAvailabilityReason,
    ComputeBackend,
    ComputeBackendInfo,
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.runtime import RuntimeBackendSelectionReason
from jaull.runtime.backend_selection import select_runtime_backend

GIB = 1024**3


def _hardware(*accelerators: AcceleratorProfile) -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=24 * GIB),
        accelerators=list(accelerators),
    )


def _backend(
    backend: ComputeBackend,
    availability: BackendAvailability,
    *,
    reason: BackendAvailabilityReason | None = None,
    source: str | None = "test",
    software_renderer: bool = False,
) -> ComputeBackendInfo:
    return ComputeBackendInfo(
        backend=backend,
        availability=availability,
        reason=reason,
        source=source,
        software_renderer=software_renderer,
    )


def _accelerator(
    *,
    name: str,
    vendor: AcceleratorVendor,
    accelerator_type: AcceleratorType,
    memory_gib: int | None = None,
    backends: list[ComputeBackendInfo],
) -> AcceleratorProfile:
    return AcceleratorProfile(
        name=name,
        vendor=vendor,
        type=accelerator_type,
        dedicated_memory_bytes=memory_gib * GIB if memory_gib is not None else None,
        shared_memory=accelerator_type is AcceleratorType.INTEGRATED,
        backends=backends,
        detection_sources=["test"],
    )


def test_nvidia_cuda_selected_when_cuda_available_and_vulkan_unknown() -> None:
    rtx = _accelerator(
        name="NVIDIA RTX 2060",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[
            _backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE),
            _backend(ComputeBackend.VULKAN, BackendAvailability.UNKNOWN),
        ],
    )

    selection = select_runtime_backend(_hardware(rtx))

    assert selection.selected_backend is ComputeBackend.CUDA
    assert selection.selected_accelerator is not None
    assert selection.selected_accelerator.name == "NVIDIA RTX 2060"
    assert selection.reason is RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE
    assert [candidate.backend for candidate in selection.alternatives] == [
        ComputeBackend.CPU
    ]


def test_nvidia_cuda_selected_with_vulkan_as_alternative() -> None:
    rtx = _accelerator(
        name="NVIDIA RTX 2060",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[
            _backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE),
            _backend(ComputeBackend.VULKAN, BackendAvailability.AVAILABLE),
        ],
    )

    selection = select_runtime_backend(_hardware(rtx))

    assert selection.selected_backend is ComputeBackend.CUDA
    assert [candidate.backend for candidate in selection.alternatives] == [
        ComputeBackend.VULKAN,
        ComputeBackend.CPU,
    ]


def test_nvidia_without_cuda_can_select_vulkan() -> None:
    rtx = _accelerator(
        name="NVIDIA RTX 2060",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[
            _backend(ComputeBackend.CUDA, BackendAvailability.UNAVAILABLE),
            _backend(ComputeBackend.VULKAN, BackendAvailability.AVAILABLE),
        ],
    )

    selection = select_runtime_backend(_hardware(rtx))

    assert selection.selected_backend is ComputeBackend.VULKAN
    assert selection.reason is RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE


def test_amd_vulkan_selected_when_hip_unknown() -> None:
    amd = _accelerator(
        name="AMD Radeon(TM) Graphics",
        vendor=AcceleratorVendor.AMD,
        accelerator_type=AcceleratorType.INTEGRATED,
        backends=[
            _backend(ComputeBackend.CUDA, BackendAvailability.UNAVAILABLE),
            _backend(ComputeBackend.HIP, BackendAvailability.UNKNOWN),
            _backend(ComputeBackend.VULKAN, BackendAvailability.AVAILABLE),
        ],
    )

    selection = select_runtime_backend(_hardware(amd))

    assert selection.selected_backend is ComputeBackend.VULKAN
    assert selection.selected_accelerator is not None
    assert selection.selected_accelerator.name == "AMD Radeon(TM) Graphics"
    assert [candidate.backend for candidate in selection.alternatives] == [
        ComputeBackend.CPU
    ]


def test_amd_hip_available_beats_vulkan() -> None:
    amd = _accelerator(
        name="AMD Radeon RX",
        vendor=AcceleratorVendor.AMD,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=8,
        backends=[
            _backend(ComputeBackend.HIP, BackendAvailability.AVAILABLE),
            _backend(ComputeBackend.VULKAN, BackendAvailability.AVAILABLE),
        ],
    )

    selection = select_runtime_backend(_hardware(amd))

    assert selection.selected_backend is ComputeBackend.HIP
    assert [candidate.backend for candidate in selection.alternatives] == [
        ComputeBackend.VULKAN,
        ComputeBackend.CPU,
    ]


def test_amd_without_available_backend_falls_back_to_cpu() -> None:
    amd = _accelerator(
        name="AMD Radeon(TM) Graphics",
        vendor=AcceleratorVendor.AMD,
        accelerator_type=AcceleratorType.INTEGRATED,
        backends=[
            _backend(ComputeBackend.HIP, BackendAvailability.UNKNOWN),
            _backend(ComputeBackend.VULKAN, BackendAvailability.UNAVAILABLE),
        ],
    )

    selection = select_runtime_backend(_hardware(amd))

    assert selection.selected_backend is ComputeBackend.CPU
    assert selection.reason is RuntimeBackendSelectionReason.NO_USABLE_ACCELERATOR
    assert selection.alternatives == []


def test_intel_vulkan_selected() -> None:
    intel = _accelerator(
        name="Intel Arc",
        vendor=AcceleratorVendor.INTEL,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=8,
        backends=[_backend(ComputeBackend.VULKAN, BackendAvailability.AVAILABLE)],
    )

    selection = select_runtime_backend(_hardware(intel))

    assert selection.selected_backend is ComputeBackend.VULKAN


def test_llvmpipe_software_renderer_falls_back_to_cpu() -> None:
    llvmpipe = _accelerator(
        name="llvmpipe",
        vendor=AcceleratorVendor.OTHER,
        accelerator_type=AcceleratorType.SOFTWARE,
        backends=[
            _backend(
                ComputeBackend.VULKAN,
                BackendAvailability.UNAVAILABLE,
                reason=BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY,
                software_renderer=True,
            )
        ],
    )

    selection = select_runtime_backend(_hardware(llvmpipe))

    assert selection.selected_backend is ComputeBackend.CPU
    assert selection.reason is RuntimeBackendSelectionReason.SOFTWARE_RENDERER_IGNORED


def test_cpu_only_falls_back_to_cpu() -> None:
    selection = select_runtime_backend(_hardware())

    assert selection.selected_backend is ComputeBackend.CPU
    assert selection.reason is RuntimeBackendSelectionReason.NO_USABLE_ACCELERATOR


def test_unknown_backend_is_not_selected() -> None:
    unknown = _accelerator(
        name="Unknown GPU",
        vendor=AcceleratorVendor.UNKNOWN,
        accelerator_type=AcceleratorType.UNKNOWN,
        backends=[_backend(ComputeBackend.VULKAN, BackendAvailability.UNKNOWN)],
    )

    selection = select_runtime_backend(_hardware(unknown))

    assert selection.selected_backend is ComputeBackend.CPU


def test_dedicated_cuda_beats_integrated_vulkan() -> None:
    amd = _accelerator(
        name="AMD Radeon(TM) Graphics",
        vendor=AcceleratorVendor.AMD,
        accelerator_type=AcceleratorType.INTEGRATED,
        backends=[_backend(ComputeBackend.VULKAN, BackendAvailability.AVAILABLE)],
    )
    nvidia = _accelerator(
        name="NVIDIA RTX 2060",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[_backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE)],
    )

    selection = select_runtime_backend(_hardware(amd, nvidia))

    assert selection.selected_backend is ComputeBackend.CUDA
    assert selection.selected_accelerator is not None
    assert selection.selected_accelerator.name == "NVIDIA RTX 2060"


def test_two_dedicated_cuda_gpus_prefer_more_memory() -> None:
    small = _accelerator(
        name="GPU A",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[_backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE)],
    )
    large = _accelerator(
        name="GPU B",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=12,
        backends=[_backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE)],
    )

    selection = select_runtime_backend(_hardware(small, large))

    assert selection.selected_accelerator is not None
    assert selection.selected_accelerator.name == "GPU B"


def test_identical_gpus_keep_discovery_order_when_equivalent() -> None:
    first = _accelerator(
        name="NVIDIA RTX 2060",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[_backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE)],
    )
    second = _accelerator(
        name="NVIDIA RTX 2060",
        vendor=AcceleratorVendor.NVIDIA,
        accelerator_type=AcceleratorType.DEDICATED,
        memory_gib=6,
        backends=[_backend(ComputeBackend.CUDA, BackendAvailability.AVAILABLE)],
    )

    selection = select_runtime_backend(_hardware(first, second))

    assert selection.selected_accelerator is not None
    assert selection.selected_accelerator.name == "NVIDIA RTX 2060"
    assert selection.alternatives[0].accelerator is not None
    assert selection.alternatives[0].accelerator.name == "NVIDIA RTX 2060"
