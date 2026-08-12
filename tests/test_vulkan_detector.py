from __future__ import annotations

import subprocess

from jaull.domain.hardware import (
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    BackendAvailabilityReason,
    ComputeBackend,
)
from jaull.hardware.vulkan import (
    accelerators_from_summary,
    detect_vulkan_accelerators,
)

AMD_VULKAN_SUMMARY = """
GPU0:
    apiVersion         = 1.3.280
    vendorID           = 0x1002
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = AMD Radeon(TM) Graphics
    driverName         = AMD proprietary driver
"""


LLVMPIPE_WSL_SUMMARY = """
GPU0:
    apiVersion         = 1.3.230
    vendorID           = 0x10005
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe (LLVM 15.0.7, 256 bits)
    driverName         = llvmpipe
"""


def test_amd_integrated_vulkan_device_is_available() -> None:
    accelerators = accelerators_from_summary(AMD_VULKAN_SUMMARY)

    assert len(accelerators) == 1
    accelerator = accelerators[0]
    assert accelerator.name == "AMD Radeon(TM) Graphics"
    assert accelerator.vendor is AcceleratorVendor.AMD
    assert accelerator.vendor_id == "0x1002"
    assert accelerator.type is AcceleratorType.INTEGRATED
    assert accelerator.shared_memory is True
    assert accelerator.detection_sources == ["vulkaninfo"]

    cuda = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.CUDA
    )
    vulkan = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.VULKAN
    )
    hip = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.HIP
    )
    assert cuda.availability is BackendAvailability.UNAVAILABLE
    assert cuda.reason is BackendAvailabilityReason.VENDOR_NOT_SUPPORTED
    assert vulkan.backend is ComputeBackend.VULKAN
    assert vulkan.availability is BackendAvailability.AVAILABLE
    assert vulkan.reason is BackendAvailabilityReason.PROBE_AVAILABLE
    assert vulkan.source == "vulkaninfo"
    assert vulkan.software_renderer is False
    assert vulkan.driver_name == "AMD proprietary driver"
    assert hip.availability is BackendAvailability.UNKNOWN
    assert hip.reason is BackendAvailabilityReason.NOT_CHECKED


def test_llvmpipe_is_software_renderer_not_gpu_acceleration() -> None:
    accelerators = accelerators_from_summary(LLVMPIPE_WSL_SUMMARY)

    assert len(accelerators) == 1
    accelerator = accelerators[0]
    assert accelerator.type is AcceleratorType.SOFTWARE
    assert accelerator.shared_memory is True

    cuda = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.CUDA
    )
    vulkan = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.VULKAN
    )
    hip = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.HIP
    )
    assert cuda.availability is BackendAvailability.UNAVAILABLE
    assert cuda.reason is BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY
    assert vulkan.backend is ComputeBackend.VULKAN
    assert vulkan.availability is BackendAvailability.UNAVAILABLE
    assert vulkan.reason is BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY
    assert vulkan.software_renderer is True
    assert "software renderer" in (vulkan.detail or "")
    assert hip.availability is BackendAvailability.UNAVAILABLE
    assert hip.reason is BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY


def test_vulkan_command_missing_does_not_fail_detection() -> None:
    def _missing(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        del command, timeout
        raise FileNotFoundError

    probe = detect_vulkan_accelerators(_missing)

    assert probe.accelerators == []
    assert any("vulkaninfo is not installed" in warning for warning in probe.warnings)


def test_malformed_vulkan_output_returns_unknown_without_crashing() -> None:
    def _malformed(
        command: tuple[str, ...], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        del command, timeout
        return subprocess.CompletedProcess(
            args=["vulkaninfo", "--summary"],
            returncode=0,
            stdout="this is not a vulkan summary",
            stderr="",
        )

    probe = detect_vulkan_accelerators(_malformed)

    assert probe.accelerators == []
    assert any("No Vulkan devices" in warning for warning in probe.warnings)
