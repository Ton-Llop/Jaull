from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from jaull.domain.hardware import (
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    BackendAvailabilityReason,
    ComputeBackend,
)
from jaull.hardware.detector import detect_hardware
from jaull.hardware.nvidia import NvidiaProcessMemorySampler, detect_nvidia_gpus


@dataclass
class _MemStub:
    total: int
    free: int


@dataclass
class _ProcStub:
    pid: int
    usedGpuMemory: int


class _FakeNvmlOk:
    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlSystemGetDriverVersion(self) -> str:
        return "555.42.02"

    def nvmlSystemGetCudaDriverVersion(self) -> int:
        return 12040

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> Any:
        return object()

    def nvmlDeviceGetName(self, handle: Any) -> str:
        return "NVIDIA GeForce RTX 2060"

    def nvmlDeviceGetMemoryInfo(self, handle: Any) -> _MemStub:
        return _MemStub(total=6 * 1024**3, free=5 * 1024**3)


class _FakeNvmlTwoIdentical(_FakeNvmlOk):
    def nvmlDeviceGetCount(self) -> int:
        return 2


class _FakeNvmlProcesses(_FakeNvmlOk):
    def __init__(self) -> None:
        self.samples = [
            [_ProcStub(pid=123, usedGpuMemory=256)],
            [_ProcStub(pid=123, usedGpuMemory=512)],
        ]

    def nvmlDeviceGetComputeRunningProcesses(self, handle: Any) -> list[_ProcStub]:
        del handle
        if not self.samples:
            return []
        return self.samples.pop(0)


class _FakeNvmlProcessQueryError(_FakeNvmlOk):
    def nvmlDeviceGetComputeRunningProcesses(self, handle: Any) -> list[_ProcStub]:
        del handle
        raise RuntimeError("process query failed")


class _FakeNvmlNoDriver:
    def nvmlInit(self) -> None:
        raise RuntimeError("Driver not loaded")

    def nvmlShutdown(self) -> None:  # pragma: no cover - not reached
        return None

    def nvmlSystemGetDriverVersion(self) -> str:  # pragma: no cover
        return ""

    def nvmlSystemGetCudaDriverVersion(self) -> int:  # pragma: no cover
        return 0

    def nvmlDeviceGetCount(self) -> int:  # pragma: no cover
        return 0

    def nvmlDeviceGetHandleByIndex(self, index: int) -> Any:  # pragma: no cover
        return object()

    def nvmlDeviceGetName(self, handle: Any) -> str:  # pragma: no cover
        return ""

    def nvmlDeviceGetMemoryInfo(self, handle: Any) -> _MemStub:  # pragma: no cover
        return _MemStub(0, 0)


def test_detect_nvidia_gpus_returns_gpu_when_nvml_ok() -> None:
    probe = detect_nvidia_gpus(_FakeNvmlOk())
    assert len(probe.gpus) == 1
    gpu = probe.gpus[0]
    assert gpu.name == "NVIDIA GeForce RTX 2060"
    assert gpu.vram_total_bytes == 6 * 1024**3
    assert gpu.cuda_version == "12.4"
    assert gpu.driver_version == "555.42.02"
    assert len(probe.accelerators) == 1
    accelerator = probe.accelerators[0]
    assert accelerator.name == "NVIDIA GeForce RTX 2060"
    assert accelerator.vendor is AcceleratorVendor.NVIDIA
    assert accelerator.type is AcceleratorType.DEDICATED
    assert accelerator.dedicated_memory_bytes == 6 * 1024**3
    assert accelerator.available_memory_bytes == 5 * 1024**3
    cuda = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.CUDA
    )
    assert cuda.availability is BackendAvailability.AVAILABLE
    assert cuda.reason is BackendAvailabilityReason.PROBE_AVAILABLE
    assert cuda.api_version == "12.4"
    hip = next(
        backend
        for backend in accelerator.backends
        if backend.backend is ComputeBackend.HIP
    )
    assert hip.availability is BackendAvailability.UNKNOWN
    assert hip.reason is BackendAvailabilityReason.NOT_CHECKED
    assert probe.warnings == []


def test_detect_nvidia_gpus_returns_empty_when_driver_missing() -> None:
    probe = detect_nvidia_gpus(_FakeNvmlNoDriver())
    assert probe.gpus == []
    assert probe.warnings, "should surface a warning explaining the missing driver"


def test_detect_hardware_works_without_nvidia_gpu() -> None:
    profile = detect_hardware(_FakeNvmlNoDriver())
    assert profile.gpus == []
    assert profile.memory.total_bytes > 0
    assert profile.cpu.logical_cores is None or profile.cpu.logical_cores >= 1
    assert any("NVIDIA" in w for w in profile.warnings)


def test_detect_hardware_populates_gpu_when_available() -> None:
    profile = detect_hardware(
        _FakeNvmlOk(), vulkan_command_runner=_missing_vulkaninfo
    )
    assert len(profile.gpus) == 1
    assert profile.gpus[0].name == "NVIDIA GeForce RTX 2060"
    assert len(profile.accelerators) == 1
    assert profile.accelerators[0].vendor is AcceleratorVendor.NVIDIA


def test_detect_hardware_adds_amd_vulkan_accelerator() -> None:
    profile = detect_hardware(
        _FakeNvmlNoDriver(), vulkan_command_runner=_amd_vulkaninfo
    )

    assert profile.gpus == []
    assert len(profile.accelerators) == 1
    accelerator = profile.accelerators[0]
    assert accelerator.name == "AMD Radeon(TM) Graphics"
    assert accelerator.vendor is AcceleratorVendor.AMD
    assert accelerator.type is AcceleratorType.INTEGRATED
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
    assert cuda.reason is BackendAvailabilityReason.VENDOR_NOT_SUPPORTED
    assert vulkan.backend is ComputeBackend.VULKAN
    assert vulkan.availability is BackendAvailability.AVAILABLE
    assert hip.availability is BackendAvailability.UNKNOWN
    assert hip.reason is BackendAvailabilityReason.NOT_CHECKED


def test_detect_hardware_merges_nvidia_nvml_and_vulkan_backends() -> None:
    profile = detect_hardware(
        _FakeNvmlOk(), vulkan_command_runner=_nvidia_vulkaninfo
    )

    assert len(profile.gpus) == 1
    assert len(profile.accelerators) == 1
    accelerator = profile.accelerators[0]
    assert accelerator.name == "NVIDIA GeForce RTX 2060"
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
    assert cuda.availability is BackendAvailability.AVAILABLE
    assert cuda.api_version == "12.4"
    assert vulkan.availability is BackendAvailability.AVAILABLE


def test_detect_hardware_does_not_merge_identical_names_without_identity() -> None:
    profile = detect_hardware(
        _FakeNvmlTwoIdentical(), vulkan_command_runner=_nvidia_vulkaninfo
    )

    assert len(profile.gpus) == 2
    assert len(profile.accelerators) == 3


def test_detect_hardware_cpu_only_remains_valid() -> None:
    profile = detect_hardware(
        _FakeNvmlNoDriver(), vulkan_command_runner=_missing_vulkaninfo
    )

    assert profile.gpus == []
    assert profile.accelerators == []
    assert profile.memory.total_bytes > 0


def test_nvidia_process_sampler_returns_peak_for_matching_pid() -> None:
    sampler = NvidiaProcessMemorySampler(_FakeNvmlProcesses())
    try:
        assert sampler.sample_pid_bytes(123) == 256
        assert sampler.sample_pid_bytes(123) == 512
    finally:
        sampler.close()


def test_nvidia_process_sampler_returns_none_when_pid_not_found() -> None:
    sampler = NvidiaProcessMemorySampler(_FakeNvmlProcesses())
    try:
        assert sampler.sample_pid_bytes(999) is None
    finally:
        sampler.close()


def test_nvidia_process_sampler_returns_none_when_nvml_unavailable() -> None:
    sampler = NvidiaProcessMemorySampler(_FakeNvmlNoDriver())
    try:
        assert sampler.sample_pid_bytes(123) is None
    finally:
        sampler.close()


def test_nvidia_process_sampler_returns_none_on_process_query_error() -> None:
    sampler = NvidiaProcessMemorySampler(_FakeNvmlProcessQueryError())
    try:
        assert sampler.sample_pid_bytes(123) is None
    finally:
        sampler.close()


def _missing_vulkaninfo(
    command: tuple[str, ...], timeout: float
) -> subprocess.CompletedProcess[str]:
    del command, timeout
    raise FileNotFoundError


def _amd_vulkaninfo(
    command: tuple[str, ...], timeout: float
) -> subprocess.CompletedProcess[str]:
    del command, timeout
    return subprocess.CompletedProcess(
        args=["vulkaninfo", "--summary"],
        returncode=0,
        stdout="""
GPU0:
    apiVersion         = 1.3.280
    vendorID           = 0x1002
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = AMD Radeon(TM) Graphics
    driverName         = AMD proprietary driver
""",
        stderr="",
    )


def _nvidia_vulkaninfo(
    command: tuple[str, ...], timeout: float
) -> subprocess.CompletedProcess[str]:
    del command, timeout
    return subprocess.CompletedProcess(
        args=["vulkaninfo", "--summary"],
        returncode=0,
        stdout="""
GPU0:
    apiVersion         = 1.3.280
    vendorID           = 0x10de
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = NVIDIA RTX 2060
    driverName         = NVIDIA proprietary driver
""",
        stderr="",
    )
