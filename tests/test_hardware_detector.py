from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaull.hardware.detector import detect_hardware
from jaull.hardware.nvidia import detect_nvidia_gpus


@dataclass
class _MemStub:
    total: int
    free: int


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
    profile = detect_hardware(_FakeNvmlOk())
    assert len(profile.gpus) == 1
    assert profile.gpus[0].name == "NVIDIA GeForce RTX 2060"
