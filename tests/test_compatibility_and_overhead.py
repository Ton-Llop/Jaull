from __future__ import annotations

from jaull.domain.enums import DiagnosticStatus  # noqa: F401  (import kept for sanity)
from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import TargetDevice
from jaull.estimator import compatibility, overhead


def _profile(
    ram: int, vram: int | None = None
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
        if vram is not None
        else []
    )
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=ram * 2, available_bytes=ram),
        storage=[],
        gpus=gpus,
        warnings=[],
    )


GIB = 1024**3


def test_status_comfortable() -> None:
    result = compatibility.assess(1 * GIB, _profile(ram=8 * GIB, vram=8 * GIB), TargetDevice.GPU)
    assert result.status is CompatibilityStatus.COMFORTABLE


def test_status_compatible_edge() -> None:
    # 80% of VRAM => COMPATIBLE (between 75% and 90%)
    result = compatibility.assess(
        int(8 * GIB * 0.8), _profile(ram=8 * GIB, vram=8 * GIB), TargetDevice.GPU
    )
    assert result.status is CompatibilityStatus.COMPATIBLE


def test_status_tight() -> None:
    # 95% => TIGHT
    result = compatibility.assess(
        int(8 * GIB * 0.95), _profile(ram=8 * GIB, vram=8 * GIB), TargetDevice.GPU
    )
    assert result.status is CompatibilityStatus.TIGHT


def test_status_insufficient_when_over_capacity() -> None:
    result = compatibility.assess(
        20 * GIB, _profile(ram=4 * GIB, vram=None), TargetDevice.CPU
    )
    assert result.status is CompatibilityStatus.INSUFFICIENT


def test_status_offloading_required_in_auto() -> None:
    # Total > VRAM but fits VRAM + RAM
    result = compatibility.assess(
        10 * GIB, _profile(ram=8 * GIB, vram=6 * GIB), TargetDevice.AUTO
    )
    assert result.status is CompatibilityStatus.OFFLOADING_REQUIRED
    assert result.effective_device is TargetDevice.GPU


def test_status_unknown_when_total_missing() -> None:
    result = compatibility.assess(None, _profile(ram=8 * GIB), TargetDevice.AUTO)
    assert result.status is CompatibilityStatus.UNKNOWN


def test_gpu_requested_without_gpu_is_insufficient() -> None:
    result = compatibility.assess(1 * GIB, _profile(ram=8 * GIB), TargetDevice.GPU)
    assert result.status is CompatibilityStatus.INSUFFICIENT
    assert any("no nvidia gpu" in r.lower() for r in result.reasons)


def test_auto_uses_gpu_when_it_fits() -> None:
    result = compatibility.assess(2 * GIB, _profile(ram=16 * GIB, vram=8 * GIB), TargetDevice.AUTO)
    assert result.status is CompatibilityStatus.COMFORTABLE
    assert result.effective_device is TargetDevice.GPU


def test_auto_falls_back_to_cpu() -> None:
    result = compatibility.assess(2 * GIB, _profile(ram=16 * GIB, vram=None), TargetDevice.AUTO)
    assert result.status is CompatibilityStatus.COMFORTABLE
    assert result.effective_device is TargetDevice.CPU


def test_overhead_min_clamp() -> None:
    est = overhead.estimate_overhead(weights_bytes=0)
    # Base (512 MiB) already exceeds min (256 MiB), so we get base + 0.
    assert est.component.bytes == 512 * 1024 * 1024


def test_overhead_scales_with_weights() -> None:
    est = overhead.estimate_overhead(weights_bytes=10 * GIB)
    # base 512 MiB + 10% of 10 GiB = 512 MiB + 1 GiB
    expected = 512 * 1024 * 1024 + int(0.10 * 10 * GIB)
    assert est.component.bytes == expected


def test_overhead_unknown_weights() -> None:
    est = overhead.estimate_overhead(weights_bytes=None)
    assert est.component.bytes == 512 * 1024 * 1024
    assert "unknown" in est.component.explanation.lower()
