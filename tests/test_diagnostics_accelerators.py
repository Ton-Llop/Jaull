from __future__ import annotations

from jaull.diagnostics import service
from jaull.domain.enums import DiagnosticStatus
from jaull.hardware.nvidia import NvidiaProbe
from jaull.hardware.vulkan import VulkanProbe, accelerators_from_summary


def test_accelerator_diagnostic_reports_llvmpipe_as_software_renderer(monkeypatch) -> None:
    llvmpipe = accelerators_from_summary(
        """
GPU0:
    apiVersion         = 1.3.230
    vendorID           = 0x10005
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe (LLVM 15.0.7, 256 bits)
    driverName         = llvmpipe
"""
    )
    monkeypatch.setattr(
        service,
        "detect_nvidia_gpus",
        lambda: NvidiaProbe([], [], [], None, None),
    )
    monkeypatch.setattr(
        service,
        "detect_vulkan_accelerators",
        lambda: VulkanProbe(accelerators=llvmpipe, warnings=[]),
    )

    result = service._check_accelerators()

    assert result.status is DiagnosticStatus.WARN
    assert "llvmpipe" in result.detail
    assert "vulkan: software renderer only" in result.detail
