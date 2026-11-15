"""The GPU Jaull names first is the one it plans with, whatever the vendor or order.

Reported on Reddit (2026-11-15): Arch Linux, Mesa 26.2.7, Ryzen 7 9800X3D + Radeon AI
PRO R9700. Vulkan lists the APU first and names it after the CPU, so the guided hardware
screen showed "AMD Ryzen 7 9800X3D 8-Core Processor" with shared memory and never the
R9700, and `scan` said "no NVIDIA GPU detected". Planning already used the R9700.
"""

from __future__ import annotations

from rich.console import Console

from jaull.domain.hardware import (
    AcceleratorType,
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
    gpu_label,
    primary_accelerator,
)
from jaull.hardware.vulkan import accelerators_from_summary
from jaull.presentation.hardware_report import render_hardware
from jaull.runtime.backend_selection import select_runtime_backend
from jaull.tui.screens.hardware_analysis import _summary_rows

GiB = 1024**3

# The two devices from the user's `vulkaninfo --summary`, in their order.
REDDIT_SUMMARY = """
GPU0:
    apiVersion         = 1.4.354
    driverVersion      = 26.2.7
    vendorID           = 0x1002
    deviceID           = 0x13c0
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = AMD Ryzen 7 9800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO)
    driverID           = DRIVER_ID_MESA_RADV
    driverName         = radv
    driverInfo         = Mesa 26.2.7-arch1.1
GPU1:
    apiVersion         = 1.4.354
    driverVersion      = 26.2.7
    vendorID           = 0x1002
    deviceID           = 0x7551
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = AMD Radeon AI PRO R9700 (RADV GFX1201)
    driverID           = DRIVER_ID_MESA_RADV
    driverName         = radv
    driverInfo         = Mesa 26.2.7-arch1.1
"""

R9700 = "AMD Radeon AI PRO R9700 (RADV GFX1201)"
APU = "AMD Ryzen 7 9800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO)"


def _reddit_profile() -> HardwareProfile:
    apu, card = accelerators_from_summary(REDDIT_SUMMARY, drm_vram_reader=None)
    card = card.model_copy(update={"dedicated_memory_bytes": int(31.9 * GiB)})
    return HardwareProfile(
        os="Linux 7.3.4-arch1-1", arch="x86_64",
        cpu=CpuInfo(model="AMD Ryzen 7 9800X3D 8-Core Processor"),
        memory=MemoryInfo(total_bytes=int(60.4 * GiB), available_bytes=int(58.1 * GiB)),
        accelerators=[apu, card],
    )


def test_the_dedicated_card_is_primary_even_when_listed_second() -> None:
    profile = _reddit_profile()

    primary = primary_accelerator(profile)

    assert primary is not None and primary.name == R9700
    assert primary.type is AcceleratorType.DEDICATED
    assert gpu_label(profile) == R9700
    # The name shown and the device planned with are the same one.
    selected = select_runtime_backend(profile).selected_accelerator
    assert selected is not None and selected.name == R9700


def test_the_guided_hardware_screen_shows_the_card_and_lists_the_apu() -> None:
    rows = dict(_summary_rows(_reddit_profile()))

    assert rows["Accelerator"] == R9700
    assert rows["VRAM"].startswith("31.9")
    assert rows["Also detected"] == f"{APU} (integrated)"


def test_scan_names_the_card_instead_of_saying_no_nvidia_gpu() -> None:
    console = Console(width=160, record=True)

    render_hardware(_reddit_profile(), console)

    text = console.export_text()
    assert "no NVIDIA GPU detected" not in text
    assert f"{R9700} (details below)" in text


def test_without_any_accelerator_the_label_says_cpu_only() -> None:
    profile = _reddit_profile().model_copy(update={"accelerators": []})

    assert primary_accelerator(profile) is None
    assert gpu_label(profile) == "none detected (CPU inference only)"
