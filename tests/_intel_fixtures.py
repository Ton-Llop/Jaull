"""A simulated Intel Arc Pro B60 machine, built through the real detection code.

Only the external probes are faked (``vulkaninfo``, the Windows adapter query and
NVML); the profile is produced by ``detect_hardware`` itself so the TUI is shown
what a real scan would assemble. Every sample here is synthetic: it follows the
documented output layouts and was **not** captured from a B60.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from jaull.domain.hardware import HardwareProfile
from jaull.hardware import detector, windows_adapters
from jaull.hardware.detector import detect_hardware

GIB = 1024**3
B60_NAME = "Intel(R) Arc(TM) Pro B60 Graphics"
IGPU_NAME = "Intel(R) Iris(R) Xe Graphics"

WINDOWS_OS = "Windows 11"
WSL2_OS = "Linux 5.15.153.1-microsoft-standard-WSL2"
LINUX_OS = "Linux 6.8.0-45-generic"

B60_SUMMARY = f"""
GPU0:
    apiVersion         = 1.4.311
    driverVersion      = 25.2.8
    vendorID           = 0x8086
    deviceID           = 0xe211
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = {B60_NAME}
    driverName         = Intel open-source Mesa driver
    driverInfo         = Mesa 25.2.8
"""

LLVMPIPE_SUMMARY = """
GPU0:
    apiVersion         = 1.3.230
    vendorID           = 0x10005
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe (LLVM 15.0.7, 256 bits)
    driverName         = llvmpipe
"""

B60_FULL = f"""
GPU0:
VkPhysicalDeviceProperties:
---------------------------
\tvendorID          = 0x8086
\tdeviceID          = 0xe211
\tdeviceName        = {B60_NAME}

VkPhysicalDeviceMemoryProperties:
=================================
\tmemoryHeapCount       = 2
\tmemoryHeaps[0]:
\t\tsize   = 25769803776 (0x600000000) (24.00 GiB)
\t\tbudget = 24696061952 (0x5c0000000) (23.00 GiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags:
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[1]:
\t\tsize   = 33474412544 (0x7ca3a1000) (31.17 GiB)
\t\tbudget = 30000000000 (0x6fc23ac00) (27.94 GiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags:
\t\t\tNone
\tmemoryTypeCount       = 4
"""

B60_WMI_ROW = {
    "Name": B60_NAME,
    "PNPDeviceID": "PCI\\VEN_8086&DEV_E211&SUBSYS_12345678&REV_00\\4&1234&0&0008",
    "DriverVersion": "32.0.101.6881",
    "AdapterCompatibility": "Intel Corporation",
}


class NoNvml:
    """NVML is not available on an Intel machine."""

    def nvmlInit(self) -> None:
        raise OSError("no NVIDIA driver")


def _completed(command: tuple[str, ...], stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout, "")


def vulkaninfo_for(summary: str, full: str = ""):  # type: ignore[no-untyped-def]
    def run(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        return _completed(command, summary if "--summary" in command else full)

    return run


def no_vulkaninfo(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
    raise FileNotFoundError("vulkaninfo")


def windows_lists(*rows: dict[str, str]):  # type: ignore[no-untyped-def]
    def run(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        return _completed(command, json.dumps(list(rows)))

    return run


@contextmanager
def machine(os_label: str) -> Iterator[None]:
    """Name the operating system a scan runs on, without touching the real host."""
    system = "Windows" if os_label.startswith("Windows") else "Linux"
    with (
        patch.object(detector, "_os_display_name", lambda: os_label),
        patch.object(windows_adapters.platform, "system", lambda: system),
    ):
        yield


def scan(
    os_label: str,
    *,
    summary: str = B60_SUMMARY,
    full: str = B60_FULL,
    windows: tuple[dict[str, str], ...] = (B60_WMI_ROW,),
    vulkaninfo_present: bool = True,
) -> HardwareProfile:
    """A real ``detect_hardware`` run with the Intel probes replaced by samples."""
    with machine(os_label):
        return detect_hardware(
            NoNvml(),
            vulkan_command_runner=(
                vulkaninfo_for(summary, full) if vulkaninfo_present else no_vulkaninfo
            ),
            windows_command_runner=windows_lists(*windows),
        )


class FakeNvml:
    """One NVIDIA card, for the regression that NVIDIA keeps its own flow."""

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

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        return object()

    def nvmlDeviceGetName(self, handle: object) -> str:
        return "NVIDIA GeForce RTX 4070"

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> object:
        class _Memory:
            total = 12 * GIB
            free = 11 * GIB

        return _Memory()


def scan_nvidia(os_label: str = LINUX_OS) -> HardwareProfile:
    """A real ``detect_hardware`` run on an NVIDIA machine with no Vulkan tooling."""
    with machine(os_label):
        return detect_hardware(
            FakeNvml(),
            vulkan_command_runner=no_vulkaninfo,
            windows_command_runner=windows_lists(),
        )
