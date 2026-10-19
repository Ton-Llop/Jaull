"""Where Jaull itself is running, read from an already detected profile.

This is deliberately separate from the accelerator and from the backend: a GPU
that Windows can see is not evidence that WSL2 exposes it, and an operating
system never selects a backend by itself. The classification only names the
environment so evidence and explanations can say which one they describe.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    BackendAvailability,
    ComputeBackend,
    HardwareProfile,
)


class ExecutionEnvironment(StrEnum):
    WINDOWS_NATIVE = "windows"
    WSL2 = "wsl2"
    WSL1 = "wsl1"
    LINUX_NATIVE = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


_WSL2_MARKERS = ("wsl2", "microsoft-standard")
_WSL_MARKER = "microsoft"


def classify_execution_environment(hardware: HardwareProfile) -> ExecutionEnvironment:
    """Classify from ``HardwareProfile.os`` (``"<system> <release>"``).

    A WSL kernel release names Microsoft; that, and not the presence of any
    particular GPU, is what separates WSL from native Linux.
    """
    return classify_os_label(hardware.os)


def classify_os_label(os_label: str) -> ExecutionEnvironment:
    label = os_label.strip().casefold()
    if label.startswith("windows"):
        return ExecutionEnvironment.WINDOWS_NATIVE
    if label.startswith("darwin"):
        return ExecutionEnvironment.MACOS
    if label.startswith("linux"):
        if any(marker in label for marker in _WSL2_MARKERS):
            return ExecutionEnvironment.WSL2
        if _WSL_MARKER in label:
            return ExecutionEnvironment.WSL1
        return ExecutionEnvironment.LINUX_NATIVE
    return ExecutionEnvironment.UNKNOWN


def acceleration_environment_note(hardware: HardwareProfile) -> str | None:
    """Explain a WSL2 session that exposes no usable Vulkan accelerator.

    Neither outcome is a verdict on the machine: the Windows host may well
    accelerate the same card, but Jaull running inside WSL2 can only use what
    WSL2 exposes, and no path through the host is assumed.
    """
    return wsl2_acceleration_note(hardware.os, hardware.accelerators)


def wsl2_acceleration_note(
    os_label: str, accelerators: Sequence[AcceleratorProfile]
) -> str | None:
    """Same rule for callers that have detected accelerators but no full profile."""
    if classify_os_label(os_label) is not ExecutionEnvironment.WSL2:
        return None
    if _has_usable_vulkan_device(accelerators):
        return None
    return (
        "Running inside WSL2: no usable Vulkan accelerator is exposed here. "
        "This does not show that the Windows host cannot accelerate the GPU, and "
        "no Windows runtime is used from WSL2 unless an explicit, verified path is configured."
    )


def _has_usable_vulkan_device(accelerators: Sequence[AcceleratorProfile]) -> bool:
    return any(
        accelerator.type is not AcceleratorType.SOFTWARE
        and any(
            backend.backend is ComputeBackend.VULKAN
            and backend.availability is BackendAvailability.AVAILABLE
            and not backend.software_renderer
            for backend in accelerator.backends
        )
        for accelerator in accelerators
    )


__all__ = [
    "ExecutionEnvironment",
    "acceleration_environment_note",
    "classify_execution_environment",
    "classify_os_label",
    "wsl2_acceleration_note",
]
