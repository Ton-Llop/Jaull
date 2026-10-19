"""Which display adapters Windows itself lists.

This is hardware *identification* only. It says that an adapter exists and what
vendor and PCI device it is; it never says which compute backend can use it, nor
how much memory it has. ``Win32_VideoController.AdapterRAM`` is a 32-bit field
that saturates near 4 GiB on modern cards, so it is read by nobody here.

Vulkan availability stays ``UNKNOWN``: an Intel Arc visible to Windows tells us
nothing about whether a Vulkan loader, a driver or a llama.cpp build can use it.
Backends are established by their own probes (``vulkaninfo``, ``--list-devices``)
and this entry then merges with whatever they report.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    BackendAvailabilityReason,
    ComputeBackend,
    ComputeBackendInfo,
)

logger = logging.getLogger(__name__)

WindowsCommandRunner = Callable[[tuple[str, ...], float], subprocess.CompletedProcess[str]]

_POWERSHELL_QUERY = (
    "Get-CimInstance Win32_VideoController | "
    "Select-Object Name,PNPDeviceID,DriverVersion,AdapterCompatibility | "
    "ConvertTo-Json -Compress"
)
_WINDOWS_COMMAND = (
    "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL_QUERY,
)
_WINDOWS_TIMEOUT_SECONDS = 10.0
_SOURCE = "windows-wmi"
_PCI_ID_RE = re.compile(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", re.IGNORECASE)
_NOT_AN_ACCELERATOR_MARKERS = (
    "microsoft basic render",
    "microsoft remote display",
    "microsoft hyper-v video",
)
_VENDOR_IDS = {
    "0x10de": AcceleratorVendor.NVIDIA,
    "0x1002": AcceleratorVendor.AMD,
    "0x8086": AcceleratorVendor.INTEL,
}
_UNAVAILABLE_WARNING = (
    "Windows display adapters could not be listed; accelerator identification "
    "relies on the other probes."
)


@dataclass(frozen=True)
class WindowsAdapterProbe:
    accelerators: list[AcceleratorProfile]
    warnings: list[str]


def detect_windows_adapters(
    command_runner: WindowsCommandRunner | None = None,
    *,
    system: str | None = None,
) -> WindowsAdapterProbe:
    """List Windows adapters; a no-op anywhere else, and never fatal."""
    if (system if system is not None else platform.system()) != "Windows":
        return WindowsAdapterProbe(accelerators=[], warnings=[])
    runner = command_runner or _run
    try:
        completed = runner(_WINDOWS_COMMAND, _WINDOWS_TIMEOUT_SECONDS)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("windows adapter listing failed", exc_info=exc)
        return WindowsAdapterProbe(accelerators=[], warnings=[_UNAVAILABLE_WARNING])
    if completed.returncode != 0:
        return WindowsAdapterProbe(accelerators=[], warnings=[_UNAVAILABLE_WARNING])
    return WindowsAdapterProbe(
        accelerators=parse_windows_video_controllers(completed.stdout), warnings=[]
    )


def parse_windows_video_controllers(output: str) -> list[AcceleratorProfile]:
    """Parse ``ConvertTo-Json`` output: one object for one adapter, a list for several."""
    if not output.strip():
        return []
    try:
        data = json.loads(output)
    except ValueError:
        return []
    rows = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
    accelerators: list[AcceleratorProfile] = []
    for row in rows:
        if isinstance(row, dict) and (accelerator := _accelerator_from_row(row)) is not None:
            accelerators.append(accelerator)
    return accelerators


def _accelerator_from_row(row: dict[str, Any]) -> AcceleratorProfile | None:
    name = _text(row.get("Name"))
    if name is None or any(marker in name.casefold() for marker in _NOT_AN_ACCELERATOR_MARKERS):
        return None
    vendor_id, device_id = _pci_ids(_text(row.get("PNPDeviceID")))
    driver_version = _text(row.get("DriverVersion"))
    return AcceleratorProfile(
        name=name,
        vendor=_VENDOR_IDS.get(vendor_id or "", AcceleratorVendor.UNKNOWN),
        type=AcceleratorType.UNKNOWN,
        vendor_id=vendor_id,
        device_id=device_id,
        detection_sources=[_SOURCE],
        backends=[
            ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=BackendAvailability.UNKNOWN,
                reason=BackendAvailabilityReason.NOT_CHECKED,
                source=_SOURCE,
                detail="Adapter listed by Windows; Vulkan availability was not checked.",
                device_name=name,
                driver_version=driver_version,
            )
        ],
    )


def _pci_ids(pnp_device_id: str | None) -> tuple[str | None, str | None]:
    if pnp_device_id is None:
        return None, None
    match = _PCI_ID_RE.search(pnp_device_id)
    if match is None:
        return None, None
    return f"0x{match.group(1).lower()}", f"0x{match.group(2).lower()}"


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _run(command: tuple[str, ...], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


__all__ = [
    "WindowsAdapterProbe",
    "WindowsCommandRunner",
    "detect_windows_adapters",
    "parse_windows_video_controllers",
]
