"""Vulkan device detection from ``vulkaninfo --summary``.

The detector models backend availability in the current environment. In WSL,
for example, ``vulkaninfo`` may expose llvmpipe; that proves the Vulkan API is
present, but not that a GPU accelerator is usable.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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

VulkanCommandRunner = Callable[[tuple[str, ...], float], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class _DrmVram:
    total_bytes: int
    available_bytes: int | None


DrmVramReader = Callable[[str | None, str | None], _DrmVram | None]

_GPU_HEADER_RE = re.compile(r"^\s*GPU\d+\s*:")
_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*=\s*(.*?)\s*$")
_VULKAN_COMMAND = ("vulkaninfo", "--summary")
_VULKAN_TIMEOUT_SECONDS = 5.0
_DRM_ROOT = Path("/sys/class/drm")


@dataclass(frozen=True)
class VulkanProbe:
    accelerators: list[AcceleratorProfile]
    warnings: list[str]


def detect_vulkan_accelerators(
    command_runner: VulkanCommandRunner | None = None,
    *,
    drm_vram_reader: DrmVramReader | None = None,
) -> VulkanProbe:
    runner = command_runner or _run_vulkaninfo
    try:
        completed = runner(_VULKAN_COMMAND, _VULKAN_TIMEOUT_SECONDS)
    except FileNotFoundError:
        return VulkanProbe(
            accelerators=[],
            warnings=["vulkaninfo is not installed; skipping Vulkan detection."],
        )
    except subprocess.TimeoutExpired as exc:
        logger.debug("vulkaninfo timed out", exc_info=exc)
        return VulkanProbe(
            accelerators=[],
            warnings=["vulkaninfo timed out; Vulkan backend availability is unknown."],
        )
    except OSError as exc:
        logger.debug("vulkaninfo failed", exc_info=exc)
        return VulkanProbe(
            accelerators=[],
            warnings=[
                "Could not execute vulkaninfo; Vulkan backend availability is unknown."
            ],
        )

    if completed.returncode != 0:
        return VulkanProbe(
            accelerators=[],
            warnings=[
                "vulkaninfo returned a non-zero exit code; "
                "Vulkan backend availability is unknown."
            ],
        )

    accelerators = accelerators_from_summary(
        completed.stdout,
        drm_vram_reader=drm_vram_reader or _read_drm_vram_bytes,
    )
    warnings: list[str] = []
    if not accelerators:
        warnings.append("No Vulkan devices could be parsed from vulkaninfo output.")
    return VulkanProbe(accelerators=accelerators, warnings=warnings)


def accelerators_from_summary(
    output: str,
    *,
    drm_vram_reader: DrmVramReader | None = None,
) -> list[AcceleratorProfile]:
    return [
        _accelerator_from_device(device, drm_vram_reader=drm_vram_reader)
        for device in _parse_devices(output)
    ]


def _run_vulkaninfo(
    command: tuple[str, ...], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
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


def _parse_devices(output: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in output.splitlines():
        if _GPU_HEADER_RE.match(line):
            if current:
                devices.append(current)
            current = {}
            continue

        match = _KEY_VALUE_RE.match(line)
        if match is None:
            continue
        key, value = match.groups()
        if key not in {
            "apiVersion",
            "vendorID",
            "deviceID",
            "deviceType",
            "deviceName",
            "driverName",
        }:
            continue
        if current is None:
            current = {}
        current[key] = value

    if current:
        devices.append(current)
    return [device for device in devices if "deviceName" in device]


def _accelerator_from_device(
    device: dict[str, str],
    *,
    drm_vram_reader: DrmVramReader | None,
) -> AcceleratorProfile:
    name = device.get("deviceName", "unknown Vulkan device")
    driver = device.get("driverName")
    device_type = device.get("deviceType", "")
    software = _is_software_renderer(name=name, driver=driver, device_type=device_type)
    accelerator_type = _accelerator_type(device_type, software=software)
    drm_vram = (
        drm_vram_reader(device.get("vendorID"), device.get("deviceID"))
        if accelerator_type is AcceleratorType.DEDICATED and drm_vram_reader is not None
        else None
    )
    backend_availability = (
        BackendAvailability.UNAVAILABLE
        if software
        else BackendAvailability.AVAILABLE
    )
    vendor = _vendor(device.get("vendorID"))
    return AcceleratorProfile(
        name=name,
        vendor=vendor,
        type=accelerator_type,
        vendor_id=device.get("vendorID"),
        device_id=device.get("deviceID"),
        dedicated_memory_bytes=drm_vram.total_bytes if drm_vram is not None else None,
        available_memory_bytes=(
            drm_vram.available_bytes if drm_vram is not None else None
        ),
        shared_memory=accelerator_type
        in {AcceleratorType.INTEGRATED, AcceleratorType.SOFTWARE},
        detection_sources=["vulkaninfo", "drm"] if drm_vram is not None else ["vulkaninfo"],
        backends=_backends_for_vulkan_device(
            vendor=vendor,
            vulkan=ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=backend_availability,
                reason=(
                    BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY
                    if software
                    else BackendAvailabilityReason.PROBE_AVAILABLE
                ),
                source="vulkaninfo",
                detail=(
                    "Vulkan API is present, but only a software renderer is exposed."
                    if software
                    else "Vulkan device exposed by vulkaninfo."
                ),
                api_version=device.get("apiVersion"),
                device_name=name,
                driver_name=driver,
                software_renderer=software,
            ),
            software=software,
        ),
    )


def _read_drm_vram_bytes(
    vendor_id: str | None,
    device_id: str | None,
    *,
    drm_root: Path = _DRM_ROOT,
) -> _DrmVram | None:
    """Read discrete AMD-style VRAM from Linux DRM sysfs when unambiguous.

    Vulkan's summary has device identity but no memory information. DRM exposes
    physical total VRAM and, on drivers that provide it, current usage. The
    latter is optional: capacity remains useful for planning, while Hardware
    Fit deliberately requires observed availability.
    """

    normalized_vendor = _normalized_pci_id(vendor_id)
    normalized_device = _normalized_pci_id(device_id)
    if normalized_vendor is None or normalized_device is None:
        return None

    matches: list[_DrmVram] = []
    try:
        cards = sorted(
            path
            for path in drm_root.iterdir()
            if re.fullmatch(r"card\d+", path.name)
        )
    except OSError:
        return None

    for card in cards:
        device = card / "device"
        if (
            _read_pci_id(device / "vendor") != normalized_vendor
            or _read_pci_id(device / "device") != normalized_device
        ):
            continue
        total_bytes = _read_non_negative_int(device / "mem_info_vram_total")
        if total_bytes is None or total_bytes <= 0:
            continue
        used_bytes = _read_non_negative_int(device / "mem_info_vram_used")
        matches.append(
            _DrmVram(
                total_bytes=total_bytes,
                available_bytes=(
                    max(0, total_bytes - used_bytes) if used_bytes is not None else None
                ),
            )
        )

    # Vulkan's summary does not expose a PCI bus ID. Do not guess which of two
    # identical cards supplies the memory reading.
    return matches[0] if len(matches) == 1 else None


def _normalized_pci_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return f"0x{int(value.strip(), 0):04x}"
    except ValueError:
        return None


def _read_pci_id(path: Path) -> str | None:
    try:
        return _normalized_pci_id(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _read_non_negative_int(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


def _backends_for_vulkan_device(
    *,
    vendor: AcceleratorVendor,
    vulkan: ComputeBackendInfo,
    software: bool,
) -> list[ComputeBackendInfo]:
    if software:
        cuda = BackendAvailability.UNAVAILABLE
        hip = BackendAvailability.UNAVAILABLE
        cuda_reason = BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY
        hip_reason = BackendAvailabilityReason.SOFTWARE_RENDERER_ONLY
    elif vendor is AcceleratorVendor.NVIDIA:
        cuda = BackendAvailability.UNKNOWN
        hip = BackendAvailability.UNKNOWN
        cuda_reason = BackendAvailabilityReason.NOT_CHECKED
        hip_reason = BackendAvailabilityReason.NOT_CHECKED
    elif vendor is AcceleratorVendor.AMD:
        cuda = BackendAvailability.UNAVAILABLE
        hip = BackendAvailability.UNKNOWN
        cuda_reason = BackendAvailabilityReason.VENDOR_NOT_SUPPORTED
        hip_reason = BackendAvailabilityReason.NOT_CHECKED
    elif vendor in {AcceleratorVendor.INTEL, AcceleratorVendor.OTHER}:
        cuda = BackendAvailability.UNAVAILABLE
        hip = BackendAvailability.UNAVAILABLE
        cuda_reason = BackendAvailabilityReason.VENDOR_NOT_SUPPORTED
        hip_reason = BackendAvailabilityReason.VENDOR_NOT_SUPPORTED
    else:
        cuda = BackendAvailability.UNKNOWN
        hip = BackendAvailability.UNKNOWN
        cuda_reason = BackendAvailabilityReason.NOT_CHECKED
        hip_reason = BackendAvailabilityReason.NOT_CHECKED

    return [
        ComputeBackendInfo(
            backend=ComputeBackend.CUDA,
            availability=cuda,
            reason=cuda_reason,
            source="vulkaninfo",
            detail=_non_vulkan_backend_detail(
                backend=ComputeBackend.CUDA,
                availability=cuda,
                vendor=vendor,
                software=software,
            ),
        ),
        vulkan,
        ComputeBackendInfo(
            backend=ComputeBackend.HIP,
            availability=hip,
            reason=hip_reason,
            source="vulkaninfo",
            detail=_non_vulkan_backend_detail(
                backend=ComputeBackend.HIP,
                availability=hip,
                vendor=vendor,
                software=software,
            ),
        ),
    ]


def _non_vulkan_backend_detail(
    *,
    backend: ComputeBackend,
    availability: BackendAvailability,
    vendor: AcceleratorVendor,
    software: bool,
) -> str:
    if software:
        return f"{backend.value} does not apply to a software renderer."
    if availability is BackendAvailability.UNKNOWN:
        return f"{backend.value} backend was not checked for this {vendor.value} device."
    return f"{backend.value} backend does not apply to this {vendor.value} device."


def _is_software_renderer(
    *, name: str, driver: str | None, device_type: str
) -> bool:
    lowered_name = name.lower()
    lowered_driver = (driver or "").lower()
    return (
        device_type == "PHYSICAL_DEVICE_TYPE_CPU"
        or "llvmpipe" in lowered_name
        or "llvmpipe" in lowered_driver
    )


def _accelerator_type(device_type: str, *, software: bool) -> AcceleratorType:
    if software:
        return AcceleratorType.SOFTWARE
    if device_type == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU":
        return AcceleratorType.DEDICATED
    if device_type == "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU":
        return AcceleratorType.INTEGRATED
    return AcceleratorType.UNKNOWN


def _vendor(vendor_id: str | None) -> AcceleratorVendor:
    normalized = (vendor_id or "").lower()
    if normalized == "0x10de":
        return AcceleratorVendor.NVIDIA
    if normalized == "0x1002":
        return AcceleratorVendor.AMD
    if normalized == "0x8086":
        return AcceleratorVendor.INTEL
    if normalized:
        return AcceleratorVendor.OTHER
    return AcceleratorVendor.UNKNOWN


__all__ = [
    "VulkanCommandRunner",
    "VulkanProbe",
    "accelerators_from_summary",
    "detect_vulkan_accelerators",
]
