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
# The summary carries no memory. The full report does, but it enumerates every
# extension and property, so it is read only when a discrete device needs it.
_VULKAN_MEMORY_COMMAND = ("vulkaninfo",)
_VULKAN_MEMORY_TIMEOUT_SECONDS = 15.0
_HEAP_START_RE = re.compile(r"^\s*memoryHeaps\[\d+\]")
_LEADING_INT_RE = re.compile(r"^\s*(\d+)")
_DEVICE_LOCAL_HEAP_FLAG = "MEMORY_HEAP_DEVICE_LOCAL_BIT"
_VULKAN_MEMORY_SOURCE = "vulkan-memory-heap"
_VULKAN_BUDGET_SOURCE = "vulkan-memory-budget"
_DRM_ROOT = Path("/sys/class/drm")


@dataclass(frozen=True)
class VulkanProbe:
    accelerators: list[AcceleratorProfile]
    warnings: list[str]


@dataclass(frozen=True)
class VulkanHeapMemory:
    """The device-local heap of one device, as the Vulkan driver reports it.

    ``total_bytes`` is the heap *size*: capacity, never free memory.
    ``budget_bytes`` exists only when the driver implements
    ``VK_EXT_memory_budget``. Per the extension it is a rough estimate of how
    much this process can allocate before allocations fail or degrade, so it is
    an OS/driver-defined allowance and not a free-memory measurement.
    """

    total_bytes: int
    budget_bytes: int | None


@dataclass(frozen=True)
class VulkanDeviceHeaps:
    vendor_id: str | None
    device_id: str | None
    name: str | None
    heap: VulkanHeapMemory | None


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
    if _needs_heap_memory(accelerators):
        accelerators, memory_warnings = _with_heap_memory(accelerators, runner)
        warnings.extend(memory_warnings)
    return VulkanProbe(accelerators=accelerators, warnings=warnings)


def _needs_heap_memory(accelerators: list[AcceleratorProfile]) -> bool:
    return any(_wants_heap_memory(item) for item in accelerators)


def _wants_heap_memory(accelerator: AcceleratorProfile) -> bool:
    """Only a discrete device with no memory from a better source reads the heaps.

    NVIDIA memory belongs to NVML, which the detector merges in; running the
    full report for those devices would only add latency.
    """
    return (
        accelerator.type is AcceleratorType.DEDICATED
        and accelerator.vendor is not AcceleratorVendor.NVIDIA
        and not accelerator.shared_memory
        and accelerator.dedicated_memory_bytes is None
    )


def _with_heap_memory(
    accelerators: list[AcceleratorProfile], runner: VulkanCommandRunner
) -> tuple[list[AcceleratorProfile], list[str]]:
    try:
        completed = runner(_VULKAN_MEMORY_COMMAND, _VULKAN_MEMORY_TIMEOUT_SECONDS)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("vulkaninfo memory report failed", exc_info=exc)
        return accelerators, [_HEAP_MEMORY_UNKNOWN]
    if completed.returncode != 0:
        return accelerators, [_HEAP_MEMORY_UNKNOWN]

    reported = parse_vulkan_heap_memory(completed.stdout)
    enriched: list[AcceleratorProfile] = []
    for accelerator in accelerators:
        heap = _matching_heap(accelerator, accelerators, reported)
        enriched.append(
            _with_heap(accelerator, heap) if heap is not None else accelerator
        )
    warnings: list[str] = []
    if any(
        _wants_heap_memory(item) and item.dedicated_memory_bytes is None for item in enriched
    ):
        warnings.append(_HEAP_MEMORY_UNKNOWN)
    if any(_VULKAN_BUDGET_SOURCE in item.detection_sources for item in enriched):
        warnings.append(_BUDGET_SEMANTICS)
    return enriched, warnings


_BUDGET_SEMANTICS = (
    "Vulkan memory budget (VK_EXT_memory_budget) is an OS/driver estimate of how much "
    "this process may allocate and can depend on system load; it is not a free-VRAM "
    "measurement, and Jaull applies its own reserve and margin on top of it."
)

_HEAP_MEMORY_UNKNOWN = (
    "Vulkan device memory could not be read unambiguously; "
    "accelerator memory is unknown, not assumed."
)


def _device_key(
    vendor_id: str | None, device_id: str | None, name: str | None
) -> tuple[str | None, str | None, str | None]:
    return (
        _normalized_pci_id(vendor_id),
        _normalized_pci_id(device_id),
        name.strip() if name is not None else None,
    )


def _matching_heap(
    accelerator: AcceleratorProfile,
    all_accelerators: list[AcceleratorProfile],
    reported: list[VulkanDeviceHeaps],
) -> VulkanHeapMemory | None:
    """Join a summary device to its heaps, or refuse when the join is not unique.

    The summary has no PCI bus ID, so two identical cards cannot be told apart.
    """
    if not _wants_heap_memory(accelerator):
        return None
    key = _device_key(accelerator.vendor_id, accelerator.device_id, accelerator.name)
    if None in key:
        return None
    same_in_summary = [
        item
        for item in all_accelerators
        if _device_key(item.vendor_id, item.device_id, item.name) == key
    ]
    same_in_report = [
        item for item in reported if _device_key(item.vendor_id, item.device_id, item.name) == key
    ]
    if len(same_in_summary) != 1 or len(same_in_report) != 1:
        return None
    return same_in_report[0].heap


def _with_heap(
    accelerator: AcceleratorProfile, heap: VulkanHeapMemory
) -> AcceleratorProfile:
    sources = [*accelerator.detection_sources, _VULKAN_MEMORY_SOURCE]
    if heap.budget_bytes is not None:
        sources.append(_VULKAN_BUDGET_SOURCE)
    return accelerator.model_copy(
        update={
            "dedicated_memory_bytes": heap.total_bytes,
            "available_memory_bytes": heap.budget_bytes,
            "detection_sources": sources,
        }
    )


def parse_vulkan_heap_memory(output: str) -> list[VulkanDeviceHeaps]:
    """Read each device's largest device-local heap from a full ``vulkaninfo`` report.

    Host-only heaps are ignored. A budget that is zero or larger than the heap
    itself is not a usable allowance and is dropped, leaving capacity only.
    """
    devices: list[dict[str, str]] = []
    heaps: list[list[dict[str, int | bool]]] = []
    current_heap: dict[str, int | bool] | None = None
    for line in output.splitlines():
        if _GPU_HEADER_RE.match(line):
            devices.append({})
            heaps.append([])
            current_heap = None
            continue
        if not devices:
            continue
        if _HEAP_START_RE.match(line):
            current_heap = {"device_local": False}
            heaps[-1].append(current_heap)
            continue
        stripped = line.strip()
        if current_heap is not None and stripped == _DEVICE_LOCAL_HEAP_FLAG:
            current_heap["device_local"] = True
            continue
        match = _KEY_VALUE_RE.match(line)
        if match is None:
            is_section_header = stripped.endswith(":") and not line.startswith((" ", "\t"))
            if stripped.startswith("memoryType") or is_section_header:
                current_heap = None
            continue
        key, value = match.groups()
        if key in {"vendorID", "deviceID", "deviceName"}:
            devices[-1].setdefault(key, value)
        elif current_heap is not None and key in {"size", "budget", "usage"}:
            number = _LEADING_INT_RE.match(value)
            if number is not None:
                current_heap[key] = int(number.group(1))
        elif key.startswith("memoryType"):
            current_heap = None

    return [
        VulkanDeviceHeaps(
            vendor_id=device.get("vendorID"),
            device_id=device.get("deviceID"),
            name=device.get("deviceName"),
            heap=_largest_device_local_heap(device_heaps),
        )
        for device, device_heaps in zip(devices, heaps, strict=True)
        if "deviceName" in device
    ]


def _largest_device_local_heap(
    heaps: list[dict[str, int | bool]],
) -> VulkanHeapMemory | None:
    candidates = [
        heap
        for heap in heaps
        if heap.get("device_local") is True
        and isinstance(heap.get("size"), int)
        and int(heap["size"]) > 0
    ]
    if not candidates:
        return None
    largest = max(candidates, key=lambda heap: int(heap["size"]))
    size = int(largest["size"])
    budget = largest.get("budget")
    usable = (
        int(budget) if isinstance(budget, int) and not isinstance(budget, bool)
        and 0 < int(budget) <= size else None
    )
    return VulkanHeapMemory(total_bytes=size, budget_bytes=usable)


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
            "driverInfo",
            "driverVersion",
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
                driver_version=device.get("driverInfo") or device.get("driverVersion"),
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
    "VulkanDeviceHeaps",
    "VulkanHeapMemory",
    "VulkanProbe",
    "accelerators_from_summary",
    "detect_vulkan_accelerators",
    "parse_vulkan_heap_memory",
]
