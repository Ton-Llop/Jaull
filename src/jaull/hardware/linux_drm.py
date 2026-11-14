"""Which GPUs the Linux kernel lists, from DRM sysfs (``/sys/class/drm/card*``).

Identification only, like ``windows_adapters``: it says that a display device
exists, its PCI vendor, device and bus, its name from ``pci.ids`` when the
system has that file, and the VRAM ``amdgpu`` reports. It never says that a
compute backend can use the card. The detector adds a card from here only when
no other probe reported it, so that a GPU whose Vulkan driver is missing or too
old is still shown, with a note on what to install, instead of disappearing.

The type stays ``UNKNOWN``, so the memory read here never enters planning:
only a backend probe makes a device usable.
"""

from __future__ import annotations

import platform
import re
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

DRM_SOURCE = "drm"
_CARD_RE = re.compile(r"card\d+")
# PCI base class 0x03: display controller.
_DISPLAY_CLASS_PREFIX = "0x03"
_VENDOR_NAMES = {
    "0x1002": (AcceleratorVendor.AMD, "AMD"),
    "0x8086": (AcceleratorVendor.INTEL, "Intel"),
    "0x10de": (AcceleratorVendor.NVIDIA, "NVIDIA"),
}
# Module-level so tests can point detection at a fake tree.
DRM_ROOT = Path("/sys/class/drm")
PCI_IDS_PATHS = (Path("/usr/share/hwdata/pci.ids"), Path("/usr/share/misc/pci.ids"))


def detect_drm_adapters(
    *,
    system: str | None = None,
    drm_root: Path | None = None,
    pci_ids_paths: tuple[Path, ...] | None = None,
) -> list[AcceleratorProfile]:
    """List display devices from DRM sysfs; empty anywhere but Linux, and never fatal."""
    if (system if system is not None else platform.system()) != "Linux":
        return []
    root = drm_root if drm_root is not None else DRM_ROOT
    try:
        cards = sorted(path for path in root.iterdir() if _CARD_RE.fullmatch(path.name))
    except OSError:
        return []
    names = _PciNames(pci_ids_paths if pci_ids_paths is not None else PCI_IDS_PATHS)
    found: list[AcceleratorProfile] = []
    seen_buses: set[str] = set()
    for card in cards:
        accelerator = _accelerator_from_card(card / "device", names)
        if accelerator is None:
            continue
        bus = accelerator.pci_bus_id
        if bus is not None:
            if bus in seen_buses:
                continue
            seen_buses.add(bus)
        found.append(accelerator)
    return found


def normalized_pci_id(value: str | None) -> str | None:
    """``0x1002``, ``1002`` or ``0X1002`` -> ``0x1002``; anything else -> None."""
    if value is None:
        return None
    text = value.strip().lower()
    try:
        number = int(text, 16) if not text.startswith("0x") else int(text, 0)
    except ValueError:
        return None
    return f"0x{number:04x}"


def normalized_pci_bus(value: str | None) -> str | None:
    """``0000:03:00.0`` and NVML's ``00000000:03:00.0`` -> ``0000:03:00.0``."""
    if value is None:
        return None
    match = re.fullmatch(r"([0-9a-f]+):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])",
                         value.strip().lower())
    if match is None:
        return None
    domain, bus, slot, function = match.groups()
    return f"{int(domain, 16):04x}:{bus}:{slot}.{function}"


def _accelerator_from_card(device: Path, names: _PciNames) -> AcceleratorProfile | None:
    pci_class = _read(device / "class")
    if pci_class is None or not pci_class.lower().startswith(_DISPLAY_CLASS_PREFIX):
        return None
    vendor_id = normalized_pci_id(_read(device / "vendor"))
    device_id = normalized_pci_id(_read(device / "device"))
    if vendor_id is None or device_id is None:
        return None
    vendor, vendor_label = _VENDOR_NAMES.get(vendor_id, (AcceleratorVendor.OTHER, "GPU"))
    model = names.device_name(vendor_id, device_id)
    name = (f"{vendor_label} {model}" if model is not None
            else f"{vendor_label} GPU (PCI {vendor_id[2:]}:{device_id[2:]})")
    vram = _read_int(device / "mem_info_vram_total")
    return AcceleratorProfile(
        name=name,
        vendor=vendor,
        type=AcceleratorType.UNKNOWN,
        vendor_id=vendor_id,
        device_id=device_id,
        pci_bus_id=normalized_pci_bus(_bus_of(device)),
        dedicated_memory_bytes=vram if vram else None,
        detection_sources=[DRM_SOURCE],
        backends=[
            ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=BackendAvailability.UNKNOWN,
                reason=BackendAvailabilityReason.NOT_CHECKED,
                source=DRM_SOURCE,
                detail="GPU listed by the Linux kernel; no Vulkan device was confirmed for it.",
                device_name=name,
                driver_name=_driver_of(device),
            )
        ],
    )


class _PciNames:
    """Device names from ``pci.ids``, read lazily and only for the vendors asked for."""

    def __init__(self, paths: tuple[Path, ...]) -> None:
        self._paths = paths
        self._names: dict[str, dict[str, str]] | None = None

    def device_name(self, vendor_id: str, device_id: str) -> str | None:
        if self._names is None:
            self._names = self._load()
        return self._names.get(vendor_id[2:], {}).get(device_id[2:])

    def _load(self) -> dict[str, dict[str, str]]:
        wanted = {vendor[2:] for vendor in _VENDOR_NAMES}
        for path in self._paths:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            names: dict[str, dict[str, str]] = {}
            vendor: str | None = None
            for line in lines:
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("\t"):
                    # "1002  Advanced Micro Devices, Inc. [AMD/ATI]"; the class list follows
                    # the vendors and starts with "C ", which ends the useful part.
                    if line.startswith("C "):
                        break
                    code = line[:4].lower()
                    vendor = code if code in wanted else None
                    continue
                if vendor is not None and not line.startswith("\t\t"):
                    # "\t7551  Navi 48 [Radeon AI PRO R9700]"
                    code, _, label = line.strip().partition(" ")
                    names.setdefault(vendor, {})[code.lower()] = label.strip()
            return names
        return {}


def _bus_of(device: Path) -> str | None:
    try:
        return device.resolve().name
    except OSError:
        return None


def _driver_of(device: Path) -> str | None:
    link = device / "driver"
    try:
        return link.resolve(strict=True).name if link.is_symlink() else None
    except OSError:
        return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read(path)
    try:
        value = int(text) if text is not None else None
    except ValueError:
        return None
    return value if value is not None and value > 0 else None


__all__ = [
    "DRM_ROOT",
    "DRM_SOURCE",
    "PCI_IDS_PATHS",
    "detect_drm_adapters",
    "normalized_pci_bus",
    "normalized_pci_id",
]
