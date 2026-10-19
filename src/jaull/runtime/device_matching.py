"""Which llama.cpp device is the accelerator Jaull selected.

Hardware discovery and llama.cpp each name a device in their own way, and
llama.cpp's numbering (``Vulkan0``, ``Vulkan1`` ...) depends on the binary, the
driver and what else is installed; it is not the order of the hardware list. A
runtime identifier is therefore only passed to ``--device`` when it can be tied
to the selected accelerator, and otherwise the launch stays automatic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    LlamaCppBackendCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
)

# Software rasterisers that llama.cpp can enumerate as ordinary devices.
_SOFTWARE_DEVICE_MARKERS = ("llvmpipe", "lavapipe", "swiftshader", "microsoft basic render")
_TRADEMARK_RE = re.compile(r"\((?:r|tm)\)", re.IGNORECASE)
_TRAILING_ANNOTATION_RE = re.compile(r"\s*\([^()]*\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class RuntimeDeviceMatchReason(StrEnum):
    MATCHED_BY_NAME = "matched_by_name"
    ONLY_HARDWARE_DEVICE = "only_hardware_device"
    NO_HARDWARE_DEVICE = "no_hardware_device"
    NAME_MISMATCH = "name_mismatch"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RuntimeDeviceMatch:
    device: LlamaCppRuntimeDevice | None
    reason: RuntimeDeviceMatchReason


def match_runtime_device(
    selection: RuntimeBackendSelection,
    capability: LlamaCppBackendCapability,
) -> RuntimeDeviceMatch:
    """Pick the runtime device for the selected accelerator, or refuse.

    A single hardware device is accepted when nothing contradicts the selection,
    because the runtime then has no other card to confuse it with. Several
    candidates need a name that identifies exactly one of them. Names are
    compared for equality after removing trademark marks and one trailing
    annotation; a near name (``RTX 2060`` against ``RTX 2060 SUPER``) is a
    different device.
    """
    candidates = [
        device for device in capability.devices if not _is_software_renderer(device)
    ]
    if not candidates:
        return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.NO_HARDWARE_DEVICE)

    accelerator = selection.selected_accelerator
    wanted = _normalized_name(accelerator.name) if accelerator is not None else None
    if wanted:
        named = [
            device
            for device in candidates
            if device.name is not None and _normalized_name(device.name) == wanted
        ]
        if len(named) == 1:
            return RuntimeDeviceMatch(named[0], RuntimeDeviceMatchReason.MATCHED_BY_NAME)
        if len(named) > 1:
            return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.AMBIGUOUS)

    if len(candidates) > 1:
        return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.AMBIGUOUS)

    only = candidates[0]
    # Vulkan enumerates every adapter, so a lone device of another name is
    # evidently not the selected card. CUDA and HIP name devices the way their
    # own vendor library does, and a single device there is accepted as before.
    if (
        wanted
        and only.name is not None
        and selection.selected_backend is ComputeBackend.VULKAN
    ):
        return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.NAME_MISMATCH)
    return RuntimeDeviceMatch(only, RuntimeDeviceMatchReason.ONLY_HARDWARE_DEVICE)


def _is_software_renderer(device: LlamaCppRuntimeDevice) -> bool:
    name = (device.name or "").casefold()
    return any(marker in name for marker in _SOFTWARE_DEVICE_MARKERS)


def _normalized_name(name: str) -> str:
    without_marks = _TRADEMARK_RE.sub("", name)
    without_annotation = _TRAILING_ANNOTATION_RE.sub("", without_marks)
    return _NON_ALNUM_RE.sub(" ", without_annotation.casefold()).strip()


def names_agree(left: str, right: str) -> bool:
    """Whether two device names are the same device, by the same rule as matching."""
    normalized = _normalized_name(left)
    return bool(normalized) and normalized == _normalized_name(right)


__all__ = [
    "RuntimeDeviceMatch",
    "RuntimeDeviceMatchReason",
    "match_runtime_device",
    "names_agree",
]
