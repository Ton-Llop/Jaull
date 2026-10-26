"""Which llama.cpp device is the accelerator Jaull selected.

Hardware discovery and llama.cpp each name a device in their own way, and
llama.cpp's numbering (``Vulkan0``, ``Vulkan1`` ...) depends on the binary, the
driver and what else is installed; it is not the order of the hardware list. A
runtime identifier is therefore only passed to ``--device`` when it can be tied
to the selected accelerator. An unconfirmed Vulkan match blocks execution;
CUDA/HIP retain their existing single-device fallback.
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
    SelectedAcceleratorRef,
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

    Vulkan requires a unique name match. CUDA/HIP retain their single-device
    fallback; several candidates need a name that identifies exactly one. Names are
    compared for equality after removing trademark marks; a trailing annotation is
    only set aside when just one of the two names has it (see ``names_agree``). A
    near name (``RTX 2060`` against ``RTX 2060 SUPER``) is a different device.
    """
    candidates = [
        device for device in capability.devices if not _is_software_renderer(device)
    ]
    if not candidates:
        return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.NO_HARDWARE_DEVICE)

    accelerator = selection.selected_accelerator
    if accelerator is not None and _full_name(accelerator.name):
        named = [
            device
            for device in candidates
            if device.name is not None and names_agree(device.name, accelerator.name)
        ]
        if len(named) == 1:
            return RuntimeDeviceMatch(named[0], RuntimeDeviceMatchReason.MATCHED_BY_NAME)
        if len(named) > 1:
            return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.AMBIGUOUS)

    if len(candidates) > 1:
        return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.AMBIGUOUS)

    only = candidates[0]
    if selection.selected_backend is ComputeBackend.VULKAN:
        # A sole visible adapter may be an integrated GPU, not the selected card.
        return RuntimeDeviceMatch(None, RuntimeDeviceMatchReason.NAME_MISMATCH)
    # Preserve the existing CUDA/HIP single-device policy.
    return RuntimeDeviceMatch(only, RuntimeDeviceMatchReason.ONLY_HARDWARE_DEVICE)


def _is_software_renderer(device: LlamaCppRuntimeDevice) -> bool:
    name = (device.name or "").casefold()
    return any(marker in name for marker in _SOFTWARE_DEVICE_MARKERS)


def _full_name(name: str) -> str:
    """The whole name, without trademark marks, case or punctuation."""
    without_marks = _TRADEMARK_RE.sub("", name)
    return _NON_ALNUM_RE.sub(" ", without_marks.casefold()).strip()


def _without_annotation(name: str) -> str:
    return _full_name(_TRAILING_ANNOTATION_RE.sub("", _TRADEMARK_RE.sub("", name)))


def _has_annotation(name: str) -> bool:
    return _TRAILING_ANNOTATION_RE.search(_TRADEMARK_RE.sub("", name)) is not None


def names_agree(left: str, right: str) -> bool:
    """Whether two device names are the same device, by the same rule as matching.

    The whole names must be equal. A trailing parenthesis is only set aside when
    exactly one of the two names carries it (``... Graphics`` against
    ``... Graphics (BMG G21)``): when both carry one, it is part of the identity.
    Mesa's Dozen layer, for instance, wraps the real name in it
    (``Microsoft Direct3D12 (Intel(R) Arc(TM) Pro B60 Graphics)``), so removing it
    would make every Dozen device the same device.
    """
    full = _full_name(left)
    if not full:
        return False
    if full == _full_name(right):
        return True
    return (
        _has_annotation(left) != _has_annotation(right)
        and _without_annotation(left) == _without_annotation(right)
    )


def selected_accelerator_matches(
    expected: SelectedAcceleratorRef | None, observed: SelectedAcceleratorRef | None,
) -> bool:
    """Match retained physical identity, without comparing discovery-source metadata."""
    return (
        expected is not None and observed is not None
        and expected.vendor is observed.vendor and names_agree(expected.name, observed.name)
        and all(
            getattr(observed, field) == getattr(expected, field)
            for field in ("uuid", "pci_bus_id", "vendor_id", "device_id")
            if getattr(expected, field) is not None
        )
    )


def runtime_device_name_matches(output: str, runtime_id: str, name: str) -> bool:
    """Tie an execution's allocation id to its own device enumeration, not a prior probe."""
    listed: set[tuple[str, str]] = set()
    for line in output.splitlines():
        vulkan = re.search(r"\bggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\|", line)
        if vulkan:
            listed.add((f"Vulkan{vulkan[1]}", vulkan[2]))
        cuda = re.search(r"\bDevice (\d+): ([^,]+),", line)
        if cuda:
            listed.add((f"CUDA{cuda[1]}", cuda[2]))
        used = re.search(
            r"using device (CUDA\d+|Vulkan\d+)\s+\((.+?)\)(?=\s*(?:\(|-|$))", line,
        )
        if used:
            listed.add((used[1], used[2]))
    prefix = runtime_id.rstrip("0123456789")
    candidates = [(device, label) for device, label in listed if device.startswith(prefix)]
    return (
        {device for device, label in candidates if names_agree(label, name)} == {runtime_id}
        and all(names_agree(label, name) for device, label in candidates if device == runtime_id)
    )


__all__ = [
    "RuntimeDeviceMatch",
    "RuntimeDeviceMatchReason",
    "match_runtime_device",
    "names_agree",
    "runtime_device_name_matches",
    "selected_accelerator_matches",
]
