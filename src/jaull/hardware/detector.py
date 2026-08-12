from __future__ import annotations

import platform
import re
from collections.abc import Callable

from jaull.domain.hardware import (
    AcceleratorProfile,
    BackendAvailability,
    ComputeBackendInfo,
    HardwareProfile,
)
from jaull.hardware.cpu import detect_cpu
from jaull.hardware.memory import detect_memory
from jaull.hardware.nvidia import NvmlProvider, detect_nvidia_gpus
from jaull.hardware.storage import detect_storage
from jaull.hardware.vulkan import VulkanCommandRunner, detect_vulkan_accelerators

# Keys reported to ``on_step``, in the order they complete. The guided UI uses
# these to advance a progress list as each real probe returns — there is no
# artificial pacing, a step turns green exactly when its probe finishes.
STEP_OS = "os"
STEP_CPU = "cpu"
STEP_GPU = "gpu"
STEP_STORAGE = "storage"
STEP_PROFILE = "profile"


def detect_hardware(
    nvml_provider: NvmlProvider | None = None,
    on_step: Callable[[str], None] | None = None,
    *,
    vulkan_command_runner: VulkanCommandRunner | None = None,
) -> HardwareProfile:
    warnings: list[str] = []
    notify = on_step if on_step is not None else _ignore

    os_name = _os_display_name()
    notify(STEP_OS)

    cpu = detect_cpu()
    memory = detect_memory()
    notify(STEP_CPU)

    nvidia = detect_nvidia_gpus(nvml_provider)
    warnings.extend(nvidia.warnings)
    vulkan = detect_vulkan_accelerators(vulkan_command_runner)
    warnings.extend(vulkan.warnings)
    accelerators = _merge_accelerators(
        [*nvidia.accelerators],
        vulkan.accelerators,
    )
    notify(STEP_GPU)

    storage = detect_storage()
    if not storage:
        warnings.append("No storage partitions could be enumerated.")
    notify(STEP_STORAGE)

    profile = HardwareProfile(
        os=os_name,
        os_version=platform.version(),
        arch=platform.machine() or "unknown",
        cpu=cpu,
        memory=memory,
        storage=storage,
        gpus=nvidia.gpus,
        accelerators=accelerators,
        warnings=warnings,
    )
    notify(STEP_PROFILE)
    return profile


def _ignore(step: str) -> None:
    del step


def _os_display_name() -> str:
    system = platform.system()
    release = platform.release()
    if system and release:
        return f"{system} {release}"
    return system or "unknown"


def _merge_accelerators(
    base: list[AcceleratorProfile],
    additions: list[AcceleratorProfile],
) -> list[AcceleratorProfile]:
    merged = list(base)
    for addition in additions:
        index = _matching_accelerator_index(merged, addition)
        if index is None:
            merged.append(addition)
            continue
        existing = merged[index]
        merged[index] = existing.model_copy(
            update={
                "uuid": existing.uuid or addition.uuid,
                "pci_bus_id": existing.pci_bus_id or addition.pci_bus_id,
                "vendor_id": existing.vendor_id or addition.vendor_id,
                "device_id": existing.device_id or addition.device_id,
                "detection_sources": _merge_sources(
                    existing.detection_sources, addition.detection_sources
                ),
                "backends": _merge_backends(existing.backends, addition.backends),
            }
        )
    return merged


def _matching_accelerator_index(
    accelerators: list[AcceleratorProfile], candidate: AcceleratorProfile
) -> int | None:
    for index, existing in enumerate(accelerators):
        if _same_stable_identity(existing, candidate):
            return index

    matches: list[int] = []
    candidate_name = _normalized_name(candidate.name)
    for index, existing in enumerate(accelerators):
        if existing.vendor is not candidate.vendor:
            continue
        existing_name = _normalized_name(existing.name)
        if existing_name == candidate_name:
            matches.append(index)
            continue
        if existing_name and candidate_name and (
            existing_name in candidate_name or candidate_name in existing_name
        ):
            matches.append(index)
            continue
        if _likely_same_device_name(existing_name, candidate_name):
            matches.append(index)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def _same_stable_identity(
    existing: AcceleratorProfile, candidate: AcceleratorProfile
) -> bool:
    if existing.uuid and candidate.uuid and existing.uuid == candidate.uuid:
        return True
    if (
        existing.pci_bus_id
        and candidate.pci_bus_id
        and existing.pci_bus_id == candidate.pci_bus_id
    ):
        return True
    return (
        existing.vendor is candidate.vendor
        and existing.vendor_id is not None
        and existing.vendor_id == candidate.vendor_id
        and existing.device_id is not None
        and existing.device_id == candidate.device_id
        and _normalized_name(existing.name) == _normalized_name(candidate.name)
    )


def _merge_backends(
    existing: list[ComputeBackendInfo],
    additions: list[ComputeBackendInfo],
) -> list[ComputeBackendInfo]:
    merged = list(existing)
    for addition in additions:
        for index, current in enumerate(merged):
            if current.backend is addition.backend:
                merged[index] = _stronger_backend(current, addition)
                break
        else:
            merged.append(addition)
    return merged


def _normalized_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _merge_sources(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys([*left, *right]))


def _likely_same_device_name(left: str, right: str) -> bool:
    left_tokens = _device_tokens(left)
    right_tokens = _device_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) >= 2 and (
        any(token.isdigit() for token in overlap)
        or overlap in (left_tokens, right_tokens)
    )


def _device_tokens(name: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", name.casefold()))
    return tokens - {"nvidia", "geforce", "amd", "intel", "graphics", "tm"}


def _stronger_backend(
    current: ComputeBackendInfo, candidate: ComputeBackendInfo
) -> ComputeBackendInfo:
    if _availability_rank(candidate.availability) > _availability_rank(
        current.availability
    ):
        return candidate
    if _availability_rank(candidate.availability) < _availability_rank(
        current.availability
    ):
        return current
    if candidate.software_renderer and not current.software_renderer:
        return candidate
    if candidate.detail and not current.detail:
        return candidate
    return current


def _availability_rank(availability: BackendAvailability) -> int:
    if availability is BackendAvailability.AVAILABLE:
        return 3
    if availability is BackendAvailability.UNAVAILABLE:
        return 2
    return 1
