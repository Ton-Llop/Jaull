"""Pure compute-backend selection from already detected hardware.

This module chooses a target backend from observed accelerator/backend
availability. It does not inspect or validate an installed runtime binary; a
Vulkan selection only means "Vulkan is the preferred target for this
environment", not "the local llama-cli was built with Vulkan support".
"""

from __future__ import annotations

from dataclasses import dataclass

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    HardwareProfile,
)
from jaull.domain.runtime import (
    RuntimeBackendCandidate,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    SelectedAcceleratorRef,
)

_BACKEND_POLICY: dict[AcceleratorVendor, tuple[ComputeBackend, ...]] = {
    AcceleratorVendor.NVIDIA: (ComputeBackend.CUDA, ComputeBackend.VULKAN),
    AcceleratorVendor.AMD: (ComputeBackend.HIP, ComputeBackend.VULKAN),
    AcceleratorVendor.INTEL: (ComputeBackend.VULKAN,),
    AcceleratorVendor.OTHER: (ComputeBackend.VULKAN,),
    AcceleratorVendor.UNKNOWN: (ComputeBackend.VULKAN,),
}

_CPU_NOTE = (
    "Backend selection is based on observed hardware/backend availability only; "
    "runtime binary support is validated in a later phase."
)


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: RuntimeBackendCandidate
    discovery_index: int
    backend_rank: int
    accelerator_rank: int
    dedicated_memory_bytes: int


def select_runtime_backend(hardware: HardwareProfile) -> RuntimeBackendSelection:
    """Choose the preferred compute backend for the current environment.

    Only ``BackendAvailability.AVAILABLE`` backends are selectable. Unknown or
    unavailable backends are preserved in hardware details but are never tried
    automatically. Software renderers such as llvmpipe are ignored for hardware
    acceleration.
    """
    ranked = _ranked_candidates(hardware)
    cpu = _cpu_candidate(
        reason=(
            RuntimeBackendSelectionReason.SOFTWARE_RENDERER_IGNORED
            if _has_software_renderer(hardware)
            else RuntimeBackendSelectionReason.NO_USABLE_ACCELERATOR
        )
    )
    if not ranked:
        return RuntimeBackendSelection(
            selected_backend=ComputeBackend.CPU,
            selected_accelerator=None,
            reason=cpu.reason,
            backend_info=None,
            alternatives=[],
            notes=[_CPU_NOTE],
        )

    selected = ranked[0].candidate
    alternatives = [item.candidate for item in ranked[1:]]
    alternatives.append(_cpu_candidate(reason=RuntimeBackendSelectionReason.CPU_FALLBACK))
    return RuntimeBackendSelection(
        selected_backend=selected.backend,
        selected_accelerator=selected.accelerator,
        reason=selected.reason,
        backend_info=selected.backend_info,
        alternatives=alternatives,
        notes=[_CPU_NOTE],
    )


def _ranked_candidates(hardware: HardwareProfile) -> list[_RankedCandidate]:
    ranked: list[_RankedCandidate] = []
    for discovery_index, accelerator in enumerate(hardware.accelerators):
        if accelerator.type is AcceleratorType.SOFTWARE:
            continue
        for backend in _available_backends(accelerator):
            policy = _BACKEND_POLICY.get(
                accelerator.vendor,
                _BACKEND_POLICY[AcceleratorVendor.UNKNOWN],
            )
            if backend.backend not in policy:
                continue
            ranked.append(
                _RankedCandidate(
                    candidate=RuntimeBackendCandidate(
                        backend=backend.backend,
                        accelerator=_accelerator_ref(accelerator),
                        backend_info=backend,
                        reason=_selection_reason(backend.backend),
                    ),
                    discovery_index=discovery_index,
                    backend_rank=policy.index(backend.backend),
                    accelerator_rank=_accelerator_rank(accelerator.type),
                    dedicated_memory_bytes=accelerator.dedicated_memory_bytes or -1,
                )
            )
    return sorted(
        ranked,
        key=lambda item: (
            item.backend_rank,
            item.accelerator_rank,
            -item.dedicated_memory_bytes,
            item.discovery_index,
        ),
    )


def _available_backends(
    accelerator: AcceleratorProfile,
) -> list[ComputeBackendInfo]:
    return [
        backend
        for backend in accelerator.backends
        if backend.availability is BackendAvailability.AVAILABLE
    ]


def _accelerator_ref(accelerator: AcceleratorProfile) -> SelectedAcceleratorRef:
    return SelectedAcceleratorRef(
        name=accelerator.name,
        vendor=accelerator.vendor,
        type=accelerator.type,
        vendor_id=accelerator.vendor_id,
        device_id=accelerator.device_id,
        pci_bus_id=accelerator.pci_bus_id,
        uuid=accelerator.uuid,
        dedicated_memory_bytes=accelerator.dedicated_memory_bytes,
        shared_memory=accelerator.shared_memory,
        detection_sources=list(accelerator.detection_sources),
    )


def _selection_reason(backend: ComputeBackend) -> RuntimeBackendSelectionReason:
    if backend is ComputeBackend.VULKAN:
        return RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE
    return RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE


def _accelerator_rank(accelerator_type: AcceleratorType) -> int:
    if accelerator_type is AcceleratorType.DEDICATED:
        return 0
    if accelerator_type is AcceleratorType.INTEGRATED:
        return 1
    return 2


def _cpu_candidate(reason: RuntimeBackendSelectionReason) -> RuntimeBackendCandidate:
    return RuntimeBackendCandidate(
        backend=ComputeBackend.CPU,
        accelerator=None,
        backend_info=None,
        reason=reason,
    )


def _has_software_renderer(hardware: HardwareProfile) -> bool:
    return any(
        accelerator.type is AcceleratorType.SOFTWARE
        for accelerator in hardware.accelerators
    )


__all__ = ["select_runtime_backend"]
