"""Evaluation placement uses observed hardware, independently of the normal plan."""

from __future__ import annotations

from jaull.domain.hardware import (
    AcceleratorType,
    AcceleratorVendor,
    ComputeBackend,
    HardwareProfile,
)
from jaull.domain.runtime import (
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    SelectedAcceleratorRef,
)
from jaull.domain.runtime_support import available_compute_backends
from jaull.runtime.backend_selection import select_runtime_backend


def quality_backend_selection(
    hardware: HardwareProfile,
    backend: ComputeBackend | None = None,
) -> RuntimeBackendSelection:
    selected = select_runtime_backend(hardware)
    if not hardware.accelerators and len(hardware.gpus) == 1:
        gpu = hardware.gpus[0]
        selected = RuntimeBackendSelection(
            selected_backend=ComputeBackend.CUDA,
            selected_accelerator=SelectedAcceleratorRef(
                name=gpu.name,
                vendor=AcceleratorVendor.NVIDIA,
                type=AcceleratorType.DEDICATED,
                dedicated_memory_bytes=gpu.vram_total_bytes,
            ),
            reason=RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
        )
    if backend is None:
        backend = selected.selected_backend
        if backend not in {ComputeBackend.CUDA, ComputeBackend.VULKAN}:
            backend = ComputeBackend.CPU
    if backend not in {ComputeBackend.CPU, ComputeBackend.CUDA, ComputeBackend.VULKAN}:
        raise ValueError("Quality evaluation supports CPU, CUDA or Vulkan, not this backend.")
    if backend not in available_compute_backends(hardware):
        raise ValueError(
            f"Selected evaluation backend {backend.value} is not available; no fallback."
        )
    if backend is ComputeBackend.CPU:
        return RuntimeBackendSelection(
            selected_backend=backend,
            reason=RuntimeBackendSelectionReason.CPU_FALLBACK,
        )
    if selected.selected_backend is backend:
        return selected
    alternative = next((item for item in selected.alternatives if item.backend is backend), None)
    if alternative is None:
        raise ValueError("Selected evaluation GPU is not identified; no fallback.")
    return RuntimeBackendSelection(
        selected_backend=backend,
        selected_accelerator=alternative.accelerator,
        backend_info=alternative.backend_info,
        reason=alternative.reason,
    )


def quality_memory_hardware(
    hardware: HardwareProfile,
    selection: RuntimeBackendSelection,
) -> HardwareProfile:
    if selection.selected_backend is ComputeBackend.CPU:
        return hardware
    selected = selection.selected_accelerator
    if selected is None:
        raise ValueError("Evaluation accelerator identity is missing.")
    accelerators = [
        item
        for item in hardware.accelerators
        if item.name == selected.name
        and (not selected.uuid or item.uuid == selected.uuid)
        and (not selected.pci_bus_id or item.pci_bus_id == selected.pci_bus_id)
    ]
    gpus = [item for item in hardware.gpus if item.name == selected.name]
    if len(accelerators) > 1 or len(gpus) > 1:
        raise ValueError("Evaluation memory device is ambiguous.")
    return hardware.model_copy(update={"accelerators": accelerators, "gpus": gpus})
