"""Stable machine identity for local evidence matching.

Runtime scans include volatile fields such as currently available RAM/VRAM.
Benchmarks and experiments should match the machine, not the instantaneous
free-memory state at scan time.
"""

from __future__ import annotations

from jaull.domain.hardware import HardwareProfile


def machine_fingerprint(hardware: HardwareProfile) -> tuple[str, ...]:
    """Return a stable fingerprint for local benchmark/experiment evidence."""

    gpus = tuple(
        sorted(f"{gpu.name}:{gpu.vram_total_bytes}" for gpu in hardware.gpus)
    )
    accelerators = tuple(
        sorted(
            ":".join(
                [
                    accelerator.name,
                    accelerator.vendor.value,
                    accelerator.type.value,
                    accelerator.vendor_id or "",
                    accelerator.device_id or "",
                    accelerator.pci_bus_id or "",
                    accelerator.uuid or "",
                    str(accelerator.dedicated_memory_bytes or ""),
                    str(accelerator.shared_memory),
                ]
            )
            for accelerator in hardware.accelerators
        )
    )
    return (
        hardware.os,
        hardware.os_version or "",
        hardware.arch,
        hardware.cpu.model or "",
        str(hardware.cpu.physical_cores or ""),
        str(hardware.cpu.logical_cores or ""),
        str(hardware.memory.total_bytes),
        *gpus,
        *accelerators,
    )


__all__ = ["machine_fingerprint"]
