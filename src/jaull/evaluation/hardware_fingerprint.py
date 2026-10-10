"""Stable machine identity for local evidence matching.

Runtime scans include volatile fields such as currently available RAM/VRAM.
Benchmarks and experiments should match the machine, not the instantaneous
free-memory state at scan time.
"""

from __future__ import annotations

from jaull.domain.hardware import HardwareProfile

#: For evidence matching: WSL reports total RAM a few KiB apart between boots;
#: 256 MiB never hides a different machine but absorbs that drift.
EVIDENCE_MEMORY_GRANULARITY = 256 * 1024 * 1024


def machine_fingerprint(
    hardware: HardwareProfile, *, memory_granularity: int | None = None,
) -> tuple[str, ...]:
    """Return a stable fingerprint for local benchmark/experiment evidence.

    ``memory_granularity`` rounds total RAM to that many bytes. The default keeps
    the exact figure that case manifests already persist; evidence matching passes
    a granularity because WSL reports a total that moves by a few KiB per boot.
    """
    memory = hardware.memory.total_bytes
    if memory_granularity is not None and memory is not None:
        memory = round(memory / memory_granularity) * memory_granularity

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
        str(memory),
        *gpus,
        *accelerators,
    )


__all__ = ["EVIDENCE_MEMORY_GRANULARITY", "machine_fingerprint"]
