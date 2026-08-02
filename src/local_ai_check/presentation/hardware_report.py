from __future__ import annotations

from rich.console import Console
from rich.table import Table

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.presentation.console import format_bytes


def render_hardware(profile: HardwareProfile, console: Console) -> None:
    table = Table(title="Local hardware", show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Operating system", profile.os)
    table.add_row("Architecture", profile.arch)
    table.add_row("CPU", profile.cpu.model or "unknown")
    table.add_row("Physical cores", _int(profile.cpu.physical_cores))
    table.add_row("Logical cores", _int(profile.cpu.logical_cores))
    table.add_row("RAM total", format_bytes(profile.memory.total_bytes))
    table.add_row("RAM available", format_bytes(profile.memory.available_bytes))

    if profile.storage:
        for storage in profile.storage:
            label = f"Storage {storage.mountpoint}"
            table.add_row(
                label,
                f"{format_bytes(storage.available_bytes)} free "
                f"/ {format_bytes(storage.total_bytes)} total",
            )
    else:
        table.add_row("Storage", "unknown")

    if profile.gpus:
        for idx, gpu in enumerate(profile.gpus):
            prefix = f"GPU{f' {idx}' if len(profile.gpus) > 1 else ''}"
            table.add_row(prefix, gpu.name)
            table.add_row("VRAM total", format_bytes(gpu.vram_total_bytes))
            table.add_row("VRAM available", format_bytes(gpu.vram_available_bytes))
            table.add_row("Driver version", gpu.driver_version or "unknown")
            table.add_row(
                "CUDA available",
                f"Yes ({gpu.cuda_version})" if gpu.cuda_version else "unknown",
            )
    else:
        table.add_row("GPU", "no NVIDIA GPU detected")

    console.print(table)

    if profile.warnings:
        console.print()
        for warning in profile.warnings:
            console.print(f"[yellow]! {warning}[/yellow]")


def _int(value: int | None) -> str:
    return str(value) if value is not None else "unknown"
