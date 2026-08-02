from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, LoadingIndicator

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.hardware.detector import detect_hardware
from local_ai_check.tui.widgets.banner import Banner
from local_ai_check.tui.widgets.summary_card import SummaryCard
from local_ai_check.tui.widgets.warnings_panel import WarningsPanel

if TYPE_CHECKING:
    from local_ai_check.tui.app import LocalAiCheckApp


class ScanScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner("Local hardware scan", "Detecting CPU, memory, storage and GPU.")
        yield LoadingIndicator(id="scan-loading")
        yield VerticalScroll(Vertical(id="scan-content"))
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._probe, thread=True)

    def _probe(self) -> None:
        profile = detect_hardware()
        self.app.call_from_thread(self._populate, profile)

    def _populate(self, profile: HardwareProfile) -> None:
        app = self._app()
        app.hardware_profile = profile

        self.query_one("#scan-loading", LoadingIndicator).display = False
        content = self.query_one("#scan-content", Vertical)
        content.remove_children()

        content.mount(
            SummaryCard(
                "System",
                [
                    ("Operating system", profile.os),
                    ("Architecture", profile.arch),
                ],
            )
        )
        content.mount(
            SummaryCard(
                "CPU",
                [
                    ("Model", profile.cpu.model or "unknown"),
                    ("Physical cores", _int(profile.cpu.physical_cores)),
                    ("Logical cores", _int(profile.cpu.logical_cores)),
                ],
            )
        )
        mem_rows: list[tuple[str, str]] = [
            ("RAM total", _fmt(profile.memory.total_bytes)),
            ("RAM available", _fmt(profile.memory.available_bytes)),
        ]
        for storage in profile.storage:
            mem_rows.append(
                (
                    f"Disk {storage.mountpoint}",
                    f"{_fmt(storage.available_bytes)} free / {_fmt(storage.total_bytes)} total",
                )
            )
        content.mount(SummaryCard("Memory and storage", mem_rows))

        if profile.gpus:
            gpu = profile.gpus[0]
            content.mount(
                SummaryCard(
                    "GPU",
                    [
                        ("Name", gpu.name),
                        ("VRAM total", _fmt(gpu.vram_total_bytes)),
                        ("VRAM available", _fmt(gpu.vram_available_bytes)),
                        ("Driver", gpu.driver_version or "unknown"),
                        (
                            "CUDA",
                            f"Yes ({gpu.cuda_version})" if gpu.cuda_version else "unknown",
                        ),
                    ],
                )
            )
        else:
            content.mount(
                SummaryCard("GPU", [("Status", "no NVIDIA GPU detected")])
            )

        if profile.warnings:
            content.mount(WarningsPanel(profile.warnings))

    def _app(self) -> LocalAiCheckApp:
        from local_ai_check.tui.app import LocalAiCheckApp

        assert isinstance(self.app, LocalAiCheckApp)
        return self.app


def _int(value: int | None) -> str:
    return str(value) if value is not None else "unknown"


def _fmt(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
