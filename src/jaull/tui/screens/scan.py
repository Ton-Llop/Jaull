from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, LoadingIndicator

from jaull.domain.hardware import HardwareProfile
from jaull.tui.widgets.banner import Banner
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class _ScanFinished(Message):
    def __init__(self, profile: HardwareProfile) -> None:
        super().__init__()
        self.profile = profile


class ScanScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._scan_closing = Event()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner("Local hardware scan", "Detecting CPU, memory, storage and GPU.")
        yield LoadingIndicator(id="scan-loading")
        yield VerticalScroll(Vertical(id="scan-content"))
        yield Footer()

    def on_mount(self) -> None:
        self._scan_closing.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-scan",
        )
        self._future = self._executor.submit(self._probe, self._app().advisor)

    def on_unmount(self) -> None:
        self._scan_closing.set()
        self._shutdown_scan_executor()

    def _shutdown_scan_executor(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _probe(self, advisor: AdvisorService) -> None:
        profile = advisor.scan_hardware()
        if self._scan_closing.is_set():
            return
        self.post_message(_ScanFinished(profile))

    @on(_ScanFinished)
    def _populate_message(self, message: _ScanFinished) -> None:
        if self._scan_closing.is_set():
            return
        self._populate(message.profile)

    def _populate(self, profile: HardwareProfile) -> None:
        self._shutdown_scan_executor()
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

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
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
