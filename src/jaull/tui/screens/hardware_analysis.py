from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from jaull.domain.hardware import HardwareProfile
from jaull.tui.widgets.progress_step import ProgressStepList
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowProgress, WorkflowStep
from jaull.workflow.progress import HARDWARE_STEPS, initial_progress

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp


class HardwareAnalysisScreen(Screen[None]):
    """Step 1 of the guided flow: detect the machine, showing real progress.

    Each checklist line turns green when its probe actually returns — the
    orchestrator drives it from `detect_hardware`'s step callback, so there is
    no artificial pacing anywhere.
    """

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield WorkflowHeader(
            WorkflowStep.HARDWARE_SCAN,
            "Analyzing your system",
            "Reading the hardware available for local inference.",
        )
        with VerticalScroll():
            yield ProgressStepList(
                "Progress", initial_progress(HARDWARE_STEPS)
            )
            yield Vertical(id="hardware-summary")
        yield Footer()

    def on_mount(self) -> None:
        # Threaded worker: the probes are blocking (psutil, NVML), so running
        # them here keeps the event loop free to repaint the checklist.
        self.run_worker(self._scan, thread=True)

    def _scan(self) -> None:
        app = self._app()

        def report(progress: WorkflowProgress) -> None:
            app.call_from_thread(self._update_progress, progress)

        profile = app.advisor.scan_hardware(on_progress=report)
        app.call_from_thread(self._finish, profile)

    def _update_progress(self, progress: WorkflowProgress) -> None:
        self.query_one(ProgressStepList).update_progress(progress)

    def _finish(self, profile: HardwareProfile) -> None:
        app = self._app()
        app.hardware_profile = profile

        container = self.query_one("#hardware-summary", Vertical)
        container.remove_children()
        container.mount(SummaryCard("Detected hardware", _summary_rows(profile)))

        if profile.warnings:
            # A missing NVIDIA GPU is a warning, never a failure: CPU-only
            # machines are a supported target.
            container.mount(WarningsPanel(profile.warnings))

        container.mount(Static("", classes="text-muted"))
        container.mount(Button("Continue", id="hw-continue", classes="-primary"))
        self.query_one("#hw-continue", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hw-continue":
            self._app().goto_requirements()

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


def _summary_rows(profile: HardwareProfile) -> list[tuple[str, str]]:
    rows = [
        ("CPU", profile.cpu.model or "unknown"),
        ("RAM", _fmt(profile.memory.total_bytes)),
    ]
    if profile.gpus:
        gpu = profile.gpus[0]
        rows.append(("GPU", gpu.name))
        rows.append(("VRAM", _fmt(gpu.vram_total_bytes)))
    else:
        rows.append(("GPU", "none detected — CPU inference only"))
    rows.append(("Platform", f"{profile.os} ({profile.arch})"))
    return rows


def _fmt(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


__all__ = ["HardwareAnalysisScreen"]
