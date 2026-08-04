from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ProgressBar, Static

from jaull.domain.hardware import HardwareProfile
from jaull.tui.widgets.progress_step import ProgressStepList
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowProgress, WorkflowStep
from jaull.workflow.progress import HARDWARE_STEPS, initial_progress

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp

# Pacing between hardware probes so the checklist actually reads as a
# checklist. Probes finish in microseconds on modern machines; without a
# floor the whole panel goes green in one frame and the user misses the
# feedback that the tool actually inspected the system.
_STEP_PACING_SECONDS = 0.35

_TOTAL_STEPS = float(len(HARDWARE_STEPS))


class HardwareAnalysisScreen(Screen[None]):
    """Step 1 of the guided flow: detect the machine, showing real progress.

    Each checklist line turns green when its probe actually returns, but a
    small pacing floor keeps the animation legible. When the scan completes
    the loading card is replaced by the summary card **in place**, so the
    user never has to scroll to see the result.
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
            yield Vertical(id="hardware-content")
        yield Footer()

    def on_mount(self) -> None:
        content = self.query_one("#hardware-content", Vertical)
        content.mount(_LoadingCard(_TOTAL_STEPS))
        # Threaded worker: the probes are blocking (psutil, NVML), so running
        # them here keeps the event loop free to repaint the progress bar.
        self.run_worker(self._scan, thread=True)

    def _scan(self) -> None:
        app = self._app()

        def report(progress: WorkflowProgress) -> None:
            # Sleep in the worker thread — never on the UI thread — so the
            # progress bar can advance in visible increments.
            time.sleep(_STEP_PACING_SECONDS)
            app.call_from_thread(self._update_progress, progress)

        profile = app.advisor.scan_hardware(on_progress=report)
        app.call_from_thread(self._finish, profile)

    def _update_progress(self, progress: WorkflowProgress) -> None:
        card = self.query_one(_LoadingCard)
        card.update_progress(progress)

    def _finish(self, profile: HardwareProfile) -> None:
        app = self._app()
        app.hardware_profile = profile

        # Replace the loading card with the results in the same slot: no
        # scrolling required to see what came back.
        content = self.query_one("#hardware-content", Vertical)
        content.remove_children()
        content.mount(SummaryCard("Detected hardware", _summary_rows(profile)))

        if profile.warnings:
            # A missing NVIDIA GPU is a warning, never a failure: CPU-only
            # machines are a supported target.
            content.mount(WarningsPanel(profile.warnings))

        content.mount(Static("", classes="text-muted"))
        content.mount(Button("Continue", id="hw-continue", classes="-primary"))
        self.query_one("#hw-continue", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hw-continue":
            self._app().goto_requirements()

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class _LoadingCard(Vertical):
    """Single card that holds a ProgressBar + the step checklist together."""

    DEFAULT_CLASSES = "card"

    def __init__(self, total_steps: float) -> None:
        super().__init__()
        self._total = total_steps
        self._done = 0

    def compose(self) -> ComposeResult:
        yield Static("Scanning hardware…", classes="card-title")
        yield ProgressBar(
            total=self._total,
            show_eta=False,
            id="hardware-progress",
        )
        yield ProgressStepList(
            "Detection steps", initial_progress(HARDWARE_STEPS)
        )

    def update_progress(self, progress: WorkflowProgress) -> None:
        self.query_one(ProgressStepList).update_progress(progress)
        done = sum(1 for step in progress.steps if step.status.name == "DONE")
        # Advance in absolute terms so an out-of-order update never rewinds.
        self.query_one("#hardware-progress", ProgressBar).update(progress=float(done))


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
