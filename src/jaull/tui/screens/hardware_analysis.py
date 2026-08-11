from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer

from jaull.domain.hardware import HardwareProfile
from jaull.tui.widgets.progress_step import ProgressStepList
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowProgress, WorkflowStep
from jaull.workflow.progress import HARDWARE_STEPS, initial_progress

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class _HardwareProgress(Message):
    def __init__(self, progress: WorkflowProgress) -> None:
        super().__init__()
        self.progress = progress


class _HardwareFinished(Message):
    def __init__(self, profile: HardwareProfile) -> None:
        super().__init__()
        self.profile = profile


class HardwareAnalysisScreen(Screen[None]):
    """Step 1 of the guided flow: detect the machine, showing real progress.

    Each checklist line turns green when its probe actually returns. When the
    scan completes the checklist is replaced by the summary in place, so the
    user never has to scroll to see the result — and the two states do not
    look alike.
    """

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._scan_closing = Event()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.HARDWARE_SCAN,
            "Hardware",
            "Reading the hardware available for local inference.",
        )
        with VerticalScroll():
            yield Vertical(id="hardware-content")
        yield Footer()

    def on_mount(self) -> None:
        self._scan_closing.clear()
        content = self.query_one("#hardware-content", Vertical)
        content.mount(_ScanChecklist())
        # A screen-local executor keeps blocking probes off the event loop
        # without involving asyncio's default executor, which can outlive
        # Textual's test harness during teardown.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-hardware",
        )
        self._future = self._executor.submit(self._scan, self._app().advisor)

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

    def _scan(self, advisor: AdvisorService) -> None:
        def report(progress: WorkflowProgress) -> None:
            if self._scan_closing.is_set():
                return
            self.post_message(_HardwareProgress(progress))

        profile = advisor.scan_hardware(on_progress=report)
        if self._scan_closing.is_set():
            return
        self.post_message(_HardwareFinished(profile))

    @on(_HardwareProgress)
    def _update_progress_message(self, message: _HardwareProgress) -> None:
        if self._scan_closing.is_set():
            return
        self._update_progress(message.progress)

    @on(_HardwareFinished)
    def _finish_message(self, message: _HardwareFinished) -> None:
        if self._scan_closing.is_set():
            return
        self._finish(message.profile)

    def _update_progress(self, progress: WorkflowProgress) -> None:
        checklist = self.query_one(_ScanChecklist)
        checklist.update_progress(progress)

    def _finish(self, profile: HardwareProfile) -> None:
        self._shutdown_scan_executor()
        app = self._app()
        app.hardware_profile = profile

        # Replace the checklist with the result in the same slot: no scrolling
        # required to see what came back, and no leftover progress UI.
        content = self.query_one("#hardware-content", Vertical)
        content.remove_children()
        content.mount(SummaryCard("This machine", _summary_rows(profile), plain=True))

        if profile.warnings:
            # A missing NVIDIA GPU is a warning, never a failure: CPU-only
            # machines are a supported target.
            content.mount(WarningsPanel(profile.warnings))

        actions = Horizontal(classes="actions-right")
        content.mount(actions)
        actions.mount(Button("Continue", id="hw-continue", classes="-primary"))
        self.query_one("#hw-continue", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hw-continue":
            self._app().goto_requirements()

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class _ScanChecklist(Vertical):
    """The live checklist, and nothing else.

    A progress bar alongside the markers said the same thing twice while being
    the heaviest element on the screen; the markers already carry both what is
    happening and how far along it is.
    """

    DEFAULT_CLASSES = "section"

    def compose(self) -> ComposeResult:
        yield ProgressStepList(
            "Analyzing this machine",
            initial_progress(HARDWARE_STEPS),
            plain=True,
        )

    def update_progress(self, progress: WorkflowProgress) -> None:
        self.query_one(ProgressStepList).update_progress(progress)


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
