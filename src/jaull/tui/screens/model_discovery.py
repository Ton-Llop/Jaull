from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from jaull.domain.requirements import UserAnswers
from jaull.tui.widgets.progress_step import ProgressStepList
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowProgress, WorkflowStep
from jaull.workflow.progress import DISCOVERY_STEPS, initial_progress
from jaull.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class _DiscoveryProgress(Message):
    def __init__(self, progress: WorkflowProgress) -> None:
        super().__init__()
        self.progress = progress


class _DiscoveryFinished(Message):
    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self.state = state


class ModelDiscoveryScreen(Screen[None]):
    """Step 3: search, inspect and rank, without freezing the interface.

    The whole pipeline runs outside the event loop and reports back with
    non-blocking Textual messages, so keystrokes — including Cancel — stay
    responsive while network calls are in flight.
    """

    BINDINGS = [("escape", "cancel", "Cancel"), ("q", "quit", "Quit")]

    def __init__(self, answers: UserAnswers) -> None:
        super().__init__()
        self._answers = answers
        self._cancel = threading.Event()
        self._discovery_closing = threading.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield WorkflowHeader(
            WorkflowStep.CANDIDATE_DISCOVERY,
            "Searching Hugging Face",
            "Only public metadata is read. No model weights are downloaded.",
        )
        with VerticalScroll():
            yield ProgressStepList("Progress", initial_progress(DISCOVERY_STEPS))
            yield Vertical(id="discovery-messages")
            with Horizontal(id="discovery-actions"):
                yield Button("Cancel", id="discovery-cancel")
        yield Footer()

    def on_mount(self) -> None:
        self._start_discovery()

    def on_unmount(self) -> None:
        self._discovery_closing.set()
        self._shutdown_discovery_executor()

    def _start_discovery(self) -> None:
        self._discovery_closing.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-discovery",
        )
        self._future = self._executor.submit(self._run, self._app().advisor)

    def _shutdown_discovery_executor(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _run(self, advisor: AdvisorService) -> None:
        def report(progress: WorkflowProgress) -> None:
            if self._discovery_closing.is_set():
                return
            self.post_message(_DiscoveryProgress(progress))

        app = self._app()
        hardware = app.hardware_profile
        state = advisor.recommend(
            self._answers,
            hardware=hardware,
            on_progress=report,
            is_cancelled=self._cancel.is_set,
        )
        if hardware is None:
            app.hardware_profile = state.hardware
        if self._discovery_closing.is_set():
            return
        self.post_message(_DiscoveryFinished(state))

    @on(_DiscoveryProgress)
    def _update_progress_message(self, message: _DiscoveryProgress) -> None:
        if self._discovery_closing.is_set():
            return
        self._update_progress(message.progress)

    @on(_DiscoveryFinished)
    def _finish_message(self, message: _DiscoveryFinished) -> None:
        if self._discovery_closing.is_set():
            return
        self._finish(message.state)

    def _update_progress(self, progress: WorkflowProgress) -> None:
        self.query_one(ProgressStepList).update_progress(progress)

    def _finish(self, state: RecommendationWorkflowState) -> None:
        self._shutdown_discovery_executor()
        app = self._app()
        app.workflow_state = state

        if self._cancel.is_set():
            self._show_cancelled()
            return
        if state.failed:
            self._show_error(state)
            return
        app.show_recommendations(state)

    def _show_cancelled(self) -> None:
        messages = self.query_one("#discovery-messages", Vertical)
        messages.remove_children()
        messages.mount(Static("Search cancelled.", classes="card-title"))
        self._replace_actions(
            [
                Button("Start again", id="discovery-restart", classes="-primary"),
                Button("Advanced tools", id="discovery-advanced"),
            ]
        )

    def _show_error(self, state: RecommendationWorkflowState) -> None:
        """A transport failure is recoverable: offer a retry, never close the app."""
        messages = self.query_one("#discovery-messages", Vertical)
        messages.remove_children()
        messages.mount(Static("Search could not be completed", classes="card-title"))
        messages.mount(WarningsPanel(state.errors or ["Unknown error."]))
        messages.mount(
            Static(
                "Hugging Face may be unreachable or rate limiting requests. "
                "You can retry, or use the advanced tools offline.",
                classes="text-muted",
            )
        )
        self._replace_actions(
            [
                Button("Retry", id="discovery-retry", classes="-primary"),
                Button("Advanced tools", id="discovery-advanced"),
                Button("Start again", id="discovery-restart"),
            ]
        )

    def _replace_actions(self, buttons: list[Button]) -> None:
        actions = self.query_one("#discovery-actions", Horizontal)
        actions.remove_children()
        for button in buttons:
            actions.mount(button)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self._app()
        button_id = event.button.id
        if button_id == "discovery-cancel":
            self.action_cancel()
        elif button_id == "discovery-retry":
            self._cancel.clear()
            self.query_one(ProgressStepList).update_progress(
                initial_progress(DISCOVERY_STEPS)
            )
            self.query_one("#discovery-messages", Vertical).remove_children()
            self._replace_actions([Button("Cancel", id="discovery-cancel")])
            self._start_discovery()
        elif button_id == "discovery-restart":
            app.restart_workflow()
        elif button_id == "discovery-advanced":
            app.goto_advanced_tools()

    def action_cancel(self) -> None:
        """Signal the worker to stop at its next checkpoint."""
        self._cancel.set()
        self.query_one("#discovery-cancel", Button).disabled = True

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


__all__ = ["ModelDiscoveryScreen"]
