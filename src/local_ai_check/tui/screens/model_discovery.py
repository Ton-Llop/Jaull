from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from local_ai_check.tui.widgets.progress_step import ProgressStepList
from local_ai_check.tui.widgets.warnings_panel import WarningsPanel
from local_ai_check.tui.widgets.workflow_header import WorkflowHeader
from local_ai_check.workflow import orchestrator
from local_ai_check.workflow.models import (
    UserAnswers,
    WorkflowProgress,
    WorkflowStep,
)
from local_ai_check.workflow.progress import DISCOVERY_STEPS, initial_progress
from local_ai_check.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from local_ai_check.tui.app import LocalAiCheckApp


class ModelDiscoveryScreen(Screen[None]):
    """Step 3: search, inspect and rank, without freezing the interface.

    The whole pipeline runs in a thread worker and reports back through
    `call_from_thread`, so keystrokes — including Cancel — stay responsive
    while network calls are in flight.
    """

    BINDINGS = [("escape", "cancel", "Cancel"), ("q", "quit", "Quit")]

    def __init__(self, answers: UserAnswers) -> None:
        super().__init__()
        self._answers = answers
        self._cancel = threading.Event()

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
        self.run_worker(self._run, thread=True)

    def _run(self) -> None:
        app = self._app()

        def report(progress: WorkflowProgress) -> None:
            app.call_from_thread(self._update_progress, progress)

        hardware = app.hardware_profile
        if hardware is None:
            hardware = orchestrator.scan_hardware(app.services)
            app.hardware_profile = hardware

        state = orchestrator.run_workflow(
            answers=self._answers,
            hardware=hardware,
            services=app.services,
            on_progress=report,
            is_cancelled=self._cancel.is_set,
        )
        app.call_from_thread(self._finish, state)

    def _update_progress(self, progress: WorkflowProgress) -> None:
        self.query_one(ProgressStepList).update_progress(progress)

    def _finish(self, state: RecommendationWorkflowState) -> None:
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
            self.run_worker(self._run, thread=True)
        elif button_id == "discovery-restart":
            app.restart_workflow()
        elif button_id == "discovery-advanced":
            app.goto_advanced_tools()

    def action_cancel(self) -> None:
        """Signal the worker to stop at its next checkpoint."""
        self._cancel.set()
        self.query_one("#discovery-cancel", Button).disabled = True

    def _app(self) -> LocalAiCheckApp:
        from local_ai_check.tui.app import LocalAiCheckApp

        assert isinstance(self.app, LocalAiCheckApp)
        return self.app


__all__ = ["ModelDiscoveryScreen"]
