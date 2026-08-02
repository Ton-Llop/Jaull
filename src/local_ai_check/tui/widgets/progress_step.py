from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from local_ai_check.workflow.models import StepStatus, WorkflowProgress

# Sober markers rather than emoji: they render identically on every terminal
# and keep the interface professional.
_MARKERS: dict[StepStatus, tuple[str, str]] = {
    StepStatus.DONE: ("✓", "step-done"),
    StepStatus.RUNNING: ("●", "step-running"),
    StepStatus.PENDING: ("○", "step-pending"),
    StepStatus.FAILED: ("✗", "step-failed"),
    StepStatus.SKIPPED: ("-", "step-pending"),
}


class ProgressStepList(Vertical):
    """Renders a :class:`WorkflowProgress` as a checklist of real operations."""

    DEFAULT_CLASSES = "card"

    def __init__(self, title: str, progress: WorkflowProgress) -> None:
        super().__init__()
        self._title = title
        self._progress = progress

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="card-title")
        yield Vertical(id="progress-steps")

    def on_mount(self) -> None:
        self._render_steps()

    def update_progress(self, progress: WorkflowProgress) -> None:
        self._progress = progress
        self._render_steps()

    def _render_steps(self) -> None:
        try:
            container = self.query_one("#progress-steps", Vertical)
        except Exception:
            # Called before mount; on_mount will render the first snapshot.
            return
        container.remove_children()
        for step in self._progress.steps:
            marker, css_class = _MARKERS[step.status]
            label = step.label
            if step.detail:
                label = f"{label} — {step.detail}"
            container.mount(Static(f"{marker} {label}", classes=css_class))


__all__ = ["ProgressStepList"]
