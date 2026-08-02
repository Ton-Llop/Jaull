from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from local_ai_check.workflow.models import WorkflowStep

# Only the steps the user actually passes through; terminal states are excluded
# because they are not positions on the path.
_ORDER: tuple[tuple[WorkflowStep, str], ...] = (
    (WorkflowStep.HARDWARE_SCAN, "Hardware"),
    (WorkflowStep.REQUIREMENTS, "Your needs"),
    (WorkflowStep.CANDIDATE_DISCOVERY, "Search"),
    (WorkflowStep.RANKING, "Results"),
)


class WorkflowHeader(Vertical):
    """A breadcrumb showing where the user is in the guided flow."""

    DEFAULT_CLASSES = "workflow-header"

    def __init__(self, current: WorkflowStep, title: str, subtitle: str = "") -> None:
        super().__init__()
        self._current = current
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="banner-title")
        if self._subtitle:
            yield Static(self._subtitle, classes="banner-subtitle")
        yield Static(self._breadcrumb(), classes="workflow-breadcrumb")

    def _breadcrumb(self) -> str:
        parts: list[str] = []
        for step, label in _ORDER:
            if step is self._current:
                parts.append(f"[b]{label}[/b]")
            else:
                parts.append(f"[dim]{label}[/dim]")
        return "  >  ".join(parts)


__all__ = ["WorkflowHeader"]
