from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from jaull.workflow.models import WorkflowStep

# Only the steps the user actually passes through; terminal states are excluded
# because they are not positions on the path.
_ORDER: tuple[tuple[WorkflowStep, str], ...] = (
    (WorkflowStep.HARDWARE_SCAN, "Hardware"),
    (WorkflowStep.REQUIREMENTS, "Your needs"),
    (WorkflowStep.CANDIDATE_DISCOVERY, "Search"),
    (WorkflowStep.RANKING, "Results"),
)

_DONE = "#b3bac6"
_CURRENT = "#3a7bd5"
_PENDING = "#8b95a5"
_JOIN = "#2b3748"


class WorkflowHeader(Vertical):
    """Where the user is in the guided flow, as a heading rather than a panel.

    Three lines and a hairline: the step title with its position, one line of
    orientation, and a breadcrumb where the current step is the only thing
    wearing the accent colour.
    """

    DEFAULT_CLASSES = "workflow-header"

    def __init__(self, current: WorkflowStep, title: str, subtitle: str = "") -> None:
        super().__init__()
        self._current = current
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(self._title, classes="workflow-title")
            position = self._position()
            if position is not None:
                yield Static(
                    f"Step {position} of {len(_ORDER)}",
                    classes="workflow-step-count",
                )
        if self._subtitle:
            yield Static(self._subtitle, classes="workflow-subtitle")
        yield Static(self._breadcrumb(), classes="workflow-breadcrumb")

    def _position(self) -> int | None:
        for index, (step, _) in enumerate(_ORDER, start=1):
            if step is self._current:
                return index
        return None

    def _breadcrumb(self) -> str:
        position = self._position() or 0
        parts: list[str] = []
        for index, (_, label) in enumerate(_ORDER, start=1):
            if index == position:
                parts.append(f"[b {_CURRENT}]{label}[/]")
            elif index < position:
                parts.append(f"[{_DONE}]{label}[/]")
            else:
                parts.append(f"[{_PENDING}]{label}[/]")
        return f"[{_JOIN}] ─── [/]".join(parts)


__all__ = ["WorkflowHeader"]
