from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from jaull.presentation.plan_labels import hardware_summary
from jaull.tui import palette
from jaull.workflow.models import WorkflowStep

# Only the steps the user actually passes through; terminal states are excluded
# because they are not positions on the path.
_ORDER: tuple[tuple[WorkflowStep, str], ...] = (
    (WorkflowStep.HARDWARE_SCAN, "Hardware"),
    (WorkflowStep.REQUIREMENTS, "Your needs"),
    (WorkflowStep.CANDIDATE_DISCOVERY, "Search"),
    (WorkflowStep.RANKING, "Results"),
)

_DONE = palette.INK_2
_CURRENT = palette.ACCENT
_PENDING = palette.INK_3
_JOIN = palette.LINE_2


class WorkflowHeader(Vertical):
    """Where the user is in the guided flow, and what machine they are on.

    Two lines and a hairline. It used to be four — title, step count, a line of
    orientation copy, and the breadcrumb — which cost five rows on every screen
    and still never said what hardware the answers applied to. The breadcrumb
    carries the position, so the separate "Step 3 of 4" is redundant, and the
    row it frees goes to the machine.

    The subtitle is accepted and ignored on purpose: the call sites still pass
    the orientation copy they always did, and dropping it is a presentation
    decision this widget is entitled to make in one place rather than in six.
    """

    DEFAULT_CLASSES = "workflow-header"

    def __init__(self, current: WorkflowStep, title: str, subtitle: str = "") -> None:
        super().__init__()
        self._current = current
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        with Horizontal(classes="context-title-row"):
            yield Static(self._title, classes="workflow-title")
            yield Static("", classes="workflow-machine", id="workflow-machine")
        yield Static(self._breadcrumb(), classes="workflow-breadcrumb")

    def on_mount(self) -> None:
        # The profile only exists once the hardware scan has run, so it is read
        # here rather than passed through every screen constructor.
        profile = getattr(self.app, "hardware_profile", None)
        summary = hardware_summary(profile)
        machine = self.query_one("#workflow-machine", Static)
        machine.update(summary)
        machine.display = bool(summary)

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
        return f"[{_JOIN}] ─ [/]".join(parts)


__all__ = ["WorkflowHeader"]
