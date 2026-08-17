from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jaull.tui import palette
from jaull.workflow.models import StepStatus, WorkflowProgress

# Sober markers rather than emoji: they render identically on every terminal
# and keep the interface professional.
_MARKERS: dict[StepStatus, tuple[str, str]] = {
    StepStatus.DONE: ("✓", "step-done"),
    StepStatus.RUNNING: ("●", "step-running"),
    StepStatus.PENDING: ("○", "step-pending"),
    StepStatus.FAILED: ("✗", "step-failed"),
    StepStatus.SKIPPED: ("-", "step-pending"),
}

# So the run logs below wear the same colours as the checklist above without
# every screen owning a copy.
_STEP_COLOURS = {
    "done": palette.OK,
    "running": palette.ACCENT,
    "failed": palette.BAD,
}


def format_step_log(messages: list[str], *, failed: bool = False, limit: int = 8) -> str:
    """Render a running list of completed steps with the checklist's glyphs.

    The validation and benchmark screens used to prefix every line with a bare
    ``-``, so the same operation looked like a checklist during the hardware
    scan and like a plain-text dump five screens later. The last entry is the
    one in flight, which is the only thing the reader is actually waiting on.
    """
    if not messages:
        return ""
    visible = messages[-limit:]
    lines: list[str] = []
    for index, message in enumerate(visible):
        last = index == len(visible) - 1
        if last and failed:
            glyph, colour = "✗", _STEP_COLOURS["failed"]
        elif last:
            glyph, colour = "●", _STEP_COLOURS["running"]
        else:
            glyph, colour = "✓", _STEP_COLOURS["done"]
        lines.append(f"[{colour}]{glyph}[/] {message}")
    return "\n".join(lines)


class ProgressStepList(Vertical):
    """Renders a :class:`WorkflowProgress` as a checklist of real operations.

    During a scan the markers are the content; they never needed a card to sit
    in, which is why the old ``plain`` flag was set at every call site.
    """

    def __init__(self, title: str, progress: WorkflowProgress) -> None:
        super().__init__(classes="section")
        self._title = title
        self._progress = progress

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="section-title")
        yield Static("", id="progress-current", classes="status-line")
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
        current = self.query_one("#progress-current", Static)
        current_label = _current_label(self._progress)
        current.update(current_label)
        current.display = bool(current_label)
        container.remove_children()
        for step in self._progress.steps:
            marker, css_class = _MARKERS[step.status]
            label = step.label
            if step.detail:
                label = f"{label} — {step.detail}"
            container.mount(Static(f"{marker} {label}", classes=css_class))


def _current_label(progress: WorkflowProgress) -> str:
    running = next(
        (step for step in progress.steps if step.status is StepStatus.RUNNING),
        None,
    )
    if running is not None:
        detail = f" — {running.detail}" if running.detail else ""
        return f"Current: {running.label}{detail}"
    failed = next(
        (step for step in progress.steps if step.status is StepStatus.FAILED),
        None,
    )
    if failed is not None:
        detail = f" — {failed.detail}" if failed.detail else ""
        return f"Stopped: {failed.label}{detail}"
    return ""


__all__ = ["ProgressStepList", "format_step_log"]
