"""Compact score visualisation used inside a RecommendationCard.

Shows the total plus the three sub-scores that most influence the ranking
today (memory fit, capability, task match) so the user can see at a glance
why a card scored what it scored — without having to open the "Technical
details" pane.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jaull.recommendation.models import ScoreBreakdown

# A two-tone rule reads as a bar at any width, where the shade-block pair
# (█ / ░) turns into visual noise once several rows sit on top of each other.
_FILLED = "#3a7bd5"
_TRACK = "#2b3748"

_BAR_WIDTH = 14


def bar_markup(value: float, width: int) -> str:
    """A `width`-column bar for a 0..1 value, as console markup."""
    filled = max(0, min(width, round(value * width)))
    return f"[{_FILLED}]{'━' * filled}[/][{_TRACK}]{'━' * (width - filled)}[/]"

_ROWS: tuple[tuple[str, str], ...] = (
    ("memory_fit", "Memory"),
    ("capability", "Capability"),
    ("task_match", "Task"),
)


class CompactScoreStrip(Vertical):
    DEFAULT_CLASSES = "compact-score-strip"

    def __init__(self, breakdown: ScoreBreakdown) -> None:
        super().__init__()
        self._breakdown = breakdown

    def compose(self) -> ComposeResult:
        yield Static(
            f"[b]{self._breakdown.out_of_100}/100[/b] overall score",
            classes="score-total",
        )
        values = self._breakdown.model_dump()
        for key, label in _ROWS:
            value = float(values.get(key, 0.0))
            yield Static(
                f"[dim]{label.ljust(10)}[/dim] {bar_markup(value, _BAR_WIDTH)} "
                f"{value * 100:3.0f}%"
            )
        if self._breakdown.hard_penalty < 1.0:
            yield Static(
                f"[yellow]Hard requirements penalty x{self._breakdown.hard_penalty:.2f}[/yellow]",
                classes="score-penalty",
            )


__all__ = ["CompactScoreStrip", "bar_markup"]
