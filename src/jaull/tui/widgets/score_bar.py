from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jaull.recommendation.models import ScoreBreakdown
from jaull.tui.widgets.compact_score_strip import bar_markup

_BAR_WIDTH = 24

_LABELS: tuple[tuple[str, str], ...] = (
    ("hardware_fit", "Hardware fit"),
    ("task_match", "Task match"),
    ("language_match", "Language"),
    ("license", "License"),
    ("metadata_quality", "Metadata"),
    ("popularity", "Popularity"),
)


class ScoreBar(Vertical):
    """The composite score with each weighted component shown as a bar."""

    DEFAULT_CLASSES = "section"

    def __init__(self, breakdown: ScoreBreakdown) -> None:
        super().__init__()
        self._breakdown = breakdown

    def compose(self) -> ComposeResult:
        yield Static("Score breakdown", classes="section-title")
        yield Static(
            f"[bold]{self._breakdown.out_of_100}/100[/bold] overall",
            classes="score-total",
        )
        values = self._breakdown.model_dump()
        for key, label in _LABELS:
            value = float(values.get(key, 0.0))
            weight = self._breakdown.weights.get(key, 0.0)
            yield Static(
                f"{label.ljust(14)} {bar_markup(value, _BAR_WIDTH)} "
                f"{value * 100:3.0f}%  "
                f"[dim](weight {weight * 100:.0f}%)[/dim]",
                classes="score-row",
            )


__all__ = ["ScoreBar"]
