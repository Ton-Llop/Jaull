"""The body of the Context tab: the memory a model needs at each context.

Built from the pieces the Results screen already uses: a ``MetricRow`` per context with the
house bar beside the number, and the coloured ``AssessmentBadge`` for the verdict. The bar is
scaled to the largest total in the sweep and never replaces the figure. The KV cache and the
confidence live in each row's tooltip, which is where this screen keeps its detail.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static

from jaull.estimator.context_fit import ContextPoint, ContextSweep
from jaull.presentation.plan_labels import format_gib
from jaull.tui.widgets.assessment_badge import AssessmentBadge
from jaull.tui.widgets.bars import ratio_bar
from jaull.tui.widgets.metric_list import MetricRow

_BAR_WIDTH = 14
_LARGEST_MARK = " ◂"


class _ContextPointRow(Horizontal):
    """One context: its size, a bar for the total, the total, and the verdict."""

    DEFAULT_CLASSES = "metric context-point"

    def __init__(self, point: ContextPoint, peak: int | None, *, largest: bool) -> None:
        super().__init__()
        self._point = point
        self._peak = peak
        self._largest = largest
        self.set_class(largest, "-largest")
        self.tooltip = (
            f"Total {format_gib(point.total_bytes)} · KV cache {format_gib(point.kv_cache_bytes)}"
            f" · confidence {point.confidence.value}"
        )

    def compose(self) -> ComposeResult:
        point = self._point
        mark = _LARGEST_MARK if self._largest else ""
        yield Static(f"{point.context_length:,}{mark}", classes="metric-label", markup=False)
        yield Static(
            ratio_bar(point.total_bytes, self._peak, _BAR_WIDTH),
            classes="metric-bar",
            markup=True,
        )
        yield Static(format_gib(point.total_bytes), classes="metric-value", markup=False)
        yield AssessmentBadge(point.status)


class ContextSweepView(Vertical):
    """Empty until asked; then loading, unavailable, or the sweep itself."""

    DEFAULT_CLASSES = "section"

    def show_loading(self) -> None:
        self._replace(
            Static("Computing the memory at each context…", classes="text-muted context-state")
        )

    def show_unavailable(self, reason: str) -> None:
        self._replace(Static(reason, classes="warning-line context-state", markup=False))

    def show_sweep(self, sweep: ContextSweep) -> None:
        totals = [p.total_bytes for p in sweep.points if p.total_bytes is not None]
        peak = max(totals) if totals else None
        if sweep.largest_fitting is None:
            headline = Static(
                "Largest context that fits: none in this sweep",
                classes="context-headline fit-bad",
                markup=False,
            )
        else:
            headline = Static(
                f"Largest context that fits: {sweep.largest_fitting:,} tokens",
                classes="context-headline fit-good",
                markup=False,
            )
        widgets: list[Widget] = [
            headline,
            MetricRow(
                "Confidence of the estimate",
                sweep.weakest_confidence.value,
                emphasis="estimated",
            ),
            Static("Memory at each context", classes="section-title context-title"),
        ]
        widgets.extend(
            _ContextPointRow(
                point, peak, largest=point.context_length == sweep.largest_fitting
            )
            for point in sweep.points
        )
        widgets.extend(
            Static(note, classes="text-muted context-note", markup=False) for note in sweep.notes
        )
        self._replace(*widgets)

    def _replace(self, *widgets: Widget) -> None:
        # Children are told apart by class, never by id: a replacement mounted before the old
        # children are gone must not collide with them.
        self.remove_children()
        self.mount(*widgets)
