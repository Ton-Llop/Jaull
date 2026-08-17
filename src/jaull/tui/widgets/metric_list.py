"""Aligned label/value rows.

The app used to render every key/value pair as a single ``Static`` holding
``f"[bold]{key.ljust(width)}[/bold] {value}"``. That lines up only while every
row is in the same widget and every glyph is one cell wide, and it gives the
value no way to sit against the right edge. Here the two halves are separate
widgets in a horizontal row, so a column of numbers lines up because the
layout says so.

Each half stays a ``Static``: several tests read a screen by walking
``query(Static)``, and text that moves into a table or a log becomes invisible
to them even though the user can still see it.
"""

from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

# How much evidence stands behind a value. `measured` outranks `estimated`
# visually because it was observed on this machine rather than predicted.
Emphasis = str


class MetricRow(Horizontal):
    """One label/value pair, optionally preceded by a bar."""

    DEFAULT_CLASSES = "metric"

    def __init__(
        self,
        label: str,
        value: str,
        *,
        emphasis: Emphasis | None = None,
        bar: str | None = None,
        value_classes: str = "",
    ) -> None:
        super().__init__()
        self._label = label
        self._value = value
        self._emphasis = emphasis
        self._bar = bar
        self._value_classes = value_classes

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="metric-label")
        if self._bar is not None:
            # Markup, so the filled and track halves can differ in colour.
            yield Static(self._bar, classes="metric-bar", markup=True)
        classes = ["metric-value"]
        if self._emphasis:
            classes.append(f"-{self._emphasis}")
        if self._value_classes:
            classes.extend(self._value_classes.split())
        yield Static(self._value, classes=" ".join(classes))


class MetricList(Vertical):
    """A titled block of :class:`MetricRow`s, or just the rows."""

    DEFAULT_CLASSES = "section"

    def __init__(
        self,
        title: str | None,
        rows: Iterable[tuple[str, str]],
        *,
        emphasis: Emphasis | None = None,
        empty_label: str = "no data",
    ) -> None:
        super().__init__()
        self._title = title
        self._rows = list(rows)
        self._emphasis = emphasis
        self._empty_label = empty_label

    def compose(self) -> ComposeResult:
        if self._title:
            yield Static(self._title, classes="section-title")
        if not self._rows:
            yield Static(self._empty_label, classes="text-muted")
            return
        for label, value in self._rows:
            yield MetricRow(label, value, emphasis=self._emphasis)


__all__ = ["MetricList", "MetricRow"]
