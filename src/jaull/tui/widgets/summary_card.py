from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jaull.tui.widgets.metric_list import MetricRow


class SummaryCard(Vertical):
    """A titled block of key/value rows.

    Rows are :class:`MetricRow`s, so the values line up against the right edge
    instead of relying on ``ljust`` padding inside a single string — which only
    ever lined up while every row lived in the same widget.

    This used to take a ``plain`` flag that chose between a bordered card and a
    heading-plus-rows section, because the guided flow and the advanced tools
    were two different visual languages. They are one now, so the flag is gone
    and every caller gets the section.
    """

    def __init__(self, title: str, rows: Iterable[tuple[str, str]]) -> None:
        super().__init__(classes="section")
        self._title = title
        self._rows = list(rows)

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="section-title")
        if not self._rows:
            yield Static("no data", classes="text-muted")
            return
        for key, value in self._rows:
            yield MetricRow(key, value)
