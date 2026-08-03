from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class SummaryCard(Vertical):
    """A titled panel with key/value rows rendered as a two-column display."""

    DEFAULT_CLASSES = "card"

    def __init__(self, title: str, rows: Iterable[tuple[str, str]]) -> None:
        super().__init__()
        self._title = title
        self._rows = list(rows)

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="card-title")
        if not self._rows:
            yield Static("[dim]no data[/dim]")
            return
        key_width = max((len(k) for k, _ in self._rows), default=0) + 2
        for key, value in self._rows:
            yield Static(f"[bold]{key.ljust(key_width)}[/bold] {value}")
