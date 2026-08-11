from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class SummaryCard(Vertical):
    """A titled panel with key/value rows rendered as a two-column display.

    ``plain`` drops the box and leaves a heading plus rows on the page. The
    guided flow uses it so that a result reads as content rather than as one
    more card; the advanced tools keep the boxed default.
    """

    def __init__(
        self,
        title: str,
        rows: Iterable[tuple[str, str]],
        *,
        plain: bool = False,
    ) -> None:
        super().__init__(classes="section" if plain else "card")
        self._title = title
        self._rows = list(rows)
        self._plain = plain

    def compose(self) -> ComposeResult:
        yield Static(
            self._title,
            classes="section-title" if self._plain else "card-title",
        )
        if not self._rows:
            yield Static("no data", classes="text-muted")
            return
        key_width = max((len(k) for k, _ in self._rows), default=0) + 2
        for key, value in self._rows:
            yield Static(f"[bold]{key.ljust(key_width)}[/bold] {value}")
