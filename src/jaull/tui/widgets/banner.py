from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jaull.tui import palette


class Banner(Vertical):
    DEFAULT_CLASSES = "banner"

    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        yield Static(f"[{palette.ACCENT}]JAULL[/]  |  {self._title}", classes="banner-title")
        yield Static(self._subtitle, classes="banner-subtitle")
