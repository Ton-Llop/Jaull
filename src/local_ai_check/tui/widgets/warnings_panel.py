from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class WarningsPanel(Vertical):
    DEFAULT_CLASSES = "warnings"

    def __init__(self, warnings: Iterable[str]) -> None:
        super().__init__()
        self._warnings = list(warnings)

    def compose(self) -> ComposeResult:
        if not self._warnings:
            self.display = False
            return
        yield Static("Warnings", classes="card-title")
        for warning in self._warnings:
            yield Static(f"! {warning}")
