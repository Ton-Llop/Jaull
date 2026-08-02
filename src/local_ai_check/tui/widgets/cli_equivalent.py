from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class CliEquivalent(Vertical):
    """Shows the equivalent CLI invocation for an action taken from the TUI."""

    DEFAULT_CLASSES = "cli-hint"

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self) -> ComposeResult:
        yield Static("Equivalent CLI", classes="cli-hint-label")
        yield Static(f"$ {self._command}")
