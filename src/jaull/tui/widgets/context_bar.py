"""The two-line block every screen opens with.

Four screens each grew their own header — ``.run-header``, ``#validation-header``,
``#benchmark-header`` and ``WorkflowHeader`` — saying the same three things in
three different shapes. One bar says them once: what you are looking at, where
you are, and which machine or execution path it applies to.

The status line is deliberately one line. Repeating the full hardware profile
on every screen is what pushed the actual content below the fold.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static


class ContextBar(Vertical):
    """Title (plus an optional right-aligned aside) over a status line."""

    DEFAULT_CLASSES = "context-bar"

    def __init__(
        self,
        title: str,
        status: str = "",
        *,
        aside: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._title = title
        self._status = status
        self._aside = aside

    def compose(self) -> ComposeResult:
        with Horizontal(classes="context-title-row"):
            yield Static(self._title, classes="context-title")
            if self._aside:
                yield Static(self._aside, classes="context-aside")
        if self._status:
            yield Static(self._status, classes="context-status")

    def update_status(self, status: str) -> None:
        """Rewrite the status line in place, without rebuilding the bar."""
        self._status = status
        for widget in self.query(".context-status").results(Static):
            widget.update(status)
            return
        # The bar was built without a status line; mount one now.
        self.mount(Static(status, classes="context-status"))


__all__ = ["ContextBar"]
