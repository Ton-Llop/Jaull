"""Collapsed detail block.

`PRODUCT.md` asks for two things at once: primary screens that explain model,
runtime, artifact, readiness and evidence, and expert power that is never
removed. A collapsed section satisfies both — the build hashes, methodology
ids, device ids and paths stay one keystroke away instead of competing with
the result for the top of the screen.

Collapsed content still lives in the DOM, so ``query(Static)`` finds it and the
screen-reading tests keep working.
"""

from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Collapsible

from jaull.tui.widgets.metric_list import MetricRow


class TechnicalDetails(Collapsible):
    """A collapsed block of raw fields, or of arbitrary widgets."""

    DEFAULT_CLASSES = "technical-details"

    def __init__(
        self,
        rows: Iterable[tuple[str, str]] = (),
        *,
        title: str = "Technical details",
        extra: Iterable[Widget] = (),
        collapsed: bool = True,
    ) -> None:
        super().__init__(title=title, collapsed=collapsed)
        self._rows = list(rows)
        self._extra = list(extra)

    def compose(self) -> ComposeResult:
        # Collapsible.compose builds the title bar and the contents container;
        # composing our own children on top of it would leave them unparented.
        yield from super().compose()

    def on_mount(self) -> None:
        contents = self.query_one(Collapsible.Contents)
        contents.mount_all(
            [MetricRow(label, value) for label, value in self._rows] + self._extra
        )


__all__ = ["TechnicalDetails"]
