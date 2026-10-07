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
        id: str | None = None,
    ) -> None:
        super().__init__(
            *(MetricRow(label, value) for label, value in rows),
            *extra,
            title=title,
            collapsed=collapsed,
            id=id,
        )


__all__ = ["TechnicalDetails"]
