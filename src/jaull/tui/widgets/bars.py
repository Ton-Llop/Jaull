"""The one bar primitive in the app.

A bar is console markup rather than a styled widget, because it has to sit
inside a line of text next to the number it describes, so its colours come
from :mod:`jaull.tui.palette` rather than from the stylesheet.

A bar never replaces a value. It makes a column of numbers comparable at a
glance; the exact figure always stays on the same row.
"""

from __future__ import annotations

from jaull.tui.palette import ACCENT, LINE_2

_FILLED = ACCENT
_TRACK = LINE_2

# The filled and empty halves use different glyphs, not just different colours.
# A bar drawn in one glyph reads as a solid full-width rule to anyone on a
# monochrome terminal or with reduced colour vision — which is to say the value
# would be encoded in colour alone. Heavy against light keeps the proportion
# legible with the colour switched off, and the pair stays quieter than █/░,
# which turns into noise once several rows sit on top of each other.
_FILLED_GLYPH = "━"
_TRACK_GLYPH = "─"


def bar_markup(value: float, width: int) -> str:
    """A `width`-column bar for a 0..1 value, as console markup."""
    filled = max(0, min(width, round(value * width)))
    return (
        f"[{_FILLED}]{_FILLED_GLYPH * filled}[/]"
        f"[{_TRACK}]{_TRACK_GLYPH * (width - filled)}[/]"
    )


def ratio_bar(value: float | None, maximum: float | None, width: int) -> str:
    """A bar for `value` relative to the largest value in its group.

    Returns an empty track when there is nothing to compare against, so a
    single unmeasured row never renders as a full bar.
    """
    if value is None or not maximum or maximum <= 0:
        return bar_markup(0.0, width)
    return bar_markup(value / maximum, width)


__all__ = ["bar_markup", "ratio_bar"]
