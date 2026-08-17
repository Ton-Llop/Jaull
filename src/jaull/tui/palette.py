"""The handful of colours that cannot live in the stylesheet.

Bars, breadcrumbs, step markers and table cells are rendered as console markup
or as Rich `Text`, not as widgets, so TCSS variables cannot reach them. Before
this module each of those four places kept its own hex literals, which meant a
palette change was a hunt rather than an edit.

Everything here mirrors a token in `styles.tcss`. If you change one, change
both — the names match deliberately.
"""

from __future__ import annotations

# Structure
BG = "#070a10"
LINE = "#1b2735"
LINE_2 = "#274156"

# Text
INK = "#d8e6f0"
INK_2 = "#8fa8bd"
INK_3 = "#6d8599"

# Meaning. One accent, three states.
ACCENT = "#00e5ff"
ACCENT_DEEP = "#0091a7"
OK = "#00ff9c"
WARN = "#ffc400"
BAD = "#ff2e63"

__all__ = [
    "ACCENT",
    "ACCENT_DEEP",
    "BAD",
    "BG",
    "INK",
    "INK_2",
    "INK_3",
    "LINE",
    "LINE_2",
    "OK",
    "WARN",
]
