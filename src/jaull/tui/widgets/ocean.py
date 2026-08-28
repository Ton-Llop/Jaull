"""The sea under the home screen, with the Jaull shark breaching out of it.

The welcome screen is a logo, a line of copy and two menu items, which leaves
most of a maximised terminal empty below the fold. This fills that gap with
something that is still the product's own artwork rather than filler.

It is drawn in quadrant blocks — see :mod:`jaull.tui.widgets.subpixel` for how
that works and what it costs. Two things share the buffer:

*The swell* is procedural — a sum of two sines at different wavelengths and
speeds, so the crest never visibly repeats — and spans the full width.

*The shark* is baked pixel art (:mod:`jaull.tui.widgets.shark_art`) composited
on top with its own alpha, which matters because the splash overlaps the
waterline. The sprite is positioned by its own waterline row rather than by its
top edge, so however it is scaled, the drawn splash always lands on the
procedural sea.

It does not loop continuously. A shark that breaches every two seconds forever
turns a menu into a distraction, so the jump plays once and then the sea rests
for a few seconds before the next one.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import NamedTuple

from jaull.tui.widgets import shark_art
from jaull.tui.widgets.subpixel import BG_RGB, RGB, Sprite, SubpixelWidget, blend

_SHARK = Sprite(shark_art)

#: Calm between jumps. Long enough that the breach is an event.
_REST_MS = 5200

#: Fraction of the band that is sea. Deep enough to hold the ripple streaks —
#: below about a quarter the sea collapses into a single lit line with nothing
#: under it, which reads as a neon rule rather than as water.
_WATER_FRACTION = 0.74

# The sprite is sized from the room above the waterline rather than from the
# band, so lowering _WATER_FRACTION can never push the shark's nose off the
# top. Both bounds are in subpixel rows, two to a character row.
_MIN_SPRITE_ROWS = 18
_MAX_SPRITE_ROWS = shark_art.HEIGHT

#: Waterline for a band too short to ever hold a shark. With nothing to
#: leap over the sea, the horizon may as well sit near the middle.
_SEA_ONLY_FRACTION = 0.45

# Ripple streaks: (depth, wavelength, speed, strength). Lighter lines running
# across the dark water under the crest. One lit surface with black beneath it
# is a line; a few streaks moving at their own speeds are a surface with
# something behind it. They travel slower the deeper they sit, which is the
# cheapest available cue for distance.
_STREAKS: tuple[tuple[float, float, float, float], ...] = (
    (2.2, 0.115, 0.62, 0.20),
    (4.0, 0.085, 0.41, 0.13),
    (6.2, 0.065, 0.27, 0.08),
)

#: Streaks lighten toward the artwork's own midtone rather than to the
#: stylesheet accent. Pulled toward full neon they stop being texture on water
#: and become more lit lines.
_STREAK_TINT = (0x2F, 0x9C, 0x85)

#: Deepest a streak can reach: its own depth, plus how far the sine swings it,
#: plus the tolerance it is matched with. Below this the inner loop skips the
#: check entirely, which on a deep band is most of the water.
_STREAK_REACH = max(depth for depth, *_ in _STREAKS) + 0.85 + 0.75

# Depth ramp, in subpixel rows below the surface.
#
# Sampled from the GIF's own water — the last frame is nothing but ripples, so
# it is a straight reading of what the artist painted the sea: #daf8e1 foam,
# #5fdcb8 and #106f61 through the midtones, #05352c in the dark. That sea is
# green, not cyan; drawn in `$accent` it was both the wrong colour and louder
# than the logo, which made the menu the second thing you looked at.
#
# The crest tops out at the GIF's bright water, not at its foam. The swell
# already smears the surface across three or four subpixels by moving it
# between columns, so putting near-white at depth zero paints a continuous pale
# band along the whole width — which is not what the artwork does either: its
# ripples are teal, with foam only as scattered glints. Those glints are the
# sparkle pass below, which is where #daf8e1 actually belongs.
#
# The tail lands just above the background so the sea dissolves into the screen
# instead of ending on a slab of colour.
_DEPTH_STOPS: tuple[tuple[float, RGB], ...] = (
    (0.0, (0x5F, 0xDC, 0xB8)),
    (0.5, (0x22, 0x90, 0x77)),
    (1.2, (0x10, 0x6F, 0x61)),
    (2.5, (0x05, 0x46, 0x3F)),
    (4.5, (0x05, 0x35, 0x2C)),
    (8.0, (0x0A, 0x1C, 0x1D)),
    (14.0, (0x08, 0x0D, 0x13)),
)

#: Foam. Scattered by the sparkle pass along the crest, never drawn as a line.
_FOAM = (0xDA, 0xF8, 0xE1)


def _ramp(steps: int = 40) -> tuple[RGB, ...]:
    """The depth stops flattened into a lookup, one entry per half subpixel."""
    out: list[RGB] = []
    for index in range(steps):
        depth = index / 2
        lower = _DEPTH_STOPS[0]
        upper = _DEPTH_STOPS[-1]
        for stop, following in pairwise(_DEPTH_STOPS):
            if stop[0] <= depth <= following[0]:
                lower, upper = stop, following
                break
        span = upper[0] - lower[0]
        ratio = 0.0 if span <= 0 else (depth - lower[0]) / span
        # Past the deepest stop there is no pair to interpolate between, and
        # the fallback pair spans the whole ramp — extrapolating there runs the
        # channels negative and Rich rejects the colour. Deeper than the last
        # stop is simply the last colour.
        ratio = 0.0 if ratio < 0 else 1.0 if ratio > 1 else ratio
        out.append(
            tuple(  # type: ignore[arg-type]
                round(a + (b - a) * ratio)
                for a, b in zip(lower[1], upper[1], strict=True)
            )
        )
    return tuple(out)


_WATER_RAMP = _ramp()


class _SpriteBox(NamedTuple):
    """Where the shark goes, in subpixels within the band."""

    left: int
    top: int
    width: int
    height: int


def _layout(sub_width: int, sub_height: int) -> tuple[float, _SpriteBox | None]:
    """Waterline and sprite box for a band of this size, all in subpixels.

    Both come out of one function because they are one decision. The sprite is
    anchored by its own waterline, so its height is dictated by the room
    *above* the sea rather than by the band: that is what guarantees the arc
    always fits instead of losing the shark's nose off the top edge.
    """
    water_row = sub_height * _WATER_FRACTION
    height = min(
        _MAX_SPRITE_ROWS, int(water_row * shark_art.HEIGHT / shark_art.WATERLINE)
    )
    width = round(height * shark_art.WIDTH / shark_art.HEIGHT)
    if height < _MIN_SPRITE_ROWS or not 0 < width <= sub_width:
        # Nothing will ever breach in a band this small, so the sea takes more
        # of it. Holding the waterline down here would spend two thirds of an
        # already short band on empty sky above a single lit line.
        return sub_height * _SEA_ONLY_FRACTION, None
    top = round(water_row - height * (shark_art.WATERLINE / shark_art.HEIGHT))
    # Centred on an even subpixel, so the sprite never straddles a cell
    # boundary differently from one frame to the next.
    left = ((sub_width - width) // 2) & ~1
    return water_row, _SpriteBox(left, top, width, height)


class OceanBand(SubpixelWidget):
    """Animated sea for the home screen. Decorative: it holds no state."""

    DEFAULT_CLASSES = "ocean"

    # -- animation clock ----------------------------------------------------
    def _sprite_frame(self) -> int | None:
        """Which GIF frame is showing, or ``None`` while the sea is resting."""
        cycle = sum(shark_art.DURATIONS) + _REST_MS
        phase = self._elapsed_ms % cycle
        for index, duration in enumerate(shark_art.DURATIONS):
            if phase < duration:
                return index
            phase -= duration
        return None

    # -- the scene ----------------------------------------------------------
    def paint(self, width: int, height: int) -> list[list[RGB]]:
        """The whole band as subpixel rows, sea first and shark on top."""
        rows = [[BG_RGB] * width for _ in range(height)]
        water_row, box = _layout(width, height)
        surface = self._surface(width, water_row)
        self._draw_water(rows, surface, width, height)
        frame_index = self._sprite_frame()
        if box is not None and frame_index is not None:
            _SHARK.composite(
                rows, frame_index, box.left, box.top, box.width, box.height
            )
        return rows

    def _surface(self, width: int, base: float) -> list[float]:
        """Sea level per subpixel column, in subpixel rows from the top."""
        time = self._elapsed_ms / 1000.0
        return [
            base
            + 1.05 * math.sin(x * 0.095 + time * 1.15)
            + 0.55 * math.sin(x * 0.205 - time * 0.73 + 1.7)
            for x in range(width)
        ]

    def _draw_water(
        self,
        rows: list[list[RGB]],
        surface: list[float],
        width: int,
        height: int,
    ) -> None:
        # Only the rows the sea can actually reach are touched; everything
        # above stays the background it was allocated as.
        first = max(0, int(min(surface)) - 1)
        tick = self._elapsed_ms // self.TICK_MS
        time = self._elapsed_ms / 1000.0
        for x in range(width):
            level = surface[x]
            sparkle = self._sparkles(x, tick)
            # A streak's depth depends only on the column and the clock, so it
            # is resolved once here rather than per subpixel down the water.
            streaks = [
                (depth + 0.85 * math.sin(x * length + time * speed), strength)
                for depth, length, speed, strength in _STREAKS
            ]
            for y in range(first, height):
                depth = y - level
                if depth < -1.0:
                    continue
                index = int(depth * 2)
                colour = _WATER_RAMP[0 if index < 0 else min(index, len(_WATER_RAMP) - 1)]
                if depth < _STREAK_REACH:
                    for streak_depth, strength in streaks:
                        if abs(depth - streak_depth) < 0.75:
                            colour = blend(colour, _STREAK_TINT, strength)
                            break
                if sparkle and -0.5 <= depth < 1.0:
                    colour = blend(colour, _FOAM, 0.55)
                # The topmost subpixel is only partly under water. Fading it
                # into the background is what keeps a shallow swell from
                # looking like a staircase.
                coverage = depth + 1.0
                if coverage < 1.0:
                    colour = blend(BG_RGB, colour, coverage)
                rows[y][x] = colour

    @staticmethod
    def _sparkles(x: int, tick: int) -> bool:
        """A deterministic twinkle, so no random state has to be carried."""
        noise = (x * 2654435761 ^ (tick // 3) * 40503) & 0xFFFF
        return noise < 320


__all__ = ["OceanBand"]
