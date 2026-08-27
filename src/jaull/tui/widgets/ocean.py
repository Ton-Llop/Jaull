"""The sea under the home screen, with the Jaull shark breaching out of it.

The welcome screen is a logo, a line of copy and two menu items, which leaves
most of a maximised terminal empty below the fold. This fills that gap with
something that is still the product's own artwork rather than filler.

How it is drawn
---------------
Every cell is a quadrant block — ``▘▝▀▖▌▞▛▗▚▐▜▄▙▟█`` — so one character carries
**four** subpixels, two across and two down. The widget composes a
``2*width x 2*height`` subpixel buffer and only turns it into text at the end.

Quadrants rather than the simpler ``▀`` half-block because a terminal cell is
roughly twice as tall as it is wide: half-blocks give square pixels but only
one per column, and at the size this band gets on a real terminal that is not
enough resolution to hold a jaw. Quadrants double the horizontal detail for the
same physical size. The cost is that a cell carries only two colours, so each
one is split into a lighter and a darker group and averaged — which pixel art
survives well, having few colours to begin with.

Two things share the buffer:

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
from functools import lru_cache
from itertools import pairwise
from typing import NamedTuple

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual.widget import Widget

from jaull.tui.palette import BG
from jaull.tui.widgets import shark_art

# One tick drives both the swell and the sprite clock. Water reads as water at
# roughly eight frames a second; faster only costs CPU on an idle menu.
_TICK_SECONDS = 0.12
_TICK_MS = 120

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

#: The stylesheet background, unpacked once so subpixels can blend into it.
_BG_RGB = (int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16))

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
_DEPTH_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
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

# Quadrant glyph per 4-bit mask of "this subpixel is the lighter colour",
# ordered top-left, top-right, bottom-left, bottom-right. Index 0 and 15 are
# never looked up: a cell whose subpixels all landed in one group is emitted as
# a space with a background, which needs no block glyph at all and merges with
# its neighbours in the run-length pass.
_QUADRANTS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"

#: Luminance spread, on the weighted 0..255000 scale used below, under which
#: four subpixels are called one colour. Small, but not zero: without it the
#: water emits a block glyph for differences no one can see, costing a segment
#: per cell and stopping long flat runs from merging.
_FLAT_TOLERANCE = 2600


def _ramp(steps: int = 40) -> tuple[tuple[int, int, int], ...]:
    """The depth stops flattened into a lookup, one entry per half subpixel."""
    out: list[tuple[int, int, int]] = []
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


def _blend(
    a: tuple[int, int, int], b: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    ratio = 0.0 if ratio < 0 else 1.0 if ratio > 1 else ratio
    return (
        round(a[0] + (b[0] - a[0]) * ratio),
        round(a[1] + (b[1] - a[1]) * ratio),
        round(a[2] + (b[2] - a[2]) * ratio),
    )


@lru_cache(maxsize=1)
def _master_frames() -> (
    tuple[tuple[tuple[tuple[int, int, int, int] | None, ...], ...], ...]
):
    """The baked art decoded once: frames of rows of RGBA, ``None`` for empty."""
    lookup: dict[str, tuple[int, int, int, int]] = {}
    for char, (hex_colour, alpha) in zip(
        shark_art.INDEX_CHARS, shark_art.PALETTE, strict=True
    ):
        value = hex_colour.lstrip("#")
        lookup[char] = (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
            alpha,
        )
    frames = []
    for frame in shark_art.FRAMES:
        rows = tuple(
            tuple(lookup.get(char) for char in line)
            for line in frame.strip("\n").splitlines()
        )
        frames.append(rows)
    return tuple(frames)


@lru_cache(maxsize=6)
def _scaled_frames(
    width: int, height: int
) -> tuple[tuple[tuple[tuple[int, int, int, int] | None, ...], ...], ...]:
    """The sprite at a given subpixel size, area-averaged in premultiplied alpha.

    Averaging straight RGB would let fully transparent subpixels drag the edges
    toward black; premultiplying keeps a soft edge the colour it looks like.
    Nearest-neighbour was the other option and it eats the teeth, which are one
    subpixel wide at this scale.
    """
    master = _master_frames()
    scaled = []
    for frame in master:
        rows = []
        for y in range(height):
            y0 = y * shark_art.HEIGHT // height
            y1 = max(y0 + 1, (y + 1) * shark_art.HEIGHT // height)
            row: list[tuple[int, int, int, int] | None] = []
            for x in range(width):
                x0 = x * shark_art.WIDTH // width
                x1 = max(x0 + 1, (x + 1) * shark_art.WIDTH // width)
                red = green = blue = alpha = 0
                count = 0
                for sy in range(y0, y1):
                    for sx in range(x0, x1):
                        count += 1
                        pixel = frame[sy][sx]
                        if pixel is None:
                            continue
                        weight = pixel[3]
                        red += pixel[0] * weight
                        green += pixel[1] * weight
                        blue += pixel[2] * weight
                        alpha += weight
                if not count or alpha == 0:
                    row.append(None)
                    continue
                row.append((red // alpha, green // alpha, blue // alpha, alpha // count))
            rows.append(tuple(row))
        scaled.append(tuple(rows))
    return tuple(scaled)


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


class OceanBand(Widget):
    """Animated sea for the home screen. Decorative: it holds no state."""

    DEFAULT_CLASSES = "ocean"

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_ms = 0
        self._styles: dict[tuple[int, int], Style] = {}
        self._frame_key: tuple[int, int, int] | None = None
        self._frame: list[Strip] = []

    def on_mount(self) -> None:
        self.set_interval(_TICK_SECONDS, self._advance)

    def _advance(self) -> None:
        self._elapsed_ms += _TICK_MS
        self.refresh()

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

    # -- drawing ------------------------------------------------------------
    def render_line(self, y: int) -> Strip:
        width = self.size.width
        strips = self._strips()
        if not 0 <= y < len(strips):
            return Strip.blank(max(0, width))
        return strips[y]

    def _strips(self) -> list[Strip]:
        """The whole band for the current tick, built once and reused.

        Textual asks for one line at a time, but the sea is not separable by
        line: a wave column and the shark on top of it are computed together.
        Without this cache every repaint would compose the entire band once per
        row it contains — twenty-odd times the work for the same picture.
        """
        width, height = self.size.width, self.size.height
        key = (width, height, self._elapsed_ms)
        if self._frame_key == key:
            return self._frame
        if width <= 0 or height <= 0:
            self._frame_key, self._frame = key, []
            return self._frame
        rows = self._subpixels(width * 2, height * 2)
        self._frame = [
            Strip(self._segments(rows[y * 2], rows[y * 2 + 1], width), width)
            for y in range(height)
        ]
        self._frame_key = key
        return self._frame

    # -- subpixels to characters --------------------------------------------
    def _segments(
        self,
        top: list[tuple[int, int, int]],
        bottom: list[tuple[int, int, int]],
        width: int,
    ) -> list[Segment]:
        """Run-length encode one character row out of two subpixel rows."""
        segments: list[Segment] = []
        run_text: str | None = None
        run_key: tuple[int, int] | None = None
        run_length = 0
        for cell in range(width):
            x = cell * 2
            glyph, fore, back = _quadrant(
                top[x], top[x + 1], bottom[x], bottom[x + 1]
            )
            key = (fore, back)
            if glyph == run_text and key == run_key:
                run_length += 1
                continue
            if run_key is not None and run_text is not None:
                segments.append(
                    Segment(run_text * run_length, self._style(run_key))
                )
            run_text, run_key, run_length = glyph, key, 1
        if run_key is not None and run_text is not None:
            segments.append(Segment(run_text * run_length, self._style(run_key)))
        return segments

    def _style(self, key: tuple[int, int]) -> Style:
        style = self._styles.get(key)
        if style is None:
            style = Style(color=f"#{key[0]:06x}", bgcolor=f"#{key[1]:06x}")
            self._styles[key] = style
        return style

    # -- the scene ----------------------------------------------------------
    def _subpixels(
        self, width: int, height: int
    ) -> list[list[tuple[int, int, int]]]:
        """The whole band as subpixel rows, sea first and shark on top."""
        rows = [[_BG_RGB] * width for _ in range(height)]
        water_row, box = _layout(width, height)
        surface = self._surface(width, water_row)
        self._draw_water(rows, surface, width, height)
        if box is not None:
            self._draw_shark(rows, box, height)
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
        rows: list[list[tuple[int, int, int]]],
        surface: list[float],
        width: int,
        height: int,
    ) -> None:
        # Only the rows the sea can actually reach are touched; everything
        # above stays the background it was allocated as.
        first = max(0, int(min(surface)) - 1)
        tick = self._elapsed_ms // _TICK_MS
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
                colour = _WATER_RAMP[
                    0 if index < 0 else min(index, len(_WATER_RAMP) - 1)
                ]
                if depth < _STREAK_REACH:
                    for streak_depth, strength in streaks:
                        if abs(depth - streak_depth) < 0.75:
                            colour = _blend(colour, _STREAK_TINT, strength)
                            break
                if sparkle and -0.5 <= depth < 1.0:
                    colour = _blend(colour, _FOAM, 0.55)
                # The topmost subpixel is only partly under water. Fading it
                # into the background is what keeps a shallow swell from
                # looking like a staircase.
                coverage = depth + 1.0
                if coverage < 1.0:
                    colour = _blend(_BG_RGB, colour, coverage)
                rows[y][x] = colour

    @staticmethod
    def _sparkles(x: int, tick: int) -> bool:
        """A deterministic twinkle, so no random state has to be carried."""
        noise = (x * 2654435761 ^ (tick // 3) * 40503) & 0xFFFF
        return noise < 320

    def _draw_shark(
        self,
        rows: list[list[tuple[int, int, int]]],
        box: _SpriteBox,
        height: int,
    ) -> None:
        frame_index = self._sprite_frame()
        if frame_index is None:
            return
        left, top = box.left, box.top
        frame = _scaled_frames(box.width, box.height)[frame_index]
        for y, row in enumerate(frame):
            target_y = top + y
            if not 0 <= target_y < height:
                continue
            target = rows[target_y]
            for x, pixel in enumerate(row):
                if pixel is None:
                    continue
                red, green, blue, alpha = pixel
                if alpha >= 250:
                    target[left + x] = (red, green, blue)
                else:
                    target[left + x] = _blend(
                        target[left + x], (red, green, blue), alpha / 255
                    )


def _quadrant(
    top_left: tuple[int, int, int],
    top_right: tuple[int, int, int],
    bottom_left: tuple[int, int, int],
    bottom_right: tuple[int, int, int],
) -> tuple[str, int, int]:
    """Four subpixels as one character: a glyph, a foreground and a background.

    A cell can only hold two colours, so the four are split at the midpoint of
    their luminance range — the lighter ones become the glyph, the darker ones
    the background, and each group is averaged. Splitting on luminance rather
    than on hue is what keeps an edge an edge: in this artwork every boundary
    that matters (body against sea, tooth against jaw, foam against water) is a
    light/dark boundary, and the glyph is chosen to fall exactly along it.
    """
    if top_left is top_right is bottom_left is bottom_right:
        # Every untouched subpixel is the *same* background tuple, so identity
        # settles the sky — most of the band on a tall terminal — before any
        # arithmetic happens. Worth about a third of the drawing cost.
        packed = (top_left[0] << 16) | (top_left[1] << 8) | top_left[2]
        return " ", packed, packed
    quad = (top_left, top_right, bottom_left, bottom_right)
    lums = [(c[0] * 299 + c[1] * 587 + c[2] * 114) for c in quad]
    low, high = min(lums), max(lums)
    if high - low <= _FLAT_TOLERANCE:
        # Uniform enough to be one colour. A space needs no block glyph, and
        # runs of them collapse in the encoder — which is most of the sky.
        red = (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0]) // 4
        green = (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1]) // 4
        blue = (quad[0][2] + quad[1][2] + quad[2][2] + quad[3][2]) // 4
        packed = (red << 16) | (green << 8) | blue
        return " ", packed, packed
    midpoint = (low + high) / 2
    mask = 0
    light_r = light_g = light_b = light_n = 0
    dark_r = dark_g = dark_b = dark_n = 0
    for bit, (colour, lum) in enumerate(zip(quad, lums, strict=True)):
        if lum > midpoint:
            mask |= 1 << bit
            light_r += colour[0]
            light_g += colour[1]
            light_b += colour[2]
            light_n += 1
        else:
            dark_r += colour[0]
            dark_g += colour[1]
            dark_b += colour[2]
            dark_n += 1
    fore = (
        ((light_r // light_n) << 16)
        | ((light_g // light_n) << 8)
        | (light_b // light_n)
    )
    back = ((dark_r // dark_n) << 16) | ((dark_g // dark_n) << 8) | (dark_b // dark_n)
    return _QUADRANTS[mask], fore, back


__all__ = ["OceanBand"]
