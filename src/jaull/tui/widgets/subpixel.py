"""Drawing pictures in a terminal, four subpixels to a character cell.

Every cell is a quadrant block — ``▘▝▀▖▌▞▛▗▚▐▜▄▙▟█`` — so one character carries
**four** subpixels, two across and two down. A widget built on this composes a
``2*width x 2*height`` buffer of plain RGB triples and only turns it into text
at the end.

Quadrants rather than the simpler ``▀`` half-block because a terminal cell is
roughly twice as tall as it is wide: half-blocks give square pixels but only
one per column, and at the sizes these bands get on a real terminal that is not
enough resolution to hold a jaw. Quadrants double the horizontal detail for the
same physical size. The cost is that a cell carries only two colours, so each
one is split into a lighter and a darker group and averaged — which pixel art
survives well, having few colours to begin with.

The consequence to keep in mind is that a subpixel is **half a cell wide and a
whole half-cell tall**: it is a 1:2 pixel, not a square one. Anything meant to
look round has to be drawn twice as wide in subpixels as it is tall, and the
baked sprites are stored pre-stretched for the same reason.

Two things live here: :class:`Sprite`, which decodes and rescales the baked art
from ``scripts/bake_shark_frames.py``, and :class:`SubpixelWidget`, which owns
the animation clock and the buffer-to-``Strip`` encoding. What is drawn into
the buffer is entirely the subclass's business.
"""

from __future__ import annotations

from typing import Protocol

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual.widget import Widget

from jaull.tui.palette import BG

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

#: One frame of decoded sprite: rows of pixels, ``None`` where nothing is drawn.
Frame = tuple[tuple[RGBA | None, ...], ...]

#: The stylesheet background, unpacked once so subpixels can blend into it.
BG_RGB: RGB = (int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16))

# Quadrant glyph per 4-bit mask of "this subpixel is the lighter colour",
# ordered top-left, top-right, bottom-left, bottom-right. Index 0 and 15 are
# never looked up: a cell whose subpixels all landed in one group is emitted as
# a space with a background, which needs no block glyph at all and merges with
# its neighbours in the run-length pass.
QUADRANTS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"

#: Luminance spread, on the weighted 0..255000 scale used below, under which
#: four subpixels are called one colour. Small, but not zero: without it a
#: gradient emits a block glyph for differences no one can see, costing a
#: segment per cell and stopping long flat runs from merging.
FLAT_TOLERANCE = 2600

#: Rescaled frames kept per sprite. Enough for a couple of live sizes; without
#: a bound, dragging a terminal edge would keep every size in between alive.
_SCALE_CACHE = 6


def blend(a: RGB, b: RGB, ratio: float) -> RGB:
    """``a`` moved ``ratio`` of the way toward ``b``, clamped at both ends."""
    ratio = 0.0 if ratio < 0 else 1.0 if ratio > 1 else ratio
    return (
        round(a[0] + (b[0] - a[0]) * ratio),
        round(a[1] + (b[1] - a[1]) * ratio),
        round(a[2] + (b[2] - a[2]) * ratio),
    )


def quadrant(
    top_left: RGB, top_right: RGB, bottom_left: RGB, bottom_right: RGB
) -> tuple[str, int, int]:
    """Four subpixels as one character: a glyph, a foreground and a background.

    A cell can only hold two colours, so the four are split at the midpoint of
    their luminance range — the lighter ones become the glyph, the darker ones
    the background, and each group is averaged. Splitting on luminance rather
    than on hue is what keeps an edge an edge: in this artwork every boundary
    that matters (body against sea, tooth against jaw, foam against water) is a
    light/dark boundary, and the glyph is chosen to fall exactly along it.

    Colours come back packed into ints so the caller can key a style cache on
    them without allocating a tuple per cell.
    """
    if top_left is top_right is bottom_left is bottom_right:
        # Every untouched subpixel is the *same* background tuple, so identity
        # settles the empty part of a band — most of it, usually — before any
        # arithmetic happens. Worth about a third of the drawing cost.
        packed = (top_left[0] << 16) | (top_left[1] << 8) | top_left[2]
        return " ", packed, packed
    quad = (top_left, top_right, bottom_left, bottom_right)
    lums = [(c[0] * 299 + c[1] * 587 + c[2] * 114) for c in quad]
    low, high = min(lums), max(lums)
    if high - low <= FLAT_TOLERANCE:
        # Uniform enough to be one colour. A space needs no block glyph, and
        # runs of them collapse in the encoder.
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
    return QUADRANTS[mask], fore, back


class BakedArt(Protocol):
    """The shape of a module written by ``scripts/bake_shark_frames.py``."""

    WIDTH: int
    HEIGHT: int
    PALETTE: tuple[tuple[str, int], ...]
    INDEX_CHARS: str
    TRANSPARENT: str
    DURATIONS: tuple[int, ...]
    FRAMES: tuple[str, ...]


class Sprite:
    """Baked pixel art, decoded once and rescaled on demand.

    The art is stored one character per subpixel indexing a palette, which is
    compact in the repository but useless to draw from directly. This unpacks
    it to RGBA the first time it is asked for and caches each size it is drawn
    at, because a band only changes size when the terminal does.
    """

    def __init__(self, art: BakedArt) -> None:
        self._art = art
        self.width = art.WIDTH
        self.height = art.HEIGHT
        self.durations = art.DURATIONS
        self._master: tuple[Frame, ...] | None = None
        self._scaled: dict[tuple[int, int], tuple[Frame, ...]] = {}

    @property
    def frame_count(self) -> int:
        return len(self._art.FRAMES)

    def master(self) -> tuple[Frame, ...]:
        """The baked frames at their stored resolution, as RGBA."""
        if self._master is None:
            lookup: dict[str, RGBA] = {}
            for char, (hex_colour, alpha) in zip(
                self._art.INDEX_CHARS, self._art.PALETTE, strict=True
            ):
                value = hex_colour.lstrip("#")
                lookup[char] = (
                    int(value[0:2], 16),
                    int(value[2:4], 16),
                    int(value[4:6], 16),
                    alpha,
                )
            self._master = tuple(
                tuple(
                    tuple(lookup.get(char) for char in line)
                    for line in frame.strip("\n").splitlines()
                )
                for frame in self._art.FRAMES
            )
        return self._master

    def scaled(self, width: int, height: int) -> tuple[Frame, ...]:
        """The frames at a given subpixel size, area-averaged in premultiplied alpha.

        Averaging straight RGB would let fully transparent subpixels drag the
        edges toward black; premultiplying keeps a soft edge the colour it
        looks like. Nearest-neighbour was the other option and it eats the
        teeth, which are one subpixel wide at these sizes.
        """
        key = (width, height)
        cached = self._scaled.get(key)
        if cached is not None:
            return cached
        frames = self._rescale(width, height)
        if len(self._scaled) >= _SCALE_CACHE:
            # Insertion-ordered, so this drops the size drawn longest ago.
            del self._scaled[next(iter(self._scaled))]
        self._scaled[key] = frames
        return frames

    def _rescale(self, width: int, height: int) -> tuple[Frame, ...]:
        scaled: list[Frame] = []
        for frame in self.master():
            rows: list[tuple[RGBA | None, ...]] = []
            for y in range(height):
                y0 = y * self.height // height
                y1 = max(y0 + 1, (y + 1) * self.height // height)
                row: list[RGBA | None] = []
                for x in range(width):
                    x0 = x * self.width // width
                    x1 = max(x0 + 1, (x + 1) * self.width // width)
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
                    row.append(
                        (red // alpha, green // alpha, blue // alpha, alpha // count)
                    )
                rows.append(tuple(row))
            scaled.append(tuple(rows))
        return tuple(scaled)

    def composite(
        self,
        rows: list[list[RGB]],
        frame_index: int,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        """Draw one frame into a subpixel buffer, honouring its alpha.

        Alpha survives the bake because these sprites always land on something
        the widget drew first — the sea, or the sonar rings — so a soft edge
        has to blend with whatever is already there rather than with a
        background known in advance.
        """
        frame = self.scaled(width, height)[frame_index]
        band_height, band_width = len(rows), len(rows[0]) if rows else 0
        for y, row in enumerate(frame):
            target_y = top + y
            if not 0 <= target_y < band_height:
                continue
            target = rows[target_y]
            for x, pixel in enumerate(row):
                if pixel is None:
                    continue
                target_x = left + x
                if not 0 <= target_x < band_width:
                    continue
                red, green, blue, alpha = pixel
                if alpha >= 250:
                    target[target_x] = (red, green, blue)
                else:
                    target[target_x] = blend(
                        target[target_x], (red, green, blue), alpha / 255
                    )


class SubpixelWidget(Widget):
    """A widget whose picture is a subpixel buffer on an animation clock.

    Subclasses implement :meth:`paint`. Everything else — when to repaint, how
    to turn subpixels into quadrant glyphs, and how to avoid doing that work
    once per line — is handled here.
    """

    #: Repaint interval, in seconds and in the milliseconds the clock counts.
    #: Both, rather than one derived from the other, so the clock advances in
    #: exact integers and an animation can be replayed deterministically.
    TICK_SECONDS = 0.12
    TICK_MS = 120

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_ms = 0
        self._styles: dict[tuple[int, int], Style] = {}
        # Named for what they are rather than for the picture: a subclass
        # wanting `self._frame` for its sprite index is the obvious thing
        # to write, and it used to shadow this.
        self._cache_key: tuple[int, int, int] | None = None
        self._cached_strips: list[Strip] = []

    def on_mount(self) -> None:
        self.set_interval(self.TICK_SECONDS, self._advance)

    def _advance(self) -> None:
        self._elapsed_ms += self.TICK_MS
        self.refresh()

    def paint(self, width: int, height: int) -> list[list[RGB]]:
        """The scene as ``height`` rows of ``width`` subpixels."""
        raise NotImplementedError

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        strips = self._strips()
        if not 0 <= y < len(strips):
            return Strip.blank(max(0, width))
        return strips[y]

    def _strips(self) -> list[Strip]:
        """The whole band for the current tick, built once and reused.

        Textual asks for one line at a time, but these scenes are not separable
        by line: a wave column and the shark on top of it are computed
        together. Without this cache every repaint would compose the entire
        band once per row it contains — twenty-odd times the work for the same
        picture.
        """
        width, height = self.size.width, self.size.height
        key = (width, height, self._elapsed_ms)
        if self._cache_key == key:
            return self._cached_strips
        if width <= 0 or height <= 0:
            self._cache_key, self._cached_strips = key, []
            return self._cached_strips
        rows = self.paint(width * 2, height * 2)
        self._cached_strips = [
            Strip(self._segments(rows[y * 2], rows[y * 2 + 1], width), width)
            for y in range(height)
        ]
        self._cache_key = key
        return self._cached_strips

    def _segments(
        self, top: list[RGB], bottom: list[RGB], width: int
    ) -> list[Segment]:
        """Run-length encode one character row out of two subpixel rows."""
        segments: list[Segment] = []
        run_text: str | None = None
        run_key: tuple[int, int] | None = None
        run_length = 0
        for cell in range(width):
            x = cell * 2
            glyph, fore, back = quadrant(top[x], top[x + 1], bottom[x], bottom[x + 1])
            key = (fore, back)
            if glyph == run_text and key == run_key:
                run_length += 1
                continue
            if run_key is not None and run_text is not None:
                segments.append(Segment(run_text * run_length, self._style(run_key)))
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


__all__ = [
    "BG_RGB",
    "FLAT_TOLERANCE",
    "QUADRANTS",
    "RGB",
    "RGBA",
    "BakedArt",
    "Frame",
    "Sprite",
    "SubpixelWidget",
    "blend",
    "quadrant",
]
