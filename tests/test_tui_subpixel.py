"""Guards for the quadrant renderer and the baked sprites drawn through it.

Nothing here asserts what anything looks like — the artwork is allowed to
change. What it does assert is the handful of ways a purely decorative widget
turns into a bug report: a character outside the block set, a colour Rich
refuses, a strip that is not exactly as wide as its widget, or a sprite written
outside the buffer it was handed.

Sizes are in characters unless a name says subpixels; the renderer works in
subpixels, two to a character in each direction.
"""

from __future__ import annotations

import pytest
from textual.geometry import Size

from jaull.tui.widgets import fin_art, shark_art, swim_art
from jaull.tui.widgets.subpixel import (
    BG_RGB,
    QUADRANTS,
    RGB,
    BakedArt,
    Sprite,
    SubpixelWidget,
    blend,
    quadrant,
)

ART: list[BakedArt] = [shark_art, swim_art, fin_art]
ART_IDS = ["jump", "swim", "fin"]


# ---------------------------------------------------------------------------
# The baked art
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_every_frame_is_the_declared_shape(art: BakedArt) -> None:
    for index, frame in enumerate(art.FRAMES):
        rows = frame.strip("\n").splitlines()
        assert len(rows) == art.HEIGHT, f"frame {index} is not HEIGHT rows"
        for y, row in enumerate(rows):
            assert len(row) == art.WIDTH, f"frame {index} row {y} is short"


@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_every_subpixel_indexes_a_real_colour(art: BakedArt) -> None:
    """A stray character would silently render as a hole in the sprite."""
    allowed = set(art.INDEX_CHARS) | {art.TRANSPARENT}
    for index, frame in enumerate(art.FRAMES):
        unknown = set(frame.strip("\n").replace("\n", "")) - allowed
        assert not unknown, f"frame {index} uses unbaked characters {unknown}"


@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_the_palette_and_its_index_agree(art: BakedArt) -> None:
    assert len(art.INDEX_CHARS) == len(art.PALETTE)
    assert len(set(art.INDEX_CHARS)) == len(art.INDEX_CHARS)
    assert len(art.DURATIONS) == len(art.FRAMES)


@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_no_index_character_could_end_the_literal_it_lives_in(art: BakedArt) -> None:
    """The frames are baked into triple-quoted strings by the generator."""
    assert '"' not in art.INDEX_CHARS
    assert "\\" not in art.INDEX_CHARS
    assert art.TRANSPARENT not in art.INDEX_CHARS


@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_every_palette_colour_is_a_colour(art: BakedArt) -> None:
    for hex_colour, alpha in art.PALETTE:
        assert hex_colour.startswith("#") and len(hex_colour) == 7
        assert 0 <= int(hex_colour[1:], 16) <= 0xFFFFFF
        assert 0 < alpha <= 255, "a fully transparent palette slot is wasted"


# ---------------------------------------------------------------------------
# Quadrant encoding
# ---------------------------------------------------------------------------
def test_four_equal_subpixels_need_no_block_glyph() -> None:
    """Most of a band is empty; it must collapse to spaces."""
    grey = (30, 40, 50)
    glyph, fore, back = quadrant(grey, grey, grey, grey)
    assert glyph == " "
    assert fore == back == (30 << 16) | (40 << 8) | 50


def test_a_light_top_over_a_dark_bottom_is_an_upper_half_block() -> None:
    light, dark = (200, 240, 220), (10, 20, 30)
    glyph, fore, back = quadrant(light, light, dark, dark)
    assert glyph == "▀"
    assert fore == (200 << 16) | (240 << 8) | 220
    assert back == (10 << 16) | (20 << 8) | 30


def test_a_single_light_corner_is_its_own_quadrant() -> None:
    light, dark = (200, 240, 220), (10, 20, 30)
    assert quadrant(light, dark, dark, dark)[0] == "▘"
    assert quadrant(dark, light, dark, dark)[0] == "▝"
    assert quadrant(dark, dark, light, dark)[0] == "▖"
    assert quadrant(dark, dark, dark, light)[0] == "▗"


def test_a_diagonal_split_keeps_both_halves() -> None:
    light, dark = (200, 240, 220), (10, 20, 30)
    assert quadrant(light, dark, dark, light)[0] == "▚"
    assert quadrant(dark, light, light, dark)[0] == "▞"


def test_every_glyph_it_can_emit_is_in_the_block_set() -> None:
    """The stylesheet's font requirement is stated in terms of this set."""
    light, dark = (255, 255, 255), (0, 0, 0)
    seen = set()
    for mask in range(16):
        corners = [light if mask & (1 << bit) else dark for bit in range(4)]
        seen.add(quadrant(*corners)[0])  # type: ignore[arg-type]
    assert seen <= set(QUADRANTS)


def test_blending_never_leaves_the_channel_range() -> None:
    """Rich rejects a negative channel outright, which crashes the screen."""
    for ratio in (-5.0, -0.1, 0.0, 0.5, 1.0, 1.1, 40.0):
        for channel in blend((0, 0, 0), (255, 255, 255), ratio):
            assert 0 <= channel <= 255


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_a_sprite_decodes_to_its_declared_size(art: BakedArt) -> None:
    sprite = Sprite(art)
    master = sprite.master()
    assert len(master) == len(art.FRAMES) == sprite.frame_count
    for frame in master:
        assert len(frame) == sprite.height
        assert all(len(row) == sprite.width for row in frame)


@pytest.mark.parametrize("art", ART, ids=ART_IDS)
def test_transparent_subpixels_decode_to_nothing(art: BakedArt) -> None:
    """``None`` rather than black: these are composited over other drawing."""
    sprite = Sprite(art)
    first = art.FRAMES[0].strip("\n").splitlines()
    decoded = sprite.master()[0]
    for y, line in enumerate(first):
        for x, char in enumerate(line):
            assert (decoded[y][x] is None) == (char == art.TRANSPARENT)


def test_rescaling_gives_the_size_that_was_asked_for() -> None:
    sprite = Sprite(fin_art)
    for width, height in ((20, 6), (40, 9), (fin_art.WIDTH, fin_art.HEIGHT)):
        frames = sprite.scaled(width, height)
        assert all(len(frame) == height for frame in frames)
        assert all(len(row) == width for frame in frames for row in frame)


def test_the_same_size_is_only_rescaled_once() -> None:
    sprite = Sprite(fin_art)
    assert sprite.scaled(30, 8) is sprite.scaled(30, 8)


def test_the_rescale_cache_cannot_grow_without_bound() -> None:
    """Dragging a terminal edge asks for every width in between."""
    sprite = Sprite(fin_art)
    for width in range(10, 40):
        sprite.scaled(width, 8)
    assert len(sprite._scaled) <= 6


def test_a_sprite_drawn_off_the_edge_writes_only_inside_the_buffer() -> None:
    """The lane clips the shark on the way in and out; it must not overrun."""
    sprite = Sprite(swim_art)
    width, height = 40, 10
    for left in (-sprite.width - 5, -3, 0, width - 2, width + 5):
        for top in (-4, 0, height - 1, height + 3):
            rows: list[list[RGB]] = [[BG_RGB] * width for _ in range(height)]
            sprite.composite(rows, 0, left, top, sprite.width, sprite.height)
            assert len(rows) == height
            assert all(len(row) == width for row in rows)


def test_a_sprite_entirely_off_the_buffer_draws_nothing() -> None:
    sprite = Sprite(swim_art)
    rows: list[list[RGB]] = [[BG_RGB] * 30 for _ in range(8)]
    sprite.composite(rows, 0, -sprite.width - 1, 0, sprite.width, sprite.height)
    assert all(pixel is BG_RGB for row in rows for pixel in row)


# ---------------------------------------------------------------------------
# The widget base
# ---------------------------------------------------------------------------
class _Flat(SubpixelWidget):
    """A widget that paints one flat colour, to exercise the base alone."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self._fixed = Size(width, height)
        self.painted = 0

    @property
    def size(self) -> Size:
        return self._fixed

    def paint(self, width: int, height: int) -> list[list[RGB]]:
        self.painted += 1
        return [[(9, 30, 40)] * width for _ in range(height)]


def test_the_base_class_refuses_to_paint_by_itself() -> None:
    """A subclass that forgets `paint` should fail loudly, not draw nothing."""

    class _Bare(SubpixelWidget):
        @property
        def size(self) -> Size:
            return Size(10, 2)

    with pytest.raises(NotImplementedError):
        _Bare().render_line(0)


@pytest.mark.parametrize("size", [(80, 7), (110, 9), (200, 3), (12, 1)])
def test_every_strip_fills_exactly_the_widget_width(size: tuple[int, int]) -> None:
    """A strip that is too long or too short corrupts the whole screen."""
    width, height = size
    widget = _Flat(width, height)
    for y in range(height):
        assert widget.render_line(y).cell_length == width, f"row {y} is wrong"


def test_the_scene_is_built_once_per_tick() -> None:
    """Textual asks line by line; recomposing per line is many times the work."""
    widget = _Flat(60, 6)
    first = widget.render_line(0)
    assert widget.render_line(0) is first
    assert widget.render_line(5) is not first
    assert widget.painted == 1
    widget._elapsed_ms += widget.TICK_MS
    assert widget.render_line(0) is not first
    assert widget.painted == 2


def test_a_line_outside_the_widget_is_blank_rather_than_an_error() -> None:
    widget = _Flat(20, 2)
    assert widget.render_line(9).cell_length == 20


def test_a_widget_with_no_height_draws_nothing() -> None:
    """Squeezed to zero by a short terminal, it must still answer politely."""
    widget = _Flat(30, 0)
    assert widget.render_line(0).cell_length == 30
    assert widget.painted == 0
