"""Guards for the home screen's sea.

The band is decorative, so nothing here asserts what it looks like. What it
does assert is that it cannot break the screen it sits on: no crash, no colour
Rich will reject, no character outside the block set, no strip wider than the
widget, and no shark drawn outside the band. Those are the ways a purely visual
widget turns into a bug report.

Sizes are given in characters; the widget works in subpixels, two to a
character in each direction, so anything passed to ``_layout`` is doubled.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.geometry import Size

from jaull.tui.app import JaullApp
from jaull.tui.widgets import shark_art
from jaull.tui.widgets.ocean import _QUADRANTS, _REST_MS, OceanBand, _layout, _quadrant

# 80x24 leaves the band about seven rows and no room for a shark; the taller
# sizes are where it breaches. 200x60 is not a supported layout, it is here
# because a band that deep is what first ran the water past its last depth
# stop.
SIZES = [(80, 24), (90, 28), (110, 32), (120, 40), (200, 60)]


class _Band(OceanBand):
    """An unmounted band with a size, so drawing can be tested without an app."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self._fixed = Size(width, height)

    @property
    def size(self) -> Size:
        return self._fixed


def _all_lines(band: _Band, height: int) -> list:
    return [band.render_line(y) for y in range(height)]


# ---------------------------------------------------------------------------
# The baked art
# ---------------------------------------------------------------------------
def test_every_frame_is_the_declared_shape() -> None:
    for index, frame in enumerate(shark_art.FRAMES):
        rows = frame.strip("\n").splitlines()
        assert len(rows) == shark_art.HEIGHT, f"frame {index} is not HEIGHT rows"
        for y, row in enumerate(rows):
            assert len(row) == shark_art.WIDTH, f"frame {index} row {y} is short"


def test_every_subpixel_indexes_a_real_colour() -> None:
    """A stray character would silently render as a hole in the shark."""
    allowed = set(shark_art.INDEX_CHARS) | {shark_art.TRANSPARENT}
    for index, frame in enumerate(shark_art.FRAMES):
        unknown = set(frame.strip("\n").replace("\n", "")) - allowed
        assert not unknown, f"frame {index} uses unbaked characters {unknown}"


def test_the_palette_and_its_index_agree() -> None:
    assert len(shark_art.INDEX_CHARS) == len(shark_art.PALETTE)
    assert len(set(shark_art.INDEX_CHARS)) == len(shark_art.INDEX_CHARS)
    assert len(shark_art.DURATIONS) == len(shark_art.FRAMES)


def test_no_index_character_could_end_the_literal_it_lives_in() -> None:
    """The frames are baked into triple-quoted strings by the generator."""
    assert '"' not in shark_art.INDEX_CHARS
    assert "\\" not in shark_art.INDEX_CHARS
    assert shark_art.TRANSPARENT not in shark_art.INDEX_CHARS


def test_the_waterline_is_inside_the_sprite() -> None:
    assert 0 < shark_art.WATERLINE < shark_art.HEIGHT


# ---------------------------------------------------------------------------
# Quadrant encoding
# ---------------------------------------------------------------------------
def test_four_equal_subpixels_need_no_block_glyph() -> None:
    """Most of the band is empty sky; it must collapse to spaces."""
    grey = (30, 40, 50)
    glyph, fore, back = _quadrant(grey, grey, grey, grey)
    assert glyph == " "
    assert fore == back == (30 << 16) | (40 << 8) | 50


def test_a_light_top_over_a_dark_bottom_is_an_upper_half_block() -> None:
    light, dark = (200, 240, 220), (10, 20, 30)
    glyph, fore, back = _quadrant(light, light, dark, dark)
    assert glyph == "▀"
    assert fore == (200 << 16) | (240 << 8) | 220
    assert back == (10 << 16) | (20 << 8) | 30


def test_a_single_light_corner_is_its_own_quadrant() -> None:
    light, dark = (200, 240, 220), (10, 20, 30)
    assert _quadrant(light, dark, dark, dark)[0] == "▘"
    assert _quadrant(dark, light, dark, dark)[0] == "▝"
    assert _quadrant(dark, dark, light, dark)[0] == "▖"
    assert _quadrant(dark, dark, dark, light)[0] == "▗"


def test_a_diagonal_split_keeps_both_halves() -> None:
    light, dark = (200, 240, 220), (10, 20, 30)
    assert _quadrant(light, dark, dark, light)[0] == "▚"
    assert _quadrant(dark, light, light, dark)[0] == "▞"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [40, 80, 110, 200])
@pytest.mark.parametrize("height", range(1, 60, 3))
def test_the_shark_is_never_placed_outside_the_band(width: int, height: int) -> None:
    """The sprite is anchored by its waterline, which is easy to get wrong."""
    sub_width, sub_height = width * 2, height * 2
    water_row, box = _layout(sub_width, sub_height)
    assert 0 <= water_row <= sub_height
    if box is None:
        return
    assert box.left >= 0
    assert box.left + box.width <= sub_width
    assert box.top >= 0
    assert box.top + box.height <= sub_height


def test_the_sprite_starts_on_a_cell_boundary() -> None:
    """An odd offset would put the shark half a cell over between resizes."""
    for width in (80, 90, 110, 111, 120, 200):
        _, box = _layout(width * 2, 60)
        assert box is not None
        assert box.left % 2 == 0


def test_a_band_with_no_room_for_a_shark_still_has_a_sea() -> None:
    water_row, box = _layout(160, 8)
    assert box is None
    assert 0 < water_row < 8


def test_a_tall_band_gets_a_shark() -> None:
    _, box = _layout(220, 30)
    assert box is not None


def test_the_sprite_keeps_the_artwork_proportions() -> None:
    _, box = _layout(400, 120)
    assert box is not None
    expected = box.height * shark_art.WIDTH / shark_art.HEIGHT
    assert abs(box.width - expected) <= 1


def test_the_sprite_is_never_scaled_past_the_baked_art() -> None:
    """Upscaling the master would only invent detail that is not there."""
    _, box = _layout(2000, 400)
    assert box is not None
    assert box.height <= shark_art.HEIGHT
    assert box.width <= shark_art.WIDTH


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", SIZES)
def test_every_strip_fills_exactly_the_widget_width(size: tuple[int, int]) -> None:
    """A strip that is too long or too short corrupts the whole screen."""
    width, height = size
    band = _Band(width, height)
    for y, strip in enumerate(_all_lines(band, height)):
        assert strip.cell_length == width, f"row {y} is {strip.cell_length} wide"


@pytest.mark.parametrize("size", SIZES)
def test_nothing_but_block_glyphs_reaches_the_terminal(size: tuple[int, int]) -> None:
    """Every glyph must be one the stylesheet's font requirement covers."""
    width, height = size
    band = _Band(width, height)
    allowed = set(_QUADRANTS)
    for tick in range(0, sum(shark_art.DURATIONS) + _REST_MS, 240):
        band._elapsed_ms = tick
        for strip in _all_lines(band, height):
            for segment in strip:
                unexpected = set(segment.text) - allowed
                assert not unexpected, f"emitted {unexpected}"


@pytest.mark.parametrize("size", SIZES)
def test_no_colour_the_terminal_would_reject(size: tuple[int, int]) -> None:
    """Regression: the depth ramp used to extrapolate past its last stop.

    Water deeper than the deepest stop ran the channels negative, which Rich
    rejects outright — so a tall enough terminal crashed the home screen rather
    than drawing a dark sea.
    """
    width, height = size
    band = _Band(width, height)
    # Walk a whole cycle so the resting sea and every jump frame are covered.
    for tick in range(0, sum(shark_art.DURATIONS) + _REST_MS, 240):
        band._elapsed_ms = tick
        for strip in _all_lines(band, height):
            for segment in strip:
                assert segment.style is not None
                for colour in (segment.style.color, segment.style.bgcolor):
                    assert colour is not None
                    triplet = colour.get_truecolor()
                    assert 0 <= triplet.red <= 255
                    assert 0 <= triplet.green <= 255
                    assert 0 <= triplet.blue <= 255


def test_a_band_with_no_height_draws_nothing() -> None:
    """At 80x24 minus a long menu the band can legitimately be squeezed to zero."""
    band = _Band(60, 0)
    assert band.render_line(0).cell_length == 60


def test_the_sea_moves() -> None:
    band = _Band(110, 15)
    band._elapsed_ms = 6000  # resting, so only the water can differ
    before = [str(strip) for strip in _all_lines(band, 15)]
    band._elapsed_ms = 6600
    after = [str(strip) for strip in _all_lines(band, 15)]
    assert before != after


def test_the_scene_is_built_once_per_tick() -> None:
    """Textual asks line by line; recomposing per line is twenty times the work."""
    band = _Band(110, 15)
    first = band.render_line(0)
    assert band.render_line(0) is first
    assert band.render_line(5) is not first
    band._elapsed_ms += 120
    assert band.render_line(0) is not first


# ---------------------------------------------------------------------------
# The jump cycle
# ---------------------------------------------------------------------------
def test_the_jump_plays_every_frame_then_rests() -> None:
    band = _Band(110, 15)
    seen = []
    for tick in range(0, sum(shark_art.DURATIONS) + _REST_MS, 20):
        band._elapsed_ms = tick
        seen.append(band._sprite_frame())
    assert set(range(len(shark_art.FRAMES))) <= set(seen), "a frame never showed"
    assert None in seen, "the sea never rests between jumps"


def test_the_cycle_repeats() -> None:
    band = _Band(110, 15)
    cycle = sum(shark_art.DURATIONS) + _REST_MS
    band._elapsed_ms = 0
    first = band._sprite_frame()
    band._elapsed_ms = cycle * 3
    assert band._sprite_frame() == first


# ---------------------------------------------------------------------------
# On the real screen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [(80, 24), (110, 32), (120, 40)])
def test_the_band_sits_inside_the_welcome_screen(size: tuple[int, int]) -> None:
    """It must absorb the gap without pushing the menu off the screen."""

    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            screen = app.screen
            band = screen.query_one(OceanBand)
            menu = screen.query_one("#welcome-menu")
            assert menu.region.bottom <= size[1], "the menu was pushed off screen"
            assert band.region.y >= menu.region.bottom, "the sea overlaps the menu"
            assert band.region.right <= size[0]
            assert band.region.bottom <= size[1]

    asyncio.run(scenario())


def test_the_home_screen_survives_a_resize() -> None:
    """Resizing rescales the sprite, which is the cache's only invalidation."""

    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for size in ((80, 24), (140, 45), (100, 30)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
                band = app.screen.query_one(OceanBand)
                for y in range(band.size.height):
                    assert band.render_line(y).cell_length == band.size.width

    asyncio.run(scenario())
