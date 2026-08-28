"""Guards for the search screen's waiting animation.

The lane is decorative, so nothing here asserts what it looks like. What it
asserts is the shape of the loop the user is promised — across, pause, back,
pause — and that a widget which sits on screen for minutes cannot break the
screen it sits on.

The quadrant encoder, the strip cache and the baked art are covered in
``test_tui_subpixel.py``; this is about the crossing.

Sizes are given in characters; the widget works in subpixels, two to a
character in each direction.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.geometry import Size

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from capture_screenshots import Gates, build_app  # noqa: E402

from jaull.tui.screens.model_discovery import ModelDiscoveryScreen  # noqa: E402
from jaull.tui.widgets.patrol import (  # noqa: E402
    _CROSSING_MS,
    _CYCLE_MS,
    _PASSES,
    _PAUSE_MS,
    SearchPatrol,
)
from jaull.tui.widgets.subpixel import BG_RGB, QUADRANTS  # noqa: E402

#: The lane takes whatever the checklist leaves, so these are the heights it
#: plausibly gets — a roomy terminal, a cramped one, and the squeezes it has to
#: survive rather than sizes it is designed for.
SIZES = [(110, 14), (80, 9), (200, 7), (90, 4), (60, 1)]

SWIM, FIN = _PASSES


class _Lane(SearchPatrol):
    """An unmounted lane with a size, so drawing can be tested without an app."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self._fixed = Size(width, height)

    @property
    def size(self) -> Size:
        return self._fixed


def _at(elapsed: int, width: int = 110, height: int = 14) -> _Lane:
    lane = _Lane(width, height)
    lane._elapsed_ms = elapsed
    return lane


def _ink(lane: _Lane) -> tuple[int, int] | None:
    """First and last subpixel column that something was drawn into."""
    rows = lane.paint(lane.size.width * 2, lane.size.height * 2)
    columns = [
        x
        for row in rows
        for x, pixel in enumerate(row)
        if pixel is not BG_RGB and pixel != BG_RGB
    ]
    return (min(columns), max(columns)) if columns else None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def test_the_cycle_is_across_pause_back_pause() -> None:
    assert _CYCLE_MS == 2 * (_CROSSING_MS + _PAUSE_MS)
    assert _at(10)._phase()[0] is SWIM  # type: ignore[index]
    assert _at(_CROSSING_MS - 10)._phase()[0] is SWIM  # type: ignore[index]
    assert _at(_CROSSING_MS + 10)._phase() is None
    assert _at(_CROSSING_MS + _PAUSE_MS + 10)._phase()[0] is FIN  # type: ignore[index]
    assert _at(_CYCLE_MS - 10)._phase() is None


def test_the_pause_is_long_enough_to_read_as_one() -> None:
    """The point of it is that the water is empty for a while, not a blink."""
    assert 3000 <= _PAUSE_MS <= 4500


def test_the_water_is_empty_between_crossings() -> None:
    for elapsed in range(_CROSSING_MS + 40, _CROSSING_MS + _PAUSE_MS, 200):
        assert _ink(_at(elapsed)) is None, f"something was drawn at {elapsed}ms"


def test_the_cycle_repeats() -> None:
    early, late = _at(700), _at(_CYCLE_MS * 3 + 700)
    assert _ink(early) == _ink(late)


def test_the_shark_goes_right_and_the_fin_comes_back_left() -> None:
    """The whole point of having two sprites, so it is worth pinning down."""
    swim = [_ink(_at(ms)) for ms in range(200, _CROSSING_MS, 400)]
    assert all(span is not None for span in swim)
    lefts = [span[0] for span in swim if span is not None]
    assert lefts == sorted(lefts) and lefts[0] < lefts[-1], "the shark did not advance"

    start = _CROSSING_MS + _PAUSE_MS
    back = [_ink(_at(ms)) for ms in range(start + 200, start + _CROSSING_MS, 400)]
    assert all(span is not None for span in back)
    rights = [span[1] for span in back if span is not None]
    assert rights == sorted(rights, reverse=True), "the fin did not come back"


def test_each_crossing_starts_and_ends_off_the_lane() -> None:
    """Entering at the margin would read as the shark blinking into existence."""
    assert _ink(_at(0)) is None
    assert _ink(_at(_CROSSING_MS - 1)) is None
    start = _CROSSING_MS + _PAUSE_MS
    assert _ink(_at(start)) is None
    assert _ink(_at(start + _CROSSING_MS - 1)) is None


@pytest.mark.parametrize("crossing", _PASSES, ids=["swim", "fin"])
def test_every_baked_frame_gets_shown(crossing: Any) -> None:
    """A frame that never renders is dead weight in the repository."""
    sprite = crossing.sprite
    seen = {
        SearchPatrol._sprite_frame(sprite, elapsed)
        for elapsed in range(0, _CROSSING_MS, 10)
    }
    assert seen == set(range(sprite.frame_count))


def test_the_sprite_loops_while_it_crosses() -> None:
    """The art is a swim cycle; the crossing is longer than one turn of it."""
    for crossing in _PASSES:
        assert sum(crossing.sprite.durations) < _CROSSING_MS


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", SIZES)
def test_nothing_but_block_glyphs_reaches_the_terminal(size: tuple[int, int]) -> None:
    """Every glyph must be one the stylesheet's font requirement covers."""
    width, height = size
    allowed = set(QUADRANTS)
    for elapsed in range(0, _CYCLE_MS, 220):
        lane = _at(elapsed, width, height)
        for y in range(height):
            for segment in lane.render_line(y):
                unexpected = set(segment.text) - allowed
                assert not unexpected, f"emitted {unexpected}"


@pytest.mark.parametrize("size", SIZES)
def test_no_colour_the_terminal_would_reject(size: tuple[int, int]) -> None:
    width, height = size
    for elapsed in range(0, _CYCLE_MS, 220):
        lane = _at(elapsed, width, height)
        for y in range(height):
            for segment in lane.render_line(y):
                assert segment.style is not None
                for colour in (segment.style.color, segment.style.bgcolor):
                    assert colour is not None
                    triplet = colour.get_truecolor()
                    assert 0 <= triplet.red <= 255
                    assert 0 <= triplet.green <= 255
                    assert 0 <= triplet.blue <= 255


@pytest.mark.parametrize("size", SIZES)
def test_every_strip_fills_exactly_the_widget_width(size: tuple[int, int]) -> None:
    width, height = size
    lane = _at(900, width, height)
    for y in range(height):
        assert lane.render_line(y).cell_length == width, f"row {y} is wrong"


def test_a_squeezed_lane_shrinks_the_shark_rather_than_cropping_it() -> None:
    """Two rows is not a design target, but a short terminal can produce it."""
    tall, short = _ink(_at(2500, 110, 14)), _ink(_at(2500, 110, 2))
    assert tall is not None and short is not None
    assert short[1] - short[0] < tall[1] - tall[0]


def test_a_lane_with_no_height_draws_nothing() -> None:
    lane = _at(900, 60, 0)
    assert lane.render_line(0).cell_length == 60


def test_the_lane_never_writes_past_its_own_width() -> None:
    """The sprite is wider than a narrow terminal for most of its crossing."""
    for elapsed in range(0, _CYCLE_MS, 130):
        rows = _at(elapsed, 30, 7).paint(60, 14)
        assert all(len(row) == 60 for row in rows)


def test_the_fin_is_drawn_a_touch_larger_than_the_swim() -> None:
    """Same animal, same distance — plus the deliberate bump in _Crossing.scale.

    Without the bump the two would draw at the same per-source-pixel scale; the
    fin carries a small enlargement so its bare tip still reads as a fin.
    """
    swim = _ink(_at(_CROSSING_MS // 2, 200, 14))
    back = _ink(_at(_CROSSING_MS + _PAUSE_MS + _CROSSING_MS // 2, 200, 14))
    assert swim is not None and back is not None
    swim_length = (swim[1] - swim[0]) / SWIM.sprite.width
    fin_length = (back[1] - back[0]) / FIN.sprite.width
    assert FIN.scale > 1.0, "the fin is supposed to be bumped up"
    assert abs(fin_length - swim_length * FIN.scale) < 0.08


def test_the_swim_sits_low_in_the_lane_but_not_at_the_bottom() -> None:
    """It hangs under a checklist; starting at its last line reads as part of it."""
    rows = _at(_CROSSING_MS // 2, 110, 20).paint(220, 40)
    inked = [y for y, row in enumerate(rows) if any(p != BG_RGB for p in row)]
    assert inked, "nothing was drawn mid-crossing"
    assert min(inked) > len(rows) * 0.25, "the swim is crowding the checklist"
    assert max(inked) <= len(rows) - 1, "the swim ran off the bottom"


# ---------------------------------------------------------------------------
# On the real screen
# ---------------------------------------------------------------------------
async def _wait_for(
    pilot: Any, predicate: Callable[[], bool], *, timeout: float = 30.0
) -> None:
    waited = 0.0
    while waited < timeout:
        await pilot.pause()
        if predicate():
            await pilot.pause()
            return
        await asyncio.sleep(0.02)
        waited += 0.02
    raise AssertionError("timed out waiting for the search screen")


@pytest.mark.parametrize("size", [(80, 24), (110, 32), (120, 40)])
def test_the_lane_sits_inside_the_search_screen(size: tuple[int, int]) -> None:
    """It must absorb the gap without pushing Cancel off the screen."""

    async def scenario() -> None:
        gates = Gates()
        app = build_app(gates)
        async with app.run_test(size=size) as pilot:
            try:
                gates.search.hold()
                app.start_guided_workflow()
                await _wait_for(pilot, lambda: bool(app.screen.query("#hw-continue")))
                app.goto_requirements()
                await _wait_for(pilot, lambda: bool(app.screen.query("#wizard-submit")))
                answers = app.screen.collect_answers()  # type: ignore[attr-defined]
                app.start_discovery(answers)
                await _wait_for(
                    pilot, lambda: isinstance(app.screen, ModelDiscoveryScreen)
                )
                await _wait_for(pilot, lambda: bool(app.screen.query(SearchPatrol)))

                lane = app.screen.query_one(SearchPatrol)
                cancel = app.screen.query_one("#discovery-actions")
                assert cancel.region.bottom <= size[1], "Cancel was pushed off screen"
                assert lane.region.y >= cancel.region.bottom, "the lane overlaps it"
                assert lane.region.right <= size[0]
                assert lane.region.bottom <= size[1]
            finally:
                gates.release_all()

    asyncio.run(scenario())


def test_the_lane_stops_when_the_search_does() -> None:
    """A shark still patrolling under a cancellation notice says the opposite."""

    async def scenario() -> None:
        gates = Gates()
        app = build_app(gates)
        async with app.run_test(size=(110, 32)) as pilot:
            try:
                gates.search.hold()
                app.start_guided_workflow()
                await _wait_for(pilot, lambda: bool(app.screen.query("#hw-continue")))
                app.goto_requirements()
                await _wait_for(pilot, lambda: bool(app.screen.query("#wizard-submit")))
                answers = app.screen.collect_answers()  # type: ignore[attr-defined]
                app.start_discovery(answers)
                await _wait_for(
                    pilot, lambda: isinstance(app.screen, ModelDiscoveryScreen)
                )
                screen = app.screen
                assert isinstance(screen, ModelDiscoveryScreen)
                assert screen.query_one(SearchPatrol).display

                screen.action_cancel()
                gates.search.release()
                await _wait_for(
                    pilot, lambda: bool(screen.query("#discovery-restart"))
                )
                assert not screen.query_one(SearchPatrol).display
            finally:
                gates.release_all()

    asyncio.run(scenario())
