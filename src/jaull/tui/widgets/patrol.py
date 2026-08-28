"""The waiting animation for the search screen: a shark on patrol.

Finding models is the one part of a guided run that genuinely takes minutes —
a Hub search, then a metadata round-trip and a header read for each shortlisted
repository. The checklist above says what is happening, but it only moves every
twenty or thirty seconds, and between ticks the screen was completely still.
Still is what a stalled program looks like.

So the gap under the checklist gets a lane of open water. The shark crosses it
left to right, the water is empty for a beat, and then only its fin comes back
the other way. Then it goes quiet again. That is the whole animation: no
progress is implied by it, because none is known between checklist ticks.

Both sprites come from GIFs that animate the crossing themselves, on a canvas
much wider than the subject. :mod:`scripts.bake_shark_frames` cuts the subject
out of that canvas frame by frame, which leaves a short loop of it swimming on
the spot and hands the crossing to this module. That split is what lets the
shark stay the same size on an 80-column terminal and a 200-column one, and
lets it be timed for a long wait rather than for a 2.5-second GIF.

It is drawn in quadrant blocks; see :mod:`jaull.tui.widgets.subpixel`.
"""

from __future__ import annotations

from typing import NamedTuple

from jaull.tui.widgets import fin_art, swim_art
from jaull.tui.widgets.subpixel import BG_RGB, RGB, Sprite, SubpixelWidget


class _Crossing(NamedTuple):
    """One traverse of the lane: which sprite, which way it goes, how big."""

    sprite: Sprite
    rightward: bool
    #: Extra size on top of the shared base scale. The swim keeps 1.0: it is the
    #: reference the distance is set from. Strictly the fin is the same animal at
    #: the same distance and would keep 1.0 too, but the fin that comes back is a
    #: bare triangle a handful of subpixels tall, and a small bump is what keeps
    #: it reading as a fin rather than as a nick in the water.
    scale: float = 1.0


_PASSES = (
    _Crossing(Sprite(swim_art), rightward=True),
    _Crossing(Sprite(fin_art), rightward=False, scale=1.2),
)

#: How long a sprite takes to cross, and how long the water stays empty after
#: it. Both are much slower than the source GIF, which crosses in 2.5 seconds
#: and never rests. This sits on screen for minutes at a time: something
#: darting past every few seconds reads as a progress bar that is lying, where
#: a slow pass and a long empty stretch read as water that happens to have a
#: shark in it.
_CROSSING_MS = 5000
_PAUSE_MS = 3500

_CYCLE_MS = len(_PASSES) * (_CROSSING_MS + _PAUSE_MS)

#: Subpixel rows the reference sprite is drawn at, and the sprite everything is
#: measured against. Both sprites take *one* base scale, from this one: sizing
#: each to its own baked height would draw the fin at a different distance from
#: the shark that just went past, and they are meant to be the same animal —
#: give or take the small deliberate bump in :attr:`_Crossing.scale`.
#:
#: It is the baked art's own height, so the reference is never resampled on a
#: terminal with room for it. Smaller was tried and the dorsal fin — a dark
#: triangle perhaps four subpixels across — stopped being identifiable as one,
#: which is also why the fin that comes back is nudged up a little from here.
_LANE_ROWS = 18
_REFERENCE = _PASSES[0].sprite

#: Where the swim sits in the band, as a fraction of the room left over. Below
#: the middle: the lane hangs under a checklist, and water that starts right at
#: its last line reads as part of it rather than as the surface below it.
_DEPTH = 0.6


class SearchPatrol(SubpixelWidget):
    """The search screen's waiting animation. Decorative: it holds no state."""

    DEFAULT_CLASSES = "patrol"

    # Twice the GIF's own frame rate. The art only needs 140ms a frame, but the
    # position is continuous, and stepping it at 14 frames a second rather than
    # 7 is the difference between swimming and skipping across the lane.
    TICK_SECONDS = 0.07
    TICK_MS = 70

    # -- animation clock ----------------------------------------------------
    def _phase(self) -> tuple[_Crossing, int] | None:
        """The crossing under way and how far into it, or ``None`` between them."""
        elapsed = self._elapsed_ms % _CYCLE_MS
        for crossing in _PASSES:
            if elapsed < _CROSSING_MS:
                return crossing, elapsed
            elapsed -= _CROSSING_MS + _PAUSE_MS
            if elapsed < 0:
                return None
        return None

    @staticmethod
    def _sprite_frame(sprite: Sprite, elapsed: int) -> int:
        """Which frame of the swim loop is showing, at the GIF's own pace."""
        phase = elapsed % sum(sprite.durations)
        for index, duration in enumerate(sprite.durations):
            if phase < duration:
                return index
            phase -= duration
        return len(sprite.durations) - 1

    # -- the scene ----------------------------------------------------------
    def paint(self, width: int, height: int) -> list[list[RGB]]:
        """Empty water, with whichever sprite is crossing it drawn on top."""
        rows = [[BG_RGB] * width for _ in range(height)]
        phase = self._phase()
        if phase is None or height <= 0:
            return rows
        crossing, elapsed = phase
        sprite = crossing.sprite

        # One shared scale sets the distance for both sprites, and the baked art
        # is only ever shrunk from there — never blown up past its stored
        # resolution — save for the small per-crossing bump the fin carries.
        scale = min(_LANE_ROWS, height) / _REFERENCE.height * crossing.scale
        drawn_height = round(sprite.height * scale)
        drawn_width = round(sprite.width * scale)
        if drawn_height <= 0 or drawn_width <= 0:
            return rows

        # The crossing runs from fully off one edge to fully off the other, so
        # the sprite enters and leaves cleanly instead of appearing at the
        # margin. The lane clips it on the way in and out, which is exactly
        # what the GIF's own canvas did before the frames were cut out of it.
        progress = elapsed / _CROSSING_MS
        travelled = round(progress * (width + drawn_width))
        left = travelled - drawn_width if crossing.rightward else width - travelled
        # Deliberately not snapped to an even column: an odd offset puts the
        # sprite half a cell over, which the quadrant renderer draws exactly
        # and which halves the size of each step across the lane.
        top = round((height - drawn_height) * _DEPTH)
        sprite.composite(
            rows,
            self._sprite_frame(sprite, elapsed),
            left,
            top,
            drawn_width,
            drawn_height,
        )
        return rows


__all__ = ["SearchPatrol"]
