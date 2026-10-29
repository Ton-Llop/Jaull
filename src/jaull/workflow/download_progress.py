"""One live line for a download: a bar, the sizes, the speed and the time left.

A model of 15 GiB takes minutes to arrive and the screens only had "Downloading artifact", so
there was nothing to tell a slow download from a stuck one. The artifact service reports bytes;
this turns them into a single line that replaces itself in the step log instead of adding a new
line each time. It lives outside the TUI because both the screens and the advisor use it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# Every progress line starts like the step it belongs to, so the screens that only recognize
# that step still show it.
PROGRESS_PREFIX = "Downloading artifact "
_BAR_WIDTH = 20
_MIB = 1024**2
_GIB = 1024**3


def format_download_progress(
    done: int, total: int, elapsed_seconds: float, *, initial_bytes: int = 0,
) -> str:
    """Progress counts all bytes; speed excludes bytes present at the session's start."""
    unit, scale = ("GiB", _GIB) if max(done, total) >= _GIB else ("MiB", _MIB)
    transferred = max(0, done - initial_bytes)
    speed = transferred / elapsed_seconds if elapsed_seconds > 0 else 0.0
    speed_text = f"  {speed / _MIB:.1f} MiB/s" if speed > 0 else ""
    if total <= 0:
        return f"{PROGRESS_PREFIX}{done / scale:.1f} {unit}{speed_text}"
    fraction = min(1.0, done / total)
    eta = ""
    if speed > 0 and done < total:
        remaining = int((total - done) / speed)
        eta = f"  ETA {remaining // 60}:{remaining % 60:02d}"
    return (
        f"{PROGRESS_PREFIX}{render_bar(fraction)} {fraction * 100:.0f} %  "
        f"{done / scale:.1f} / {total / scale:.1f} {unit}{speed_text}{eta}"
    )


def render_bar(fraction: float) -> str:
    """A block bar `_BAR_WIDTH` cells wide; it is only full when the download is."""
    fraction = max(0.0, min(1.0, fraction))
    filled = _BAR_WIDTH if fraction >= 1.0 else min(_BAR_WIDTH - 1, round(fraction * _BAR_WIDTH))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def is_progress_line(message: str) -> bool:
    return message.startswith(PROGRESS_PREFIX)


def record_step(messages: list[str], message: str) -> None:
    """Append a step to a log; a progress line replaces the previous progress line."""
    if messages and is_progress_line(message) and is_progress_line(messages[-1]):
        messages[-1] = message
    else:
        messages.append(message)


class DownloadProgressReporter:
    """Turns byte counts into progress lines, a few times a second and always at the end."""

    def __init__(
        self,
        report: Callable[[str], None],
        *,
        clock: Callable[[], float] = time.monotonic,
        min_interval: float = 0.25,
    ) -> None:
        self._report = report
        self._clock = clock
        self._min_interval = min_interval
        self._started: float | None = None
        self._last: float | None = None
        self._initial_bytes = 0
        self._previous_bytes = 0

    def __call__(self, done: int, total: int) -> None:
        now = self._clock()
        if self._started is None or done < self._previous_bytes:
            self._started = now
            self._initial_bytes = done
            self._last = None
        self._previous_bytes = done
        finished = 0 < total <= done
        if self._last is not None and not finished and now - self._last < self._min_interval:
            return
        self._last = now
        self._report(format_download_progress(
            done, total, now - self._started, initial_bytes=self._initial_bytes,
        ))
