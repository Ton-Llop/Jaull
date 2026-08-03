from __future__ import annotations

import shutil

from rich.console import Console


def make_console() -> Console:
    # When stdout is piped Rich falls back to a very narrow default width, which
    # collapses two-column tables. Detect the real terminal width up front and, when
    # unavailable, pick a sensible default so tables stay readable.
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    return Console(width=max(width, 80))


def format_bytes(value: int | None) -> str:
    if value is None or value < 0:
        return "unknown"
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(value)
    for unit in units:
        if size < step:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} EiB"
