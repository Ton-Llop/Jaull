"""The QR drawn in the terminal must hold exactly the modules the QR library made."""

from __future__ import annotations

import segno

from jaull.presentation.qr import QR_BORDER_MODULES, QR_QUIET_ZONE_MODULES, qr_lines

# A filled half or full block is a dark module of the code; a space is background.
_FILLED = {"█": (True, True), "▀": (True, False), "▄": (False, True), " ": (False, False)}


def _modules(lines: list[str]) -> list[list[bool]]:
    """Back to a grid of dark modules, two grid rows per text line."""
    grid: list[list[bool]] = []
    for line in lines:
        top, bottom = [], []
        for char in line:
            upper, lower = _FILLED[char]
            top.append(upper)
            bottom.append(lower)
        grid += [top, bottom]
    return grid


def test_the_drawing_holds_the_same_modules_as_the_qr_with_an_empty_margin() -> None:
    url = "https://192.168.1.107:8090/"
    lines = qr_lines(url)
    expected = [[bool(cell) for cell in row] for row in segno.make(url, error="m").matrix]
    size = len(expected)
    grid = _modules(lines)

    assert len({len(line) for line in lines}) == 1  # a clean rectangle
    assert len(grid[0]) == size + 2 * QR_BORDER_MODULES
    inner = [row[QR_BORDER_MODULES:QR_BORDER_MODULES + size]
             for row in grid[QR_BORDER_MODULES:QR_BORDER_MODULES + size]]
    assert inner == expected
    # The margin is empty all around: no white box, the theme's background shows.
    assert not any(grid[0]) and not any(row[0] for row in grid)


def test_the_classic_version_is_the_normal_code_with_a_light_quiet_zone() -> None:
    url = "https://192.168.1.107:8090/"
    expected = [[bool(cell) for cell in row] for row in segno.make(url, error="m").matrix]
    size = len(expected)
    grid = _modules(qr_lines(url, inverted=False))  # here a filled cell is a LIGHT module

    zone = QR_QUIET_ZONE_MODULES
    inner = [[not cell for cell in row[zone:zone + size]] for row in grid[zone:zone + size]]
    assert inner == expected
    assert all(grid[0]) and all(row[0] for row in grid)  # the quiet zone is light all around
