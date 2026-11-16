"""A QR code drawn with terminal blocks, so a phone can read an address off the screen."""

from __future__ import annotations

import segno

# Inverted code: an empty margin, the terminal's own background shows through it.
QR_BORDER_MODULES = 2
# Classic code: the light quiet zone around it that every scanner expects.
QR_QUIET_ZONE_MODULES = 3
# Two QR rows per text line. A filled half is what gets drawn; empty is the background.
_CELLS = {(True, True): "\u2588", (True, False): "\u2580", (False, True): "\u2584",
          (False, False): " "}


def qr_lines(text: str, *, inverted: bool = True) -> list[str]:
    """``text`` as a QR code made of blocks.

    Inverted (the default): the filled cells are the DARK modules. Draw them in the text
    colour and leave the background alone, so it follows the theme, with no white box. Phone
    cameras usually read it; some scanners do not.

    Not inverted: the classic code. The filled cells are the LIGHT modules and the quiet zone;
    draw them white on black. Every scanner reads it.
    """
    matrix = [[bool(cell) for cell in row] for row in segno.make(text, error="m").matrix]
    size = len(matrix)
    border = QR_BORDER_MODULES if inverted else QR_QUIET_ZONE_MODULES
    width = height = size + 2 * border
    if height % 2:
        height += 1  # the last text line needs a bottom half; it is margin

    def filled(x: int, y: int) -> bool:
        column, row = x - border, y - border
        dark = 0 <= column < size and 0 <= row < size and matrix[row][column]
        return dark if inverted else not dark

    return ["".join(_CELLS[(filled(x, y), filled(x, y + 1))] for x in range(width))
            for y in range(0, height, 2)]


__all__ = ["QR_BORDER_MODULES", "QR_QUIET_ZONE_MODULES", "qr_lines"]
