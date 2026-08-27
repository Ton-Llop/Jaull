"""Bake the shark GIF into a palette-indexed sprite the TUI can draw.

Pillow is a *build-time* tool here, never a runtime dependency. The home
screen has to draw the shark with nothing but the standard library, so the
frames are quantized once, written into
``src/jaull/tui/widgets/shark_art.py``, and read back from there as plain
Python data. Re-run this only when the source GIF changes:

    uv run --python 3.12 --with pillow python scripts/bake_shark_frames.py SOURCE.gif

Two decisions are worth knowing before changing a number below.

*The tone curve is not decoration, but it is restrained.* The artwork is a
near-black body with a teal rim, drawn for a white page: 1.6% of its opaque
pixels are pure black and most of the body sits under 8% lightness. On
``$bg`` (#070a10, itself 4%) that body is within a few values of the
background and the shark disappears — all that survives is the splash and the
teeth. So the shadows are lifted just clear of the ground. What is
deliberately *not* done is recolouring: the GIF's midtones are muted teal
(#479ba0, 39% saturation) and its highlights a pale green-white, and both are
left where the artist put them.

*Alpha is kept per pixel.* The splash overlaps the animated waterline, so the
widget composites the sprite over water rather than over a known background.
Pre-flattening the frames here would ring every soft edge with dark fringes
where it crosses the sea.
"""

from __future__ import annotations

import argparse
import colorsys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "src" / "jaull" / "tui" / "widgets" / "shark_art.py"

# The GIF is a 384x512 canvas with dead sky above the arc. This is the union
# of every frame's alpha bounding box, so nothing of the animation is cut.
CROP = (0, 38, 384, 512)

# Master resolution, in *subpixels* — the widget draws with quadrant blocks, so
# a character cell holds two of these across and two down.
#
# That is why the art is not stored at the source's own proportions. A cell is
# about twice as tall as it is wide, so a quadrant subpixel is half a cell wide
# and a full half-cell tall: twice as many columns as rows are needed to cover
# the same picture. 72x44 displays as the source's 384x474 within half a
# percent, and 44 rows is also the tallest the widget will ever draw, so the
# largest shark on the largest terminal comes straight out of here 1:1.
WIDTH = 72
HEIGHT = 44

# Where the sea surface sits in the source art, read off the last frame: the
# ripple rings peak around y=428. The widget lines its waves up with this row
# so the procedural swell meets the drawn splash instead of crossing it.
WATERLINE_SOURCE_Y = 428

# Tone curve. See the module docstring for why this exists.
#
# Deliberately gentle. An earlier pass used gamma 0.62 with a saturation boost
# and a hard pull toward $accent, which did make the shark readable but turned
# the artwork into a neon cutout: the GIF's own midtones are *muted* teal
# (#479ba0 sits at 39% saturation) and its highlights are a pale green-white,
# and none of that survived. This lifts the near-black body just clear of the
# background and leaves everything above it close to what the artist drew.
GAMMA = 0.80
FLOOR = 0.11
SATURATION = 1.05
TINT = (0x00, 0xE5, 0xFF)  # $accent
TINT_AMOUNT = 0.06

# One character per subpixel, so the palette has to fit in a set of glyphs that
# are safe inside a Python string literal: no double quote (three in a row
# would end the block) and no backslash. `.` is reserved for transparent.
_INDEX_CHARS = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!#$%&'()*+,-/:;<=>?@[]^_`{|}~"
)
TRANSPARENT_CHAR = "."
ALPHA_FLOOR = 8  # below this a pixel is simply not there


def tone_map(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Lift the body off the background without flattening the highlights."""
    r, g, b = (v / 255 for v in rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = FLOOR + (1 - FLOOR) * (lightness**GAMMA)
    saturation = min(1.0, saturation * SATURATION)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Only the darks are tinted. Pulling the highlights toward the accent too
    # would erase the white of the teeth and the foam, which is what gives the
    # sprite its read at this size.
    weight = TINT_AMOUNT * (1 - lightness)
    r, g, b = (
        channel + (target / 255 - channel) * weight
        for channel, target in zip((r, g, b), TINT, strict=True)
    )
    return tuple(max(0, min(255, round(v * 255))) for v in (r, g, b))  # type: ignore[return-value]


def load_frames(source: Path) -> tuple[list[list[tuple[int, int, int, int]]], list[int]]:
    """Every frame as a flat RGBA pixel list, plus its GIF duration in ms."""
    image = Image.open(source)
    frames: list[list[tuple[int, int, int, int]]] = []
    durations: list[int] = []
    for index in range(image.n_frames):
        image.seek(index)
        # BOX is an area average. LANCZOS rings on hard pixel-art edges and at
        # this reduction the halos read as noise.
        resized = image.convert("RGBA").crop(CROP).resize((WIDTH, HEIGHT), Image.BOX)
        sample = resized.load()
        pixels: list[tuple[int, int, int, int]] = []
        for y in range(HEIGHT):
            for x in range(WIDTH):
                r, g, b, a = sample[x, y]
                pixels.append(
                    (0, 0, 0, 0) if a < ALPHA_FLOOR else (*tone_map((r, g, b)), a)
                )
        frames.append(pixels)
        durations.append(int(image.info.get("duration", 300)))
    return frames, durations


def build_palette(
    frames: Iterable[list[tuple[int, int, int, int]]],
) -> list[tuple[int, int, int, int]]:
    """The most common colours, colour and alpha quantized together.

    Alpha rides in the palette rather than in a parallel array because the
    soft edges only take a handful of distinct values once the art is this
    small — splitting them would double the file for no fidelity.
    """
    counts: Counter[tuple[int, int, int, int]] = Counter()
    for pixels in frames:
        for r, g, b, a in pixels:
            if a == 0:
                continue
            # A coarse grid on purpose. The body is a large flat field of
            # near-identical teals; bucketed finely it wins every slot by sheer
            # area and starves the teeth, foam and rim highlights that are what
            # actually make the silhouette readable.
            counts[(r >> 3 << 3, g >> 3 << 3, b >> 3 << 3, _quantize_alpha(a))] += 1
    return [colour for colour, _ in counts.most_common(len(_INDEX_CHARS))]


def _quantize_alpha(alpha: int) -> int:
    """Snap to a few levels, keeping 255 exactly.

    Opaque has to survive as opaque: rounding it down to 240 would make the
    whole shark faintly translucent and let the waves show through its body.
    """
    levels = (0, 64, 128, 192, 255)
    return min(levels, key=lambda level: abs(level - alpha))


def nearest(
    palette: list[tuple[int, int, int, int]], colour: tuple[int, int, int, int]
) -> int:
    """Closest palette entry, weighting alpha so a soft edge stays soft."""
    r, g, b, a = colour
    best_index, best_distance = 0, None
    for index, (pr, pg, pb, pa) in enumerate(palette):
        distance = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2 + 3 * (a - pa) ** 2
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def encode(
    frames: list[list[tuple[int, int, int, int]]],
    palette: list[tuple[int, int, int, int]],
) -> list[list[str]]:
    cache: dict[tuple[int, int, int, int], str] = {}
    encoded: list[list[str]] = []
    for pixels in frames:
        rows: list[str] = []
        for y in range(HEIGHT):
            row: list[str] = []
            for x in range(WIDTH):
                colour = pixels[y * WIDTH + x]
                if colour[3] == 0:
                    row.append(TRANSPARENT_CHAR)
                    continue
                if colour not in cache:
                    cache[colour] = _INDEX_CHARS[nearest(palette, colour)]
                row.append(cache[colour])
            rows.append("".join(row))
        encoded.append(rows)
    return encoded


def render_module(
    encoded: list[list[str]],
    palette: list[tuple[int, int, int, int]],
    durations: list[int],
    source: Path,
) -> str:
    waterline = round(
        (WATERLINE_SOURCE_Y - CROP[1]) / (CROP[3] - CROP[1]) * HEIGHT
    )
    # Split across lines so the generated file stays inside the line limit.
    used = _INDEX_CHARS[: len(palette)]
    index_chars = "\n".join(
        f'    "{used[start : start + 48]}"' for start in range(0, len(used), 48)
    )
    colours = "\n".join(
        f'    ("#{r:02x}{g:02x}{b:02x}", {a}),' for r, g, b, a in palette
    )
    frames = "\n".join(
        '    """\\\n' + "\n".join(rows) + '\n""",' for rows in encoded
    )
    return f'''"""Baked frames of the Jaull shark. Generated — do not edit by hand.

Regenerate with::

    uv run --python 3.12 --with pillow python scripts/bake_shark_frames.py \\
        {source.name}

Each frame is {HEIGHT} rows of {WIDTH} characters, one per quadrant
subpixel: a character cell holds two across and two down. A character indexes
:data:`PALETTE`; ``{TRANSPARENT_CHAR}`` is transparent. See
``scripts/bake_shark_frames.py`` for why the art is toned the way it is and
why alpha survives into this file.
"""

from __future__ import annotations

WIDTH = {WIDTH}
HEIGHT = {HEIGHT}

#: Subpixel row of the sea surface. The waves meet the drawn splash here.
WATERLINE = {waterline}

#: Index character -> (hex colour, alpha 0-255).
PALETTE: tuple[tuple[str, int], ...] = (
{colours}
)

#: Index characters, in palette order. ``INDEX_CHARS.index(ch)`` reads a subpixel.
INDEX_CHARS = (
{index_chars}
)

TRANSPARENT = "{TRANSPARENT_CHAR}"

#: Milliseconds each frame is held, straight from the GIF.
DURATIONS: tuple[int, ...] = ({", ".join(str(d) for d in durations)},)

FRAMES: tuple[str, ...] = (
{frames}
)

__all__ = [
    "DURATIONS",
    "FRAMES",
    "HEIGHT",
    "INDEX_CHARS",
    "PALETTE",
    "TRANSPARENT",
    "WATERLINE",
    "WIDTH",
]
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the shark GIF")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    frames, durations = load_frames(args.source)
    palette = build_palette(frames)
    encoded = encode(frames, palette)
    args.output.write_text(
        render_module(encoded, palette, durations, args.source), encoding="utf-8"
    )
    print(
        f"{args.output.relative_to(ROOT)}: {len(encoded)} frames, "
        f"{WIDTH}x{HEIGHT}, {len(palette)} colours"
    )


if __name__ == "__main__":
    main()
