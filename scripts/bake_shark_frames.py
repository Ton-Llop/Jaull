"""Bake a shark GIF into a palette-indexed sprite the TUI can draw.

Pillow is a *build-time* tool here, never a runtime dependency. The screens
have to draw these with nothing but the standard library, so the frames are
quantized once, written into ``src/jaull/tui/widgets/``, and read back from
there as plain Python data.

Three animations come out of this, listed in :data:`SPRITES`. They share every
decision below; the crop, the resolution and the anchor are all a preset
carries::

    uv run --python 3.12 --with pillow python scripts/bake_shark_frames.py \
        jump jaull_jump_clean.gif
    uv run --python 3.12 --with pillow python scripts/bake_shark_frames.py \
        swim jaull_search_swim_right.gif
    uv run --python 3.12 --with pillow python scripts/bake_shark_frames.py \
        fin jaull_search_fin_left.gif

Three decisions are worth knowing before changing a number below.

*The tone curve is not decoration, but it is restrained.* The artwork is a
near-black body with a teal rim, drawn for a white page: most of the body sits
under 8% lightness, and on ``$bg`` (#070a10, itself 4%) that is within a few
values of the background — the shark disappears and only the wake survives. So
the shadows are lifted just clear of the ground, and nothing else is
recoloured: the midtones are muted teal (#479ba0, 39% saturation) and the
highlights a pale green-white, and both are left where the artist put them.

There are two lifts, and which one a preset uses is not a style choice. See
:data:`LIFTS`.

*Alpha is kept per pixel.* Every sprite lands on something the widget drew
first, so none of them can be pre-flattened against a known background without
ringing every soft edge with dark fringes.

*A sprite that crosses the screen is cut out of its canvas, not baked with
it.* The two search GIFs animate the traverse themselves: the subject slides
across a 420px canvas, most of which is empty water in any given frame. Baking
the canvas would spend the file on that emptiness **and** tie the shark's size
to the width of the terminal, so a wide window would get an absurdly long
shark. Instead each frame is cropped to a window that tracks the subject, and
the widget moves it. See :class:`Travel`.
"""

from __future__ import annotations

import argparse
import colorsys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WIDGETS = ROOT / "src" / "jaull" / "tui" / "widgets"


class Travel(NamedTuple):
    """How a subject that crosses its canvas is cut out of each frame.

    Each frame is cropped to a window anchored on the subject's leading edge —
    the nose of the swimming shark, the front of the fin's bow wave — so the
    subject sits still in its window while its body and wake keep animating.
    The traverse then becomes a property of the band rather than of the GIF:
    the widget places the sprite where it likes, at whatever size, and a wide
    terminal gets a shark that crosses it rather than one stretched across it.

    Frames where the subject touches an edge of the canvas are **dropped**.
    There the GIF is clipping its own subject, and baking that cut in would
    leave a shark with its nose sliced off floating in open water, because the
    sprite no longer covers the same fraction of the band that it covered of
    the canvas. The band clips at its own edges instead, which is the same
    picture and always in the right place. What survives is a short loop of
    the subject entirely in frame, which is all the animation there is: these
    GIFs animate their traverse, and the body barely moves within it.
    """

    #: Which edge of the subject leads: ``"right"`` swims right, ``"left"``
    #: comes back.
    lead: str
    #: Source pixels of subject and wake to keep behind the leading edge.
    window: int
    #: Source rows kept, the same for every frame, so vertical motion survives.
    rows: tuple[int, int]


class Sprite(NamedTuple):
    """One animation: what to cut out of the GIF and how big to store it.

    ``width`` and ``height`` are in *subpixels* — every widget here draws with
    quadrant blocks, so a character cell holds two across and two down.

    That is why no sprite is stored at its source's own proportions. A cell is
    about twice as tall as it is wide, so a quadrant subpixel is half a cell
    wide and a full half-cell tall: twice as many columns as rows are needed to
    cover the same picture.
    """

    module: str
    #: Fixed crop, for a sprite that stays where the artist put it.
    crop: tuple[int, int, int, int] | None
    #: Tracking crop, for a sprite that crosses its canvas. Excludes ``crop``.
    travel: Travel | None
    width: int
    height: int
    #: Row of the sea surface in *source* pixels, for a sprite that meets water.
    waterline_source_y: int | None
    #: Which shadow lift this source needs. A key of :data:`LIFTS`.
    lift: str
    #: Short subject, for the generated module's first docstring line.
    headline: str
    #: Who draws it, for the line under that.
    drawn_by: str


# The two search sprites are deliberately baked at one shared scale — 18
# subpixel rows to the swimming shark's 83 source rows — so the fin that comes
# back is the same animal at the same distance as the one that went past.
#
# 18 rather than something smaller because the wake is drawn as thin bright
# lines and scattered spray: under about sixteen rows the area average turns
# both into an even speckle and the fin stops reading as cutting water.
_SEARCH_ROWS = 18
_SEARCH_SOURCE_ROWS = 83


def _search_size(window: int, rows: int) -> tuple[int, int]:
    """Subpixel size of a search sprite, at the scale both of them share."""
    scale = _SEARCH_ROWS / _SEARCH_SOURCE_ROWS
    # Doubled across, because a subpixel is half a cell wide and a whole
    # half-cell tall. Rounded to even columns so the sprite lands on cell
    # boundaries as it moves.
    return round(window * scale * 2 / 2) * 2, round(rows * scale)


SPRITES: dict[str, Sprite] = {
    # The union of every frame's alpha bounding box, so nothing is cut. 72x44
    # displays the source's 384x474 within half a percent, and 44 rows is also
    # the tallest the band will ever draw, so the largest shark on the largest
    # terminal comes straight out of the file 1:1.
    "jump": Sprite(
        module="shark_art",
        crop=(0, 38, 384, 512),
        travel=None,
        width=72,
        height=44,
        # Read off the last frame: the ripple rings peak around y=428. The
        # ocean lines its waves up with this row so the procedural swell meets
        # the drawn splash instead of crossing it.
        waterline_source_y=428,
        lift="lightness",
        headline="the shark breaching",
        drawn_by="Drawn by :mod:`jaull.tui.widgets.ocean` over the home screen's sea.",
    ),
    # 276 is a few pixels over the widest the shark and its wake ever get, so
    # the window never clips the tail of the trail.
    "swim": Sprite(
        module="swim_art",
        crop=None,
        travel=Travel(lead="right", window=276, rows=(21, 104)),
        width=_search_size(276, 83)[0],
        height=_search_size(276, 83)[1],
        waterline_source_y=None,
        lift="channels",
        headline="the shark crossing to the right",
        drawn_by="Drawn by :mod:`jaull.tui.widgets.patrol` while the search runs.",
    ),
    "fin": Sprite(
        module="fin_art",
        crop=None,
        travel=Travel(lead="left", window=188, rows=(28, 98)),
        width=_search_size(188, 70)[0],
        height=_search_size(188, 70)[1],
        waterline_source_y=None,
        lift="channels",
        headline="the fin coming back to the left",
        drawn_by="Drawn by :mod:`jaull.tui.widgets.patrol` while the search runs.",
    ),
}

# Tone curve. See the module docstring for why the shadows are lifted at all.
#
# Deliberately gentle. An earlier pass used gamma 0.62 with a saturation boost
# and a hard pull toward $accent, which did make the shark readable but turned
# the artwork into a neon cutout, and none of the artist's muted midtones
# survived. These lift the near-black body just clear of the background and
# leave everything above it close to what was drawn.
SATURATION = 1.05
TINT = (0x00, 0xE5, 0xFF)  # $accent
TINT_AMOUNT = 0.06

_LIGHTNESS_GAMMA = 0.80
_LIGHTNESS_FLOOR = 0.11
# Brighter than the jump's lift, and deliberately so. That one sits on top of
# a lit sea, which gives the silhouette its edge for free; these two cross bare
# background at a fifth of the size, where the same values left the fin reading
# as a smudge rather than as something cutting water.
_CHANNEL_GAMMA = 0.78
_CHANNEL_FLOOR = 0.16


def _lift_lightness(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Raise the colour's lightness, holding its hue and saturation."""
    hue, lightness, saturation = colorsys.rgb_to_hls(*(v / 255 for v in rgb))
    lightness = _LIGHTNESS_FLOOR + (1 - _LIGHTNESS_FLOOR) * (
        lightness**_LIGHTNESS_GAMMA
    )
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def _lift_channels(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Add a little of every channel, tapering to nothing by white."""
    channels = [(v / 255) ** _CHANNEL_GAMMA for v in rgb]
    red, green, blue = (c + _CHANNEL_FLOOR * (1 - c) for c in channels)
    return red, green, blue


#: The two ways a preset can lift its shadows, and it is not a style choice.
#:
#: ``lightness`` is the obvious one and it is right for the jump, whose darks
#: are drawn with all three channels non-zero. The search GIFs paint theirs
#: with red at exactly zero, which HLS reports as *fully saturated* no matter
#: how dark it is — so raising the lightness of a near-black body walks it up
#: to pure neon cyan instead of to a lighter version of itself, and the shark
#: and its wake collapse into one colour. ``channels`` adds a little of every
#: channel instead, which keeps a dark teal a dark teal.
LIFTS = {"lightness": _lift_lightness, "channels": _lift_channels}

# One character per subpixel, so the palette has to fit in a set of glyphs that
# are safe inside a Python string literal: no double quote (three in a row
# would end the block) and no backslash. `.` is reserved for transparent.
_INDEX_CHARS = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!#$%&'()*+,-/:;<=>?@[]^_`{|}~"
)
TRANSPARENT_CHAR = "."
ALPHA_FLOOR = 8  # below this a pixel is simply not there


def tone_map(rgb: tuple[int, int, int], lift: str) -> tuple[int, int, int]:
    """Lift the body off the background without flattening the highlights."""
    r, g, b = LIFTS[lift](rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    saturation = min(1.0, saturation * SATURATION)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Only the darks are tinted. Pulling the highlights toward the accent too
    # would erase the white of the teeth and the foam, which is what gives the
    # sprite its read at these sizes.
    weight = TINT_AMOUNT * (1 - lightness)
    r, g, b = (
        channel + (target / 255 - channel) * weight
        for channel, target in zip((r, g, b), TINT, strict=True)
    )
    return tuple(max(0, min(255, round(v * 255))) for v in (r, g, b))  # type: ignore[return-value]


def _alpha_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    """The bounding box of everything solid enough to count as drawn."""
    alpha = image.convert("RGBA").getchannel("A")
    return alpha.point(lambda value: 255 if value > ALPHA_FLOOR else 0).getbbox()


Crop = tuple[int, int, int, int]


def frame_crops(image: Image.Image, sprite: Sprite) -> list[tuple[int, Crop]]:
    """Every frame that is kept, as its source index and the box cut out of it."""
    if sprite.travel is None:
        assert sprite.crop is not None
        return [(index, sprite.crop) for index in range(image.n_frames)]

    travel = sprite.travel
    top, bottom = travel.rows
    kept: list[tuple[int, Crop]] = []
    for index in range(image.n_frames):
        image.seek(index)
        box = _alpha_box(image)
        if box is None or box[0] == 0 or box[2] == image.width:
            continue  # running off the canvas; see Travel for why it is dropped
        edge = box[2] if travel.lead == "right" else box[0]
        kept.append(
            (index, (edge - travel.window, top, edge, bottom))
            if travel.lead == "right"
            else (index, (edge, top, edge + travel.window, bottom))
        )
    if len(kept) < 2:
        raise SystemExit(
            f"{sprite.module}: only {len(kept)} frames hold the whole subject, "
            "which is not an animation. Check Travel.rows and the canvas size."
        )
    return kept


def load_frames(
    source: Path, sprite: Sprite
) -> tuple[list[list[tuple[int, int, int, int]]], list[int]]:
    """Every frame as a flat RGBA pixel list, plus its GIF duration in ms."""
    image = Image.open(source)
    frames: list[list[tuple[int, int, int, int]]] = []
    durations: list[int] = []
    for index, crop in frame_crops(image, sprite):
        image.seek(index)
        # BOX is an area average. LANCZOS rings on hard pixel-art edges and at
        # this reduction the halos read as noise.
        resized = (
            image.convert("RGBA")
            .crop(crop)
            .resize((sprite.width, sprite.height), Image.BOX)
        )
        sample = resized.load()
        pixels: list[tuple[int, int, int, int]] = []
        for y in range(sprite.height):
            for x in range(sprite.width):
                r, g, b, a = sample[x, y]
                pixels.append(
                    (0, 0, 0, 0) if a < ALPHA_FLOOR else (*tone_map((r, g, b), sprite.lift), a)
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
    """Snap to a few levels, keeping both ends of the range honest.

    Opaque has to survive as opaque: rounding 255 down to 240 would make the
    whole shark faintly translucent and let the waves show through its body.

    And nothing here may round down to *nothing*. :data:`ALPHA_FLOOR` has
    already decided what counts as not drawn, so a level at zero would let the
    quantizer overrule it — which it did, and it cost the fin eleven of its
    ninety-one palette slots on colours that render as empty space, along with
    the faintest of the spray that makes the wake read as water.
    """
    levels = (32, 64, 128, 192, 255)
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
    sprite: Sprite,
) -> list[list[str]]:
    cache: dict[tuple[int, int, int, int], str] = {}
    encoded: list[list[str]] = []
    for pixels in frames:
        rows: list[str] = []
        for y in range(sprite.height):
            row: list[str] = []
            for x in range(sprite.width):
                colour = pixels[y * sprite.width + x]
                if colour[3] == 0:
                    row.append(TRANSPARENT_CHAR)
                    continue
                if colour not in cache:
                    cache[colour] = _INDEX_CHARS[nearest(palette, colour)]
                row.append(cache[colour])
            rows.append("".join(row))
        encoded.append(rows)
    return encoded


def _waterline_block(sprite: Sprite) -> str:
    """The sea-surface anchor, for the sprite that has one."""
    if sprite.waterline_source_y is None or sprite.crop is None:
        return ""
    top, bottom = sprite.crop[1], sprite.crop[3]
    row = round((sprite.waterline_source_y - top) / (bottom - top) * sprite.height)
    return (
        "\n#: Subpixel row of the sea surface. The waves meet the drawn splash here."
        f"\nWATERLINE = {row}\n"
    )


def render_module(
    encoded: list[list[str]],
    palette: list[tuple[int, int, int, int]],
    durations: list[int],
    source: Path,
    sprite: Sprite,
    preset: str,
) -> str:
    # Split across lines so the generated file stays inside the line limit.
    used = _INDEX_CHARS[: len(palette)]
    index_chars = "\n".join(
        f'    "{used[start : start + 48]}"' for start in range(0, len(used), 48)
    )
    colours = "\n".join(f'    ("#{r:02x}{g:02x}{b:02x}", {a}),' for r, g, b, a in palette)
    frames = "\n".join('    """\\\n' + "\n".join(rows) + '\n""",' for rows in encoded)
    exports = ["DURATIONS", "FRAMES", "HEIGHT", "INDEX_CHARS", "PALETTE", "TRANSPARENT"]
    if _waterline_block(sprite):
        exports.append("WATERLINE")
    exports.append("WIDTH")
    all_block = "\n".join(f'    "{name}",' for name in exports)
    return f'''"""Baked frames of {sprite.headline}. Generated — do not edit by hand.

{sprite.drawn_by}

Regenerate with::

    uv run --python 3.12 --with pillow python scripts/bake_shark_frames.py \\
        {preset} {source.name}

Each frame is {sprite.height} rows of {sprite.width} characters, one per quadrant
subpixel: a character cell holds two across and two down. A character indexes
:data:`PALETTE`; ``{TRANSPARENT_CHAR}`` is transparent. See
``scripts/bake_shark_frames.py`` for why the art is toned the way it is and
why alpha survives into this file.
"""

from __future__ import annotations

WIDTH = {sprite.width}
HEIGHT = {sprite.height}
{_waterline_block(sprite)}
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
{all_block}
]
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(SPRITES), help="which animation")
    parser.add_argument("source", type=Path, help="the GIF to bake")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sprite = SPRITES[args.preset]
    output = args.output or WIDGETS / f"{sprite.module}.py"
    frames, durations = load_frames(args.source, sprite)
    palette = build_palette(frames)
    encoded = encode(frames, palette, sprite)
    output.write_text(
        render_module(encoded, palette, durations, args.source, sprite, args.preset),
        encoding="utf-8",
    )
    print(
        f"{output.relative_to(ROOT)}: {len(encoded)} frames, "
        f"{sprite.width}x{sprite.height}, {len(palette)} colours"
    )


if __name__ == "__main__":
    main()
