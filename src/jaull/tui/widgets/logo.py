from __future__ import annotations

import math

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

_BASE_LOGO = r"""
     ██╗ █████╗ ██╗   ██╗██╗     ██╗
     ██║██╔══██╗██║   ██║██║     ██║
     ██║███████║██║   ██║██║     ██║
██   ██║██╔══██║██║   ██║██║     ██║
╚█████╔╝██║  ██║╚██████╔╝███████╗███████╗
 ╚════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝
""".strip("\n")

_SUBTITLE = " Made by Ton — TFG "
_SUBTITLE += "  · MIT License "



class Logo(Vertical):
    """Animated ASCII logo used on the Home screen only."""

    DEFAULT_CLASSES = "logo"

    def __init__(self) -> None:
        super().__init__()
        self._frame = 0
        self._logo_widget: Static | None = None
        self._subtitle_widget: Static | None = None

    def compose(self) -> ComposeResult:
        self._logo_widget = Static("", classes="logo-ascii", markup=True)
        self._subtitle_widget = Static("", classes="logo-subtitle", markup=True)
        yield self._logo_widget
        yield self._subtitle_widget

    def on_mount(self) -> None:
        self._refresh_logo()
        self.set_interval(0.28, self._refresh_logo)

    def _refresh_logo(self) -> None:
        if self._logo_widget is None or self._subtitle_widget is None:
            return

        self._logo_widget.update(self._build_logo_markup())
        self._subtitle_widget.update(self._build_subtitle_markup())
        self._frame += 1

    def _build_logo_markup(self) -> str:
        palette = ["#102731", "#17404a", "#235b66", "#2e6a76", "#3c7d8b"]
        lines: list[str] = []
        for row, line in enumerate(_BASE_LOGO.splitlines()):
            parts: list[str] = []
            for col, ch in enumerate(line):
                if ch == " ":
                    parts.append(" ")
                    continue

                shimmer = math.sin((self._frame * 0.3) + (row * 0.45) + (col * 0.25))
                base_color = palette[(row + col + self._frame // 2) % len(palette)]
                accent = 0.12 + (shimmer + 1) * 0.08
                color = self._blend_hex(base_color, "#7fb8c3", accent)
                parts.append(f"[{color}]{ch}[/]")
            lines.append("".join(parts))
        return "\n".join(lines)

    def _build_subtitle_markup(self) -> str:
        shimmer = math.sin(self._frame * 0.25)
        accent = 0.15 + (shimmer + 1) * 0.08
        color = self._blend_hex("#5b7d86", "#8fb0ba", accent)
        return f"[{color}]{_SUBTITLE}[/]"

    @staticmethod
    def _blend_hex(color_a: str, color_b: str, ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))

        def to_rgb(value: str) -> tuple[int, int, int]:
            value = value.lstrip("#")
            r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
            return r, g, b

        r1, g1, b1 = to_rgb(color_a)
        r2, g2, b2 = to_rgb(color_b)

        r = round(r1 + (r2 - r1) * ratio)
        g = round(g1 + (g2 - g1) * ratio)
        b = round(b1 + (b2 - b1) * ratio)

        return f"#{r:02x}{g:02x}{b:02x}"
