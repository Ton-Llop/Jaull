from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ProgressBar, Static


class MemoryUsageBar(Vertical):
    """Simple labelled bar showing `used / total` for RAM or VRAM.

    ``plain`` drops the box, for the guided screens.
    """

    def __init__(
        self,
        label: str,
        used_bytes: int | None,
        total_bytes: int | None,
        *,
        plain: bool = False,
    ) -> None:
        super().__init__(classes="section" if plain else "card")
        self._label = label
        self._used = used_bytes
        self._total = total_bytes
        self._plain = plain

    def compose(self) -> ComposeResult:
        yield Static(
            self._label,
            classes="section-title" if self._plain else "card-title",
        )
        if self._total is None or self._total == 0:
            yield Static("capacity unknown", classes="text-muted")
            return
        used = self._used or 0
        percent = min(100.0, (used / self._total) * 100)
        bar = ProgressBar(total=100, show_eta=False, show_percentage=True)
        yield bar
        bar.update(progress=percent)
        yield Static(
            f"{_format(used)} / {_format(self._total)}  ({percent:.1f}%)"
        )


def _format(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
