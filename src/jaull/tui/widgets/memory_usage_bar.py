from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ProgressBar, Static


class MemoryUsageBar(Vertical):
    """Simple labelled bar showing `used / total` for RAM or VRAM."""

    DEFAULT_CLASSES = "card"

    def __init__(self, label: str, used_bytes: int | None, total_bytes: int | None) -> None:
        super().__init__()
        self._label = label
        self._used = used_bytes
        self._total = total_bytes

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="card-title")
        if self._total is None or self._total == 0:
            yield Static("[dim]capacity unknown[/dim]")
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
