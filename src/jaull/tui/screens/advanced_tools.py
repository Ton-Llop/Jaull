from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, ListItem, ListView, Static

from jaull.tui.widgets.banner import Banner
from jaull.tui.widgets.summary_card import SummaryCard

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp

# The original tool menu, preserved: every screen and binding that existed
# before the guided flow is still reachable from here.
_MENU_ITEMS = [
    ("scan", "Scan local hardware"),
    ("inspect", "Inspect Hugging Face model"),
    ("estimate", "Estimate model memory"),
    ("doctor", "Run diagnostics"),
    ("welcome", "Back to welcome"),
]


class AdvancedToolsScreen(Screen[None]):
    """The pre-existing tools, for users who know what they are looking for."""

    # This is the mode the tool shortcuts belong to, so here they are shown.
    # Re-declaring them on the screen overrides the hidden app-level bindings.
    BINDINGS = [
        Binding("enter", "select", "Open", show=False),
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("s", "app.goto_scan", "Scan"),
        Binding("i", "app.goto_inspect", "Inspect"),
        Binding("e", "app.goto_estimate", "Estimate"),
        Binding("d", "app.goto_doctor", "Doctor"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner(
            "Advanced tools",
            "Direct access to hardware scan, model inspection and memory estimation.",
        )
        with VerticalScroll(), Vertical():
            summary = self._hardware_summary()
            if summary:
                yield SummaryCard("System summary", summary)
            yield Static("Choose a tool", classes="card-title")
            yield ListView(
                *[
                    ListItem(Static(label), id=f"menu-{key}", classes="menu-item")
                    for key, label in _MENU_ITEMS
                ],
                id="advanced-menu",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._activate((event.item.id or "").removeprefix("menu-"))

    def action_select(self) -> None:
        menu = self.query_one("#advanced-menu", ListView)
        highlighted = menu.highlighted_child
        if highlighted is None or highlighted.id is None:
            return
        self._activate(highlighted.id.removeprefix("menu-"))

    def _activate(self, key: str) -> None:
        app = self._app()
        if key == "welcome":
            app.pop_screen()
            return
        if key in {"scan", "inspect", "estimate", "doctor"}:
            app.push_screen(key)

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app

    def _hardware_summary(self) -> list[tuple[str, str]]:
        profile = self._app().hardware_profile
        if profile is None:
            return []
        gpu = profile.gpus[0].name if profile.gpus else "no NVIDIA GPU detected"
        return [
            ("OS", profile.os),
            ("CPU", profile.cpu.model or "unknown"),
            ("RAM", _fmt(profile.memory.available_bytes) + " available"),
            ("GPU", gpu),
        ]


def _fmt(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


__all__ = ["AdvancedToolsScreen"]
