from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, ListItem, ListView, Static

from local_ai_check.tui.widgets.credits import CreditsPanel
from local_ai_check.tui.widgets.logo import Logo
from local_ai_check.tui.widgets.summary_card import SummaryCard

if TYPE_CHECKING:
    from local_ai_check.tui.app import LocalAiCheckApp


_MENU_ITEMS = [
    ("scan", "Scan local hardware"),
    ("inspect", "Inspect Hugging Face model"),
    ("estimate", "Estimate model memory"),
    ("doctor", "Run diagnostics"),
    ("quit", "Quit"),
]


class HomeScreen(Screen[None]):
    BINDINGS = [
        ("enter", "select", "Open"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll():
            yield Logo()
            with Vertical():
                summary = self._hardware_summary()
                if summary:
                    yield SummaryCard("System summary", summary)
                yield Static("Choose an action", classes="card-title")
                yield ListView(
                    *[
                        ListItem(Static(label), id=f"menu-{key}", classes="menu-item")
                        for key, label in _MENU_ITEMS
                    ],
                    id="home-menu",
                )
            yield Static("Scroll ↓ for credits", classes="text-muted")
            yield CreditsPanel()
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        key = item_id.removeprefix("menu-")
        self._activate(key)

    def action_select(self) -> None:
        menu = self.query_one("#home-menu", ListView)
        highlighted = menu.highlighted_child
        if highlighted is None or highlighted.id is None:
            return
        self._activate(highlighted.id.removeprefix("menu-"))

    def _activate(self, key: str) -> None:
        app = self._app()
        if key == "quit":
            app.exit()
            return
        if key in {"scan", "inspect", "estimate", "doctor"}:
            app.push_screen(key)

    def _app(self) -> LocalAiCheckApp:
        from local_ai_check.tui.app import LocalAiCheckApp

        assert isinstance(self.app, LocalAiCheckApp)
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
