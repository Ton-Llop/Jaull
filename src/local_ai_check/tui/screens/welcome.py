from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, ListItem, ListView, Static

from local_ai_check.tui.widgets.credits import CreditsPanel
from local_ai_check.tui.widgets.logo import Logo

if TYPE_CHECKING:
    from local_ai_check.tui.app import LocalAiCheckApp

_MENU_ITEMS = [
    ("guided", "Start guided analysis"),
    ("advanced", "Advanced tools"),
    ("quit", "Quit"),
]


class WelcomeScreen(Screen[None]):
    """Entry point: two ways in, guided first."""

    BINDINGS = [
        ("enter", "select", "Open"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll():
            yield Logo()
            with Vertical(classes="card"):
                yield Static(
                    "Find suitable local AI models for your hardware and use case.",
                    classes="banner-subtitle",
                )
                yield Static("Choose an action", classes="card-title")
                yield ListView(
                    *[
                        ListItem(Static(label), id=f"welcome-{key}", classes="menu-item")
                        for key, label in _MENU_ITEMS
                    ],
                    id="welcome-menu",
                )
            yield Static(
                "Guided analysis scans your machine, asks a few plain questions and "
                "recommends models it can justify. Advanced tools keep the original "
                "scan, inspect, estimate and doctor screens.",
                classes="text-muted",
            )
            yield CreditsPanel()
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._activate((event.item.id or "").removeprefix("welcome-"))

    def action_select(self) -> None:
        menu = self.query_one("#welcome-menu", ListView)
        highlighted = menu.highlighted_child
        if highlighted is None or highlighted.id is None:
            return
        self._activate(highlighted.id.removeprefix("welcome-"))

    def _activate(self, key: str) -> None:
        app = self._app()
        if key == "quit":
            app.exit()
        elif key == "guided":
            app.start_guided_workflow()
        elif key == "advanced":
            app.push_screen("advanced")

    def _app(self) -> LocalAiCheckApp:
        from local_ai_check.tui.app import LocalAiCheckApp

        assert isinstance(self.app, LocalAiCheckApp)
        return self.app


__all__ = ["WelcomeScreen"]
