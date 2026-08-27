from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, ListItem, ListView, Static

from jaull.tui.widgets.logo import Logo
from jaull.tui.widgets.ocean import OceanBand

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp

# Two ways in, and nothing else: quitting already has a key and a footer entry,
# so listing it as a third "choice" only dilutes the actual decision.
_MENU_ITEMS = [
    (
        "guided",
        "Guided analysis",
        "Scan this machine, answer a few questions, get ranked models",
    ),
    (
        "advanced",
        "Advanced tools",
        "Scan, inspect, estimate, doctor",
    ),
]


class WelcomeScreen(Screen[None]):
    """Entry point: two ways in, guided first."""

    BINDINGS = [
        # ListView already activates on Enter; the binding is a fallback for
        # tests and for focus landing elsewhere, so it stays out of the footer.
        Binding("enter", "select", "Open", show=False),
        # Nothing to go back to from the entry screen.
        Binding("escape", "back", "Back", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll():
            yield Logo()
            with Vertical(id="welcome-body"):
                yield Static(
                    "Find local AI models this machine can actually run.",
                    id="welcome-tagline",
                )
                yield ListView(
                    *[
                        ListItem(
                            Static(label, classes="menu-label"),
                            Static(hint, classes="menu-hint"),
                            id=f"welcome-{key}",
                            classes="menu-item",
                        )
                        for key, label, hint in _MENU_ITEMS
                    ],
                    id="welcome-menu",
                )
            # Last child, and the only one that flexes: it takes whatever the
            # menu leaves rather than pushing anything below the fold.
            yield OceanBand()
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
        if key == "guided":
            app.start_guided_workflow()
        elif key == "advanced":
            app.push_screen("advanced")

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


__all__ = ["WelcomeScreen"]
