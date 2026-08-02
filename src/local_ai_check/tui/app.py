from __future__ import annotations

from pathlib import Path

from textual.app import App

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.tui.screens.doctor import DoctorScreen
from local_ai_check.tui.screens.estimate import EstimateScreen
from local_ai_check.tui.screens.home import HomeScreen
from local_ai_check.tui.screens.inspect import InspectScreen
from local_ai_check.tui.screens.scan import ScanScreen


class LocalAiCheckApp(App[None]):
    """Interactive terminal front-end for local-ai-check."""

    TITLE = "local-ai-check"
    SUB_TITLE = "Interactive terminal UI"
    CSS_PATH = str(Path(__file__).with_name("styles.tcss"))

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "back", "Back"),
        ("s", "goto_scan", "Scan"),
        ("i", "goto_inspect", "Inspect"),
        ("e", "goto_estimate", "Estimate"),
        ("d", "goto_doctor", "Doctor"),
        ("h", "goto_home", "Home"),
    ]

    SCREENS = {
        "home": HomeScreen,
        "scan": ScanScreen,
        "inspect": InspectScreen,
        "estimate": EstimateScreen,
        "doctor": DoctorScreen,
    }

    # In-memory cache so pantalla Home puede mostrar un resumen sin re-probar el hardware.
    hardware_profile: HardwareProfile | None = None

    def on_mount(self) -> None:
        self.push_screen("home")

    async def action_back(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def action_goto_home(self) -> None:
        self._replace_top("home")

    def action_goto_scan(self) -> None:
        self._replace_top("scan")

    def action_goto_inspect(self) -> None:
        self._replace_top("inspect")

    def action_goto_estimate(self) -> None:
        self._replace_top("estimate")

    def action_goto_doctor(self) -> None:
        self._replace_top("doctor")

    def _replace_top(self, name: str) -> None:
        # Keep the stack shallow: navigating from a subscreen replaces it.
        if len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen(name)


def run() -> None:
    LocalAiCheckApp().run()
