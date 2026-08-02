from __future__ import annotations

from pathlib import Path

from textual.app import App

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.tui.screens.advanced_tools import AdvancedToolsScreen
from local_ai_check.tui.screens.doctor import DoctorScreen
from local_ai_check.tui.screens.estimate import EstimateScreen
from local_ai_check.tui.screens.hardware_analysis import HardwareAnalysisScreen
from local_ai_check.tui.screens.inspect import InspectScreen
from local_ai_check.tui.screens.model_discovery import ModelDiscoveryScreen
from local_ai_check.tui.screens.recommendation_results import (
    RecommendationResultsScreen,
)
from local_ai_check.tui.screens.requirements_wizard import RequirementsWizardScreen
from local_ai_check.tui.screens.scan import ScanScreen
from local_ai_check.tui.screens.welcome import WelcomeScreen
from local_ai_check.workflow.container import ServiceContainer
from local_ai_check.workflow.models import UserAnswers
from local_ai_check.workflow.state import RecommendationWorkflowState


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
        "welcome": WelcomeScreen,
        "advanced": AdvancedToolsScreen,
        "scan": ScanScreen,
        "inspect": InspectScreen,
        "estimate": EstimateScreen,
        "doctor": DoctorScreen,
    }

    # In-memory cache so screens can show a hardware summary without re-probing.
    hardware_profile: HardwareProfile | None = None
    # Result of the most recent guided run, kept for the results screen.
    workflow_state: RecommendationWorkflowState | None = None

    def __init__(self, services: ServiceContainer | None = None) -> None:
        super().__init__()
        # Services are injected rather than constructed inside screens, so tests
        # can drive the whole guided flow without touching the network. Resolved
        # lazily: building the real container imports huggingface_hub and opens
        # a network-capable client, which a test supplying fakes must not pay for.
        self._services = services

    @property
    def services(self) -> ServiceContainer:
        if self._services is None:
            self._services = ServiceContainer.default()
        return self._services

    def on_mount(self) -> None:
        self.push_screen("welcome")

    # ------------------------------------------------------------------
    # Guided workflow navigation
    # ------------------------------------------------------------------
    def start_guided_workflow(self) -> None:
        self.push_screen(HardwareAnalysisScreen())

    def goto_requirements(self) -> None:
        self.push_screen(RequirementsWizardScreen())

    def start_discovery(self, answers: UserAnswers) -> None:
        self.push_screen(ModelDiscoveryScreen(answers))

    def show_recommendations(self, state: RecommendationWorkflowState) -> None:
        self.push_screen(RecommendationResultsScreen(state))

    def restart_workflow(self) -> None:
        """Drop every guided screen and return to a fresh welcome menu.

        The bottom of the stack is Textual's own default screen, so unwinding to
        it and pushing `welcome` again is what actually leaves the user on the
        welcome menu — stopping at a stack depth of one would leave the default
        screen on top instead.
        """
        self.workflow_state = None
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen("welcome")

    def goto_advanced_tools(self) -> None:
        self.restart_workflow()
        self.push_screen("advanced")

    # ------------------------------------------------------------------
    # Global bindings — unchanged behaviour for the pre-existing tools.
    # ------------------------------------------------------------------
    async def action_back(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def action_goto_home(self) -> None:
        self.restart_workflow()

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
