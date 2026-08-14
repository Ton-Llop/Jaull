from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from jaull.advisor.service import AdvisorService
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import UserAnswers
from jaull.tui.screens.advanced_tools import AdvancedToolsScreen
from jaull.tui.screens.doctor import DoctorScreen
from jaull.tui.screens.estimate import EstimateScreen
from jaull.tui.screens.hardware_analysis import HardwareAnalysisScreen
from jaull.tui.screens.inspect import InspectScreen
from jaull.tui.screens.model_discovery import ModelDiscoveryScreen
from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen
from jaull.tui.screens.recommendation_execution import (
    RecommendationExecutionScreen,
)
from jaull.tui.screens.recommendation_results import (
    RecommendationResultsScreen,
)
from jaull.tui.screens.recommendation_validation import (
    RecommendationValidationScreen,
)
from jaull.tui.screens.requirements_wizard import RequirementsWizardScreen
from jaull.tui.screens.scan import ScanScreen
from jaull.tui.screens.welcome import WelcomeScreen
from jaull.workflow.container import ServiceContainer
from jaull.workflow.state import RecommendationWorkflowState


class JaullApp(App[None]):
    """Interactive terminal front-end for jaull."""

    TITLE = "jaull"
    SUB_TITLE = "Interactive terminal UI"
    CSS_PATH = str(Path(__file__).with_name("styles.tcss"))

    # The tool shortcuts stay bound everywhere — hiding a binding does not
    # unbind it — but only Back and Quit earn footer space. During the guided
    # flow a footer advertising Scan/Inspect/Estimate/Doctor/Home is noise
    # about a mode the user is not in; AdvancedToolsScreen shows them again.
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("s", "goto_scan", "Scan", show=False),
        Binding("i", "goto_inspect", "Inspect", show=False),
        Binding("e", "goto_estimate", "Estimate", show=False),
        Binding("d", "goto_doctor", "Doctor", show=False),
        Binding("h", "goto_home", "Home", show=False),
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

    def __init__(
        self,
        services: ServiceContainer | None = None,
        advisor: AdvisorService | None = None,
    ) -> None:
        super().__init__()
        # Services are injected rather than constructed inside screens, so tests
        # can drive the whole guided flow without touching the network. Resolved
        # lazily: building the real container imports huggingface_hub and opens
        # a network-capable client, which a test supplying fakes must not pay for.
        # Either ``services`` (backwards-compatible) or ``advisor`` may be passed;
        # passing ``advisor`` also fixes ``services`` to the container it wraps.
        if advisor is not None:
            self._advisor: AdvisorService | None = advisor
            self._services: ServiceContainer | None = advisor.services
        else:
            self._advisor = None
            self._services = services

    @property
    def services(self) -> ServiceContainer:
        if self._services is None:
            self._services = self.advisor.services
        return self._services

    @property
    def advisor(self) -> AdvisorService:
        if self._advisor is None:
            if self._services is not None:
                # Explicitly injected services, mainly tests/custom composition.
                # Ho he ficat pq no em detecta al portatil ara, ja que no te
                # la mateixa source que el CLI doctor.
                self._advisor = AdvisorService(services=self._services)
            else:
                # Real application: use the canonical production wiring.
                self._advisor = AdvisorService.default()
                self._services = self._advisor.services

        return self._advisor

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

    def run_recommendation(self, recommendation: object) -> None:
        from jaull.recommendation.models import ModelRecommendation

        assert isinstance(recommendation, ModelRecommendation)
        self.push_screen(RecommendationExecutionScreen(recommendation))

    def validate_recommendation(self, recommendation: object) -> None:
        from jaull.recommendation.models import ModelRecommendation

        assert isinstance(recommendation, ModelRecommendation)
        self.push_screen(RecommendationValidationScreen(recommendation))

    def benchmark_recommendation(self, recommendation: object) -> None:
        from jaull.recommendation.models import ModelRecommendation

        assert isinstance(recommendation, ModelRecommendation)
        self.push_screen(RecommendationBenchmarkScreen(recommendation))

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
    JaullApp().run()
