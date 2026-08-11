"""TUI-level tests for the guided flow.

Every service is injected, so no test here reaches the network or constructs an
``HfClient``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from threading import Event, Lock

from textual.widgets import Button, Checkbox, DataTable, RadioSet

from jaull.domain.requirements import UseCase
from jaull.tui.app import JaullApp
from jaull.tui.screens.advanced_tools import AdvancedToolsScreen
from jaull.tui.screens.hardware_analysis import HardwareAnalysisScreen
from jaull.tui.screens.model_discovery import ModelDiscoveryScreen
from jaull.tui.screens.recommendation_results import (
    RecommendationCompareScreen,
    RecommendationResultsScreen,
    export_report,
)
from jaull.tui.screens.requirements_wizard import RequirementsWizardScreen
from jaull.tui.screens.welcome import WelcomeScreen
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.workflow.container import ServiceContainer
from jaull.workflow.progress import HARDWARE_STEPS
from tests._workflow_fixtures import (
    GIB,
    FakeHfClient,
    FakeSearchClient,
    candidate,
    hardware,
    size_driven_estimator,
    transformers_analysis,
)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def _services(
    repo_ids: tuple[str, ...] = ("org/Coder-7B", "org/Coder-13B"),
    detect_delay: float = 0.0,
) -> ServiceContainer:
    search = FakeSearchClient(
        default=[
            candidate(repo_id=repo, tags=["text-generation", "code"])
            for repo in repo_ids
        ]
    )
    estimator = size_driven_estimator(vram_budget=24 * GIB)

    def detect(**kwargs: object) -> object:
        if detect_delay:
            time.sleep(detect_delay)
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for key, _ in HARDWARE_STEPS:
                on_step(key)
        return hardware()

    return ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=lambda repo_id, client=None: transformers_analysis(  # type: ignore[arg-type]
            repo_id=repo_id
        ),
        estimate_memory=lambda **kwargs: estimator(  # type: ignore[arg-type]
            kwargs["analysis"], kwargs["inference_cfg"]
        ),
    )


def _services_with_blocking_detector(
    *,
    started: Event,
    release: Event,
) -> ServiceContainer:
    base = _services()

    def detect(**kwargs: object) -> object:
        started.set()
        release.wait(timeout=5)
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for key, _ in HARDWARE_STEPS:
                on_step(key)
        return hardware()

    return ServiceContainer(
        hf_client=base.hf_client,
        search_client=base.search_client,
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=base.inspect_model,
        estimate_memory=base.estimate_memory,
        capability_analyzer=base.capability_analyzer,
        range_client_factory=base.range_client_factory,
    )


async def _settle(pilot, attempts: int = 60) -> None:  # type: ignore[no-untyped-def]
    """Let worker threads hand their results back to the event loop."""
    for _ in range(attempts):
        await pilot.pause()
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Welcome
# ---------------------------------------------------------------------------
def test_welcome_screen_offers_guided_and_advanced() -> None:
    """Two product decisions, and only two: `q` already quits."""

    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            assert isinstance(screen, WelcomeScreen)
            ids = {item.id for item in screen.query("ListItem")}
            assert ids == {"welcome-guided", "welcome-advanced"}

    _run(scenario())


def test_entering_guided_mode_opens_the_hardware_screen() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.start_guided_workflow()
            await pilot.pause()
            assert isinstance(pilot.app.screen, HardwareAnalysisScreen)

    _run(scenario())


def test_entering_advanced_tools_keeps_the_original_menu() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.push_screen("advanced")
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, AdvancedToolsScreen)
            ids = {item.id for item in screen.query("ListItem")}
            assert {"menu-scan", "menu-inspect", "menu-estimate", "menu-doctor"} <= ids

    _run(scenario())


# ---------------------------------------------------------------------------
# Hardware progress
# ---------------------------------------------------------------------------
def test_hardware_screen_shows_progress_and_a_continue_button() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.start_guided_workflow()
            await _settle(pilot)
            assert app.hardware_profile is not None
            # After completion the loading card is replaced by the summary
            # card in the same slot — no scrolling needed to see the result.
            assert pilot.app.screen.query(SummaryCard)
            assert pilot.app.screen.query_one("#hw-continue", Button)

    _run(scenario())


def test_hardware_screen_can_be_closed_before_detector_finishes() -> None:
    async def scenario() -> None:
        started = Event()
        release = Event()
        app = JaullApp(
            services=_services_with_blocking_detector(
                started=started,
                release=release,
            )
        )

        async with app.run_test(size=(120, 50)) as pilot:
            app.start_guided_workflow()
            await pilot.pause()
            assert isinstance(pilot.app.screen, HardwareAnalysisScreen)
            assert started.wait(timeout=1)

            app.pop_screen()
            await pilot.pause()
            assert isinstance(pilot.app.screen, WelcomeScreen)

            release.set()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(pilot.app.screen, WelcomeScreen)

    _run(scenario())


def test_hardware_screen_repeated_start_and_back_does_not_hang() -> None:
    async def scenario() -> None:
        starts = [Event() for _ in range(3)]
        releases = [Event() for _ in range(3)]
        lock = Lock()
        calls = 0
        base = _services()

        def detect(**kwargs: object) -> object:
            nonlocal calls
            with lock:
                index = calls
                calls += 1
            starts[index].set()
            releases[index].wait(timeout=5)
            on_step = kwargs.get("on_step")
            if callable(on_step):
                for key, _ in HARDWARE_STEPS:
                    on_step(key)
            return hardware()

        app = JaullApp(
            services=ServiceContainer(
                hf_client=base.hf_client,
                search_client=base.search_client,
                detect_hardware=detect,  # type: ignore[arg-type]
                inspect_model=base.inspect_model,
                estimate_memory=base.estimate_memory,
                capability_analyzer=base.capability_analyzer,
                range_client_factory=base.range_client_factory,
            )
        )

        async with app.run_test(size=(120, 50)) as pilot:
            for index in range(3):
                app.start_guided_workflow()
                await pilot.pause()
                assert isinstance(pilot.app.screen, HardwareAnalysisScreen)
                assert starts[index].wait(timeout=1)
                app.pop_screen()
                await pilot.pause()
                assert isinstance(pilot.app.screen, WelcomeScreen)
                releases[index].set()
                await pilot.pause()

    _run(scenario())


def test_network_and_probes_do_not_block_the_event_loop() -> None:
    """The UI must still answer keystrokes while a slow probe is running."""

    async def scenario() -> None:
        app = JaullApp(services=_services(detect_delay=0.6))
        async with app.run_test() as pilot:
            app.start_guided_workflow()
            await pilot.pause()

            started = time.perf_counter()
            # If the probe ran on the event loop this would block for ~0.6 s.
            await pilot.pause()
            await pilot.pause()
            elapsed = time.perf_counter() - started
            assert elapsed < 0.4, elapsed
            assert isinstance(pilot.app.screen, HardwareAnalysisScreen)

            await _settle(pilot)
            assert app.hardware_profile is not None

    _run(scenario())


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------
def test_wizard_collects_answers_from_its_widgets() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.push_screen(RequirementsWizardScreen())
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RequirementsWizardScreen)

            answers = screen.collect_answers()
            assert answers is not None
            assert answers.use_case is UseCase.GENERAL_CHAT
            assert answers.languages == ["English"]

    _run(scenario())


def test_document_question_is_hidden_unless_the_use_case_needs_it() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.push_screen(RequirementsWizardScreen())
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RequirementsWizardScreen)

            card = screen.query_one("#q-documents-card")
            assert card.display is False

            radio_set = screen.query_one("#q-use-case", RadioSet)
            for button in radio_set.query("RadioButton"):
                if button.id == "uc-document_qa":
                    button.value = True
                    break
            await pilot.pause()
            assert card.display is True

            answers = screen.collect_answers()
            assert answers is not None
            assert answers.document_scale is not None

    _run(scenario())


def test_wizard_rejects_an_empty_language_selection() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.push_screen(RequirementsWizardScreen())
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RequirementsWizardScreen)

            screen.query_one("#lang-english", Checkbox).value = False
            await pilot.pause()
            assert screen.collect_answers() is None

    _run(scenario())


def test_wizard_rejects_unrecognised_other_languages() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.push_screen(RequirementsWizardScreen())
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RequirementsWizardScreen)

            screen.query_one("#lang-other", Checkbox).value = True
            screen.query_one("#lang-other-input").value = "klingon"  # type: ignore[attr-defined]
            await pilot.pause()
            assert screen.collect_answers() is None

    _run(scenario())


# ---------------------------------------------------------------------------
# Discovery and results
# ---------------------------------------------------------------------------
def test_wizard_navigates_into_discovery() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.push_screen(RequirementsWizardScreen())
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RequirementsWizardScreen)
            answers = screen.collect_answers()
            assert answers is not None

            app.start_discovery(answers)
            await pilot.pause()
            # With fake services the run can finish within a single tick, so
            # either the progress screen or its successor is a valid landing.
            assert isinstance(
                pilot.app.screen,
                ModelDiscoveryScreen | RecommendationResultsScreen,
            )

    _run(scenario())


def test_full_guided_run_reaches_the_results_screen() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test(size=(120, 50)) as pilot:
            app.start_guided_workflow()
            await _settle(pilot)

            app.goto_requirements()
            await pilot.pause()
            wizard = pilot.app.screen
            assert isinstance(wizard, RequirementsWizardScreen)
            answers = wizard.collect_answers()
            assert answers is not None

            app.start_discovery(answers)
            await _settle(pilot)

            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)
            assert screen.query_one("#res-run-0", Button)
            assert app.workflow_state is not None
            assert app.workflow_state.recommendations

    _run(scenario())


def test_results_screen_can_render_a_comparison(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test(size=(120, 50)) as pilot:
            app.start_guided_workflow()
            await _settle(pilot)
            app.goto_requirements()
            await pilot.pause()
            wizard = pilot.app.screen
            assert isinstance(wizard, RequirementsWizardScreen)
            answers = wizard.collect_answers()
            assert answers is not None
            app.start_discovery(answers)
            await _settle(pilot)

            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)
            screen.query_one("#res-compare", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, RecommendationCompareScreen)
            assert pilot.app.screen.query(DataTable)

            state = app.workflow_state
            assert state is not None
            written = export_report(state, tmp_path / "report.json")
            assert len(written) == 2
            assert written[0].exists() and written[1].exists()
            assert "schema_version" in written[0].read_text(encoding="utf-8")

    _run(scenario())


def test_restart_returns_to_the_welcome_screen() -> None:
    async def scenario() -> None:
        app = JaullApp(services=_services())
        async with app.run_test() as pilot:
            app.start_guided_workflow()
            await _settle(pilot)
            app.restart_workflow()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(pilot.app.screen, WelcomeScreen)
            assert app.workflow_state is None

    _run(scenario())


def test_services_are_injected_and_never_built_by_screens() -> None:
    """Constructing the app with fakes must not create a real HfClient."""
    services = _services()
    app = JaullApp(services=services)
    assert app.services is services
    assert isinstance(app.services.hf_client, FakeHfClient)
