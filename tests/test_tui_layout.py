"""Layout guards for the guided flow at realistic terminal sizes.

These assert reachability, not pixels: a button that renders past the right
edge cannot be clicked, and a composer below the fold cannot be typed into. The
exact position of anything is deliberately not asserted, so restyling stays
cheap.

The fakes come from ``scripts/capture_screenshots.py`` so the screens under
test are the same ones the documentation shows. Nothing here touches the
network or llama.cpp.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Static, TabbedContent, TextArea

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from capture_screenshots import build_app  # noqa: E402

from jaull.tui.screens.estimate import EstimateScreen  # noqa: E402
from jaull.tui.screens.execution_paths import ExecutionPathsScreen  # noqa: E402
from jaull.tui.screens.recommendation_execution import (  # noqa: E402
    RecommendationExecutionScreen,
)
from jaull.tui.screens.recommendation_results import (  # noqa: E402
    RecommendationResultsScreen,
)
from jaull.tui.screens.requirements_wizard import RequirementsWizardScreen  # noqa: E402
from jaull.tui.widgets.action_button import ActionButton  # noqa: E402
from jaull.tui.widgets.selection_workspace import SelectionWorkspace  # noqa: E402
from tests._workflow_fixtures import gguf_analysis  # noqa: E402

# The sizes the interface is designed against: a comfortable terminal, a
# medium one, and 80x24 — the classic default, and the width at which the old
# results row collapsed its model-name column to one character and wrapped a
# repository id one letter per line. Anything narrower is out of scope.
SIZES = [(110, 32), (90, 28), (80, 24)]


def _run(coro: Any) -> None:
    asyncio.run(coro)


async def _wait_for(
    pilot: Any,
    predicate: Callable[[], bool],
    *,
    timeout: float = 30.0,
) -> None:
    # `pilot.pause(0)`, not `pilot.pause()`: the argument-less form waits for
    # process-wide CPU idle, so one busy thread anywhere pins every poll at its
    # 1 s ceiling. See the note in `test_tui_guided_workflow._wait_for`.
    # The deadline is wall clock too, because counting `waited += 0.05` assumed
    # an iteration costs only its sleep, and it does not.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await pilot.pause(0)
        if predicate():
            await pilot.pause(0)
            return
        await asyncio.sleep(0.05)
    screen = pilot.app.screen
    recommendations = getattr(getattr(screen, "_state", None), "recommendations", None)
    recommendation_count = (
        len(recommendations) if recommendations is not None else "n/a"
    )
    button_ids = [button.id for button in screen.query(Button) if button.id]
    raise AssertionError(
        "the UI never reached the expected state "
        f"(screen={type(screen).__name__}, "
        f"recommendations={recommendation_count}, buttons={button_ids})"
    )


def _assert_buttons_fit(screen: Any, label: str) -> None:
    """No button may hang off either side of the viewport."""
    width = screen.size.width
    for button in screen.query(Button):
        if not button.display:
            continue
        region = button.region
        assert region.x >= 0, f"{label}: {button.id} starts left of the viewport"
        assert region.right <= width, (
            f"{label}: {button.id} runs past the right edge "
            f"({region.right} > {width})"
        )


def _assert_visible(screen: Any, selector: str, label: str) -> None:
    """The widget must be on screen without scrolling."""
    widget = screen.query_one(selector)
    region = widget.region
    assert region.width > 0 and region.height > 0, f"{label}: {selector} has no size"
    assert region.y >= 0 and region.bottom <= screen.size.height, (
        f"{label}: {selector} is outside the viewport "
        f"({region.y}..{region.bottom} in {screen.size.height} rows)"
    )
    assert region.right <= screen.size.width, f"{label}: {selector} is cut off"


def test_secondary_action_labels_are_literal_and_keep_brackets_when_relabelled() -> None:
    button = ActionButton("Validate [local]")
    assert button.label.plain == "[ Validate [local] ]"
    button.set_action_label("Paths [2]")
    assert button.action_label == "Paths [2]"
    assert button.label.plain == "[ Paths [2] ]"


@pytest.mark.parametrize("width", [120, 121, 150, 207, 208, 256])
def test_selection_panes_fill_the_entire_available_width(width: int) -> None:
    class WorkspaceScreen(Screen[None]):
        def compose(self) -> ComposeResult:
            yield SelectionWorkspace(
                lambda: iter([Static("Models")]), lambda: iter([Static("Selected model")]),
                list_label="Models", master_id="models", detail_id="selected-model",
            )

    async def scenario() -> None:
        app = build_app()
        async with app.run_test(size=(width, 40)) as pilot:
            await app.push_screen(WorkspaceScreen())
            await pilot.pause()
            screen = app.screen
            workspace = screen.query_one(SelectionWorkspace)
            panes = screen.query_one(".workspace-panes")
            left = screen.query_one(".workspace-master")
            right = screen.query_one(".workspace-detail")
            assert not workspace.compact
            assert panes.region.width == width
            assert left.region.x == panes.content_region.x
            assert left.region.right == right.region.x
            assert right.region.right == panes.content_region.right

    _run(scenario())


@pytest.mark.parametrize("size", [(150, 42), (120, 32), (80, 24)])
def test_results_inspector_keeps_selection_and_controls_across_resize(
    size: tuple[int, int],
) -> None:
    async def scenario() -> None:
        app = build_app()
        async with app.run_test(size=size) as pilot:
            app.start_guided_workflow()
            await _wait_for(pilot, lambda: bool(app.screen.query("#hw-continue")))
            app.goto_requirements()
            await _wait_for(pilot, lambda: isinstance(app.screen, RequirementsWizardScreen))
            answers = app.screen.collect_answers()
            app.start_discovery(answers)
            await _wait_for(
                pilot, lambda: isinstance(app.screen, RecommendationResultsScreen)
                and bool(app.screen.query("#res-run-0")),
            )
            screen = app.screen
            workspace = screen.query_one(SelectionWorkspace)
            run = screen.query_one("#res-run-1", Button)
            if workspace.compact:
                await pilot.click("#workspace-list")
                assert screen.query_one("#results-body").display
                assert not screen.query_one("#results-inspector").display
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected == 1, repr(app.focused)
            # Browsing the list does not force the inspector open on each key.
            if workspace.compact:
                assert screen.query_one("#results-body").display
            await pilot.press("enter")
            await pilot.pause()
            assert screen.query_one("#rec-detail-1").display
            assert not screen.query_one("#rec-detail-0").display
            _assert_visible(screen, "#res-run-1", "selected actions")
            detail = screen.query_one("#rec-detail-1")
            subtitle = detail.query_one(".inspector-subtitle")
            actions = detail.query_one(".rec-actions")
            tabs = detail.query_one(TabbedContent)
            assert subtitle.region.bottom <= actions.region.y
            assert actions.region.bottom <= tabs.region.y
            if not workspace.compact:
                left = screen.query_one("#results-body").region
                right = screen.query_one("#results-inspector").region
                assert left.right <= right.x
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert workspace.compact
            assert not screen.query_one("#results-body").display
            assert screen.query_one("#results-inspector").display
            _assert_visible(screen, "#res-run-1", "narrow selected actions")
            assert actions.region.bottom <= tabs.region.y
            await pilot.resize_terminal(150, 42)
            await pilot.pause()
            assert not workspace.compact
            assert screen.query_one("#results-body").display
            assert screen.query_one("#res-run-1", Button) is run
            _assert_buttons_fit(screen, "resized results")
            screen.query_one("#res-paths-1", Button).press()
            await _wait_for(
                pilot, lambda: isinstance(app.screen, ExecutionPathsScreen)
                and bool(app.screen.query(".path-option"))
                and app.screen.query_one("#paths-selected").display,
            )
            await pilot.resize_terminal(*size)
            await pilot.pause()
            paths = app.screen
            path_actions = paths.query_one("#paths-actions")
            assert paths.query_one("#paths-selected-meta").region.bottom <= path_actions.region.y
            assert path_actions.region.bottom <= paths.query_one(TabbedContent).region.y
            _assert_visible(paths, "#paths-run", "selected path actions")
            _assert_buttons_fit(paths, "selected path actions")

    _run(scenario())


@pytest.mark.parametrize("size", [(160, 50), (110, 55), (80, 24)])
def test_estimate_form_uses_available_height_without_an_empty_results_pane(
    size: tuple[int, int],
) -> None:
    async def scenario() -> None:
        app = build_app()
        async with app.run_test(size=size) as pilot:
            app.push_screen("estimate")
            await _wait_for(pilot, lambda: isinstance(app.screen, EstimateScreen))
            screen = app.screen
            assert isinstance(screen, EstimateScreen)
            screen._render_form(gguf_analysis())
            await _wait_for(
                pilot,
                lambda: bool(screen.query("#est-run"))
                and screen.query_one("#est-run").region.height > 0,
            )
            body = screen.query_one("#est-body", VerticalScroll)
            label = f"estimate {size}"
            if size[1] >= 50:
                assert body.max_scroll_y == 0, "empty results must not reserve a separate pane"
                _assert_visible(screen, "#est-device", label)
                _assert_visible(screen, "#est-run", label)
                assert screen.query_one("#est-run").region.bottom <= body.content_region.bottom
            else:
                assert body.max_scroll_y > 0
                body.scroll_to_widget(
                    screen.query_one("#est-run"), animate=False, immediate=True,
                )
                await pilot.pause()
                _assert_visible(screen, "#est-run", label)
                assert screen.query_one("#est-run").region.bottom <= body.content_region.bottom
            assert not screen.query_one("#est-output").display
            _assert_buttons_fit(screen, label)

    _run(scenario())


@pytest.mark.parametrize("size", SIZES)
def test_guided_flow_stays_usable_at(size: tuple[int, int]) -> None:
    label = f"{size[0]}x{size[1]}"

    async def scenario() -> None:
        app = build_app()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            _assert_buttons_fit(app.screen, f"welcome {label}")

            app.start_guided_workflow()
            await _wait_for(pilot, lambda: bool(app.screen.query("#hw-continue")))
            _assert_buttons_fit(app.screen, f"hardware {label}")
            # Continuing is the whole point of the screen: never below the fold.
            _assert_visible(app.screen, "#hw-continue", f"hardware {label}")

            app.goto_requirements()
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, RequirementsWizardScreen)
                and app.screen.is_mounted
                and bool(app.screen.query("#wizard-submit"))
                and app.screen.query_one("#wizard-submit").region.height > 0,
            )
            wizard = app.screen
            assert isinstance(wizard, RequirementsWizardScreen)
            _assert_buttons_fit(wizard, f"wizard {label}")
            # Six questions legitimately scroll; submitting must still be
            # reachable by scrolling to it.
            submit = wizard.query_one("#wizard-submit", Button)
            wizard.query_one("#wizard-body").scroll_to_widget(
                submit, animate=False, immediate=True
            )
            await _wait_for(
                pilot,
                lambda: submit.region.y >= 0
                and submit.region.bottom <= wizard.size.height,
            )
            _assert_visible(wizard, "#wizard-submit", f"wizard {label}")

            answers = wizard.collect_answers()
            assert answers is not None
            app.start_discovery(answers)
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, RecommendationResultsScreen)
                and bool(app.screen.query("#res-run-0")),
            )
            results = app.screen
            _assert_buttons_fit(results, f"results {label}")
            # Running the best match is the primary action of the screen.
            _assert_visible(results, "#res-run-0", f"results {label}")

            results.query_one("#res-run-0", Button).press()
            await _wait_for(
                pilot, lambda: isinstance(app.screen, RecommendationExecutionScreen)
            )
            run_screen = app.screen
            _assert_buttons_fit(run_screen, f"run {label}")
            # A composer you cannot see is a screen you cannot use.
            _assert_visible(run_screen, "#run-prompt-input", f"run {label}")
            _assert_visible(run_screen, "#run-generate", f"run {label}")
            assert not run_screen.query_one("#run-prompt-input", TextArea).disabled

    _run(scenario())


@pytest.mark.parametrize("size", SIZES)
def test_run_history_keeps_the_composer_in_place(size: tuple[int, int]) -> None:
    """A long history scrolls; it never pushes the composer off screen."""
    label = f"{size[0]}x{size[1]}"

    async def scenario() -> None:
        app = build_app()
        async with app.run_test(size=size) as pilot:
            app.start_guided_workflow()
            await _wait_for(pilot, lambda: bool(app.screen.query("#hw-continue")))
            app.goto_requirements()
            await _wait_for(
                pilot, lambda: isinstance(app.screen, RequirementsWizardScreen)
            )
            wizard = app.screen
            assert isinstance(wizard, RequirementsWizardScreen)
            answers = wizard.collect_answers()
            assert answers is not None
            app.start_discovery(answers)
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, RecommendationResultsScreen)
                and bool(app.screen.query("#res-run-0")),
            )
            app.screen.query_one("#res-run-0", Button).press()
            await _wait_for(
                pilot, lambda: isinstance(app.screen, RecommendationExecutionScreen)
            )
            screen = app.screen
            assert isinstance(screen, RecommendationExecutionScreen)

            for index in range(4):
                screen.query_one("#run-prompt-input", TextArea).load_text(
                    f"prompt {index}"
                )
                screen.query_one("#run-generate", Button).press()
                await _wait_for(
                    pilot,
                    lambda index=index: len(screen.query(".inference-response"))
                    == index + 1,
                )

            _assert_visible(screen, "#run-prompt-input", f"run history {label}")
            _assert_visible(screen, "#run-generate", f"run history {label}")
            _assert_visible(screen, "#run-status", f"run history {label}")

    _run(scenario())
