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
from textual.widgets import Button, TextArea

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from capture_screenshots import build_app  # noqa: E402

from jaull.tui.screens.recommendation_execution import (  # noqa: E402
    RecommendationExecutionScreen,
)
from jaull.tui.screens.recommendation_results import (  # noqa: E402
    RecommendationResultsScreen,
)
from jaull.tui.screens.requirements_wizard import RequirementsWizardScreen  # noqa: E402

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
    timeout: float = 10.0,
) -> None:
    waited = 0.0
    while waited < timeout:
        await pilot.pause()
        if predicate():
            await pilot.pause()
            return
        await asyncio.sleep(0.05)
        waited += 0.05
    raise AssertionError("the UI never reached the expected state")


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
                pilot, lambda: isinstance(app.screen, RequirementsWizardScreen)
            )
            wizard = app.screen
            assert isinstance(wizard, RequirementsWizardScreen)
            _assert_buttons_fit(wizard, f"wizard {label}")
            # Six questions legitimately scroll; submitting must still be
            # reachable by scrolling to it.
            submit = wizard.query_one("#wizard-submit", Button)
            wizard.query_one("#wizard-body").scroll_to_widget(submit, animate=False)
            await pilot.pause()
            _assert_visible(wizard, "#wizard-submit", f"wizard {label}")

            answers = wizard.collect_answers()
            assert answers is not None
            app.start_discovery(answers)
            await _wait_for(
                pilot, lambda: isinstance(app.screen, RecommendationResultsScreen)
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
                pilot, lambda: isinstance(app.screen, RecommendationResultsScreen)
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
