from __future__ import annotations

import asyncio

from local_ai_check.domain.estimation import CompatibilityStatus
from local_ai_check.tui.app import LocalAiCheckApp
from local_ai_check.tui.screens.home import HomeScreen
from local_ai_check.tui.screens.scan import ScanScreen
from local_ai_check.tui.widgets.assessment_badge import AssessmentBadge
from local_ai_check.tui.widgets.cli_equivalent import CliEquivalent
from local_ai_check.tui.widgets.summary_card import SummaryCard


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_app_starts_at_home_screen() -> None:
    async def scenario() -> None:
        app = LocalAiCheckApp()
        async with app.run_test() as pilot:
            assert isinstance(pilot.app.screen, HomeScreen)

    _run(scenario())


def test_navigation_to_scan_screen() -> None:
    async def scenario() -> None:
        app = LocalAiCheckApp()
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(pilot.app.screen, ScanScreen)

    _run(scenario())


def test_quit_binding_exits_cleanly() -> None:
    async def scenario() -> None:
        app = LocalAiCheckApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
        # Reaching here without exception means the exit propagated cleanly.

    _run(scenario())


def test_summary_card_stores_rows() -> None:
    card = SummaryCard("Test", [("Key", "Value"), ("Other", "Data")])
    assert card._title == "Test"
    assert len(card._rows) == 2


def test_assessment_badge_covers_all_statuses() -> None:
    for status in CompatibilityStatus:
        badge = AssessmentBadge(status)
        # Every status must yield a badge widget with a non-empty CSS class.
        assert badge.has_class("badge")


def test_cli_equivalent_widget_stores_command() -> None:
    widget = CliEquivalent("local-ai-check scan")
    assert widget._command == "local-ai-check scan"


def test_home_shows_credits_widget() -> None:
    from local_ai_check.tui.widgets.credits import CreditsPanel

    async def scenario() -> None:
        app = LocalAiCheckApp()
        async with app.run_test() as pilot:
            # The Home screen must include the CreditsPanel at the bottom of the scroll.
            credits = pilot.app.screen.query(CreditsPanel)
            assert len(credits) == 1

    _run(scenario())
