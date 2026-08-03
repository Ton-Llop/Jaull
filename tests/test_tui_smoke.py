from __future__ import annotations

import asyncio

from jaull.domain.estimation import CompatibilityStatus
from jaull.tui.app import JaullApp
from jaull.tui.screens.advanced_tools import AdvancedToolsScreen
from jaull.tui.screens.scan import ScanScreen
from jaull.tui.screens.welcome import WelcomeScreen
from jaull.tui.widgets.assessment_badge import AssessmentBadge
from jaull.tui.widgets.cli_equivalent import CliEquivalent
from jaull.tui.widgets.summary_card import SummaryCard


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_app_starts_at_welcome_screen() -> None:
    """The guided flow is the entry point; the tool menu moved to Advanced tools."""

    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test() as pilot:
            assert isinstance(pilot.app.screen, WelcomeScreen)

    _run(scenario())


def test_navigation_to_scan_screen() -> None:
    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(pilot.app.screen, ScanScreen)

    _run(scenario())


def test_advanced_tools_reaches_every_original_screen() -> None:
    """Every pre-existing screen must remain registered after the restructure."""

    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test() as pilot:
            app.push_screen("advanced")
            await pilot.pause()
            assert isinstance(pilot.app.screen, AdvancedToolsScreen)

            for name, expected in (
                ("scan", "ScanScreen"),
                ("inspect", "InspectScreen"),
                ("estimate", "EstimateScreen"),
                ("doctor", "DoctorScreen"),
            ):
                app.push_screen(name)
                await pilot.pause()
                assert type(pilot.app.screen).__name__ == expected
                app.pop_screen()
                await pilot.pause()

    _run(scenario())


def test_advanced_menu_lists_every_original_tool() -> None:
    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test() as pilot:
            app.push_screen("advanced")
            await pilot.pause()
            ids = {item.id for item in pilot.app.screen.query("ListItem")}
            assert {
                "menu-scan",
                "menu-inspect",
                "menu-estimate",
                "menu-doctor",
            } <= ids

    _run(scenario())


def test_quit_binding_exits_cleanly() -> None:
    async def scenario() -> None:
        app = JaullApp()
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
    widget = CliEquivalent("jaull scan")
    assert widget._command == "jaull scan"


