"""Keyboard focus falls back to tab order without taking over native controls."""

import asyncio

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, ListView, Select

from jaull.tui.app import JaullApp
from jaull.tui.screens.advanced_tools import AdvancedToolsScreen


class _Form(Screen[None]):
    pressed = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(can_focus=False):
            yield Button("Unavailable", disabled=True)
            yield Button("First", id="first")
            yield Input("abc", id="name")
            yield Select([("One", 1), ("Two", 2)], id="choice", allow_blank=False)
            yield Button("Submit", id="submit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.pressed = event.button.id == "submit"


def test_arrows_move_focus_without_stealing_widget_keys() -> None:
    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test(size=(100, 30)) as pilot:
            form = _Form()
            await app.push_screen(form)
            await pilot.pause()
            first = form.query_one("#first", Button)
            name = form.query_one("#name", Input)
            select = form.query_one("#choice", Select)
            submit = form.query_one("#submit", Button)
            first.focus()
            await pilot.press("down")
            assert app.focused is name
            name.cursor_position = len(name.value)
            await pilot.press("left")
            assert app.focused is name
            assert name.cursor_position == 2
            await pilot.press("tab", "enter", "down", "enter")
            assert select.value == 2
            assert app.focused is select
            await pilot.press("right")
            assert app.focused is submit
            await pilot.press("up")
            assert app.focused is select
            await pilot.press("right", "enter")
            assert form.pressed
            await pilot.press("right")
            assert app.focused is first  # The disabled control is skipped.

    asyncio.run(scenario())


def test_welcome_opens_advanced_tools_with_arrows_and_enter() -> None:
    async def scenario() -> None:
        app = JaullApp()
        async with app.run_test(size=(100, 30)) as pilot:
            menu = app.screen.query_one("#welcome-menu", ListView)
            await pilot.press("down", "down")
            assert app.focused is menu
            assert menu.index == 2
            await pilot.press("enter")
            assert isinstance(app.screen, AdvancedToolsScreen)

    asyncio.run(scenario())
