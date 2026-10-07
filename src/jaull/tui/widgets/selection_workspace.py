"""Stable master/detail panes, with one pane at a time in small terminals."""

from collections.abc import Callable

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.widgets import Button

from jaull.tui.widgets.action_button import ActionButton


class SelectionWorkspace(Vertical):
    DEFAULT_CLASSES = "selection-workspace"

    def __init__(
        self,
        master: Callable[[], ComposeResult],
        detail: Callable[[], ComposeResult],
        *,
        list_label: str,
        master_id: str,
        detail_id: str,
    ) -> None:
        super().__init__()
        self._master = master
        self._detail = detail
        self._list_label = list_label
        self._master_id = master_id
        self._detail_id = detail_id
        self._show_list = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="workspace-switch"):
            yield ActionButton(self._list_label, id="workspace-list")
            yield ActionButton("Selected", id="workspace-detail")
        with Horizontal(classes="workspace-panes"):
            with VerticalScroll(id=self._master_id, classes="workspace-master"):
                yield from self._master()
            with Vertical(id=self._detail_id, classes="workspace-detail"):
                yield from self._detail()

    def on_mount(self) -> None:
        self._apply()

    def on_resize(self, event: Resize) -> None:
        self._apply()

    @on(Button.Pressed, "#workspace-list")
    def _list_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._show_list = True
        self._apply()
        master = self.query_one(".workspace-master")
        selected = list(master.query(".selectable.-selected"))
        choices = selected or list(master.query(".selectable"))
        (choices[0] if choices else master).focus()

    @on(Button.Pressed, "#workspace-detail")
    def _detail_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.show_detail()

    def show_detail(self) -> None:
        self._show_list = False
        self._apply()
        if self.compact:
            self.call_after_refresh(self._focus_detail)

    def _focus_detail(self) -> None:
        for button in self.query_one(".workspace-detail").query(Button):
            if not button.disabled and all(node.display for node in button.ancestors):
                button.focus()
                return

    @property
    def compact(self) -> bool:
        """Whether one pane at a time is all that fits.

        A workspace mounted into an already-laid-out container runs its
        ``on_mount`` before it has been given a size, and a width of zero would
        read as the narrowest possible terminal. Falling back to the screen is
        not an approximation: the panes span it, so until layout says otherwise
        the screen's width is the honest answer.
        """
        return (self.size.width or self.screen.size.width) < 120

    def _apply(self) -> None:
        self.set_class(self.compact, "-compact")
        self.query_one(".workspace-switch").display = self.compact
        self.query_one(".workspace-master").display = not self.compact or self._show_list
        self.query_one(".workspace-detail").display = not self.compact or not self._show_list
        self.query_one("#workspace-list").set_class(self._show_list, "-active")
        self.query_one("#workspace-detail").set_class(not self._show_list, "-active")
