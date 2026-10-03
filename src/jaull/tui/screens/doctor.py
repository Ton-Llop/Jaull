from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, LoadingIndicator, Static

from jaull.domain.enums import DiagnosticStatus
from jaull.domain.model import DiagnosticResult
from jaull.tui import palette
from jaull.tui.widgets.banner import Banner
from jaull.tui.widgets.selection_workspace import SelectionWorkspace

_STATUS_GLYPHS: dict[DiagnosticStatus, str] = {
    DiagnosticStatus.OK: "✓",
    DiagnosticStatus.WARN: "!",
    DiagnosticStatus.FAIL: "✗",
}
_STATUS_CLASSES: dict[DiagnosticStatus, str] = {
    DiagnosticStatus.OK: "status-ok",
    DiagnosticStatus.WARN: "status-warn",
    DiagnosticStatus.FAIL: "status-fail",
}


class _DoctorFinished(Message):
    def __init__(self, results: list[DiagnosticResult]) -> None:
        super().__init__()
        self.results = results



class _CheckRow(Vertical):
    """One diagnostic in the list; its reading lives in the other pane."""

    DEFAULT_CLASSES = "selectable"
    BINDINGS = [("enter", "choose", "Select")]

    class Chosen(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, index: int, result: DiagnosticResult, *, selected: bool) -> None:
        super().__init__(id=f"doctor-check-{index}")
        self._index = index
        self._result = result
        self.detail = _CheckDetail(index, result)
        self._selected = selected
        self.can_focus = True
        self.set_class(selected, "-selected")

    def compose(self) -> ComposeResult:
        label, style = _STATUS_STYLES[self._result.status]
        # One Static, not a row of two: the status travels as glyph, word and
        # colour inside the same line, so nothing has to negotiate width with
        # the name and a long check name is never truncated against it.
        yield Static(
            f"[{style}]{_STATUS_GLYPHS[self._result.status]}[/] "
            f"{self._result.name}  [{style}]{label}[/]",
            classes="rec-name",
        )

    def on_mount(self) -> None:
        self._apply()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self._apply()

    def _apply(self) -> None:
        self.detail.display = self._selected

    def on_click(self) -> None:
        self.post_message(self.Chosen(self._index))

    def action_choose(self) -> None:
        self.post_message(self.Chosen(self._index))


class _CheckDetail(Vertical):
    """What one check found, with room for the text a table cell truncated."""

    DEFAULT_CLASSES = "doctor-detail"

    def __init__(self, index: int, result: DiagnosticResult) -> None:
        super().__init__(id=f"doctor-detail-{index}")
        self._result = result

    def compose(self) -> ComposeResult:
        yield Static(self._result.name, classes="inspector-title")
        yield Static(
            _STATUS_STYLES[self._result.status][0],
            classes=f"inspector-subtitle {_STATUS_CLASSES[self._result.status]}",
        )
        yield Static(self._result.detail or "No further detail.", classes="doctor-detail-body")


class DoctorScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._doctor_closing = Event()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None

    _rows: list[_CheckRow]
    _selected: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner("Environment diagnostics", "Runs the doctor checks and shows their status.")
        yield LoadingIndicator(id="doctor-loading")
        yield Vertical(id="doctor-content")
        yield Footer()

    def on_mount(self) -> None:
        self._rows = []
        self._doctor_closing.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-doctor",
        )
        self._future = self._executor.submit(self._worker)

    def on_unmount(self) -> None:
        self._doctor_closing.set()
        self._shutdown_doctor_executor()

    def _shutdown_doctor_executor(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _worker(self) -> None:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        results = self.app.advisor.diagnostics()
        if self._doctor_closing.is_set():
            return
        self.post_message(_DoctorFinished(results))

    @on(_DoctorFinished)
    def _populate_message(self, message: _DoctorFinished) -> None:
        if self._doctor_closing.is_set():
            return
        self._populate(message.results)

    def _populate(self, results: list[DiagnosticResult]) -> None:
        self._shutdown_doctor_executor()
        self.query_one("#doctor-loading", LoadingIndicator).display = False
        self._rows = [
            _CheckRow(index, result, selected=index == 0)
            for index, result in enumerate(results)
        ]
        self._selected = 0
        content = self.query_one("#doctor-content", Vertical)
        content.remove_children()
        if not self._rows:
            content.mount(Static("No diagnostics were produced.", classes="text-muted"))
            return
        content.mount(
            SelectionWorkspace(
                self._compose_checks,
                self._compose_details,
                list_label="Checks",
                master_id="doctor-checks",
                detail_id="doctor-detail",
            )
        )

    def _compose_checks(self) -> ComposeResult:
        yield Static("Checks", classes="section-title")
        yield from self._rows

    def _compose_details(self) -> ComposeResult:
        for row in self._rows:
            row.detail.display = row._index == self._selected
            yield row.detail

    @on(_CheckRow.Chosen)
    def _check_chosen(self, message: _CheckRow.Chosen) -> None:
        self._selected = message.index
        for row in self._rows:
            row.set_selected(row._index == message.index)


# `.status-ok` and friends never applied here: a DataTable cell is rendered
# content, not a widget, so a CSS class on the table could not reach it. The
# colour has to travel with the cell — and the word carries the state on its
# own, so nothing depends on seeing it.
_STATUS_STYLES: dict[DiagnosticStatus, tuple[str, str]] = {
    DiagnosticStatus.OK: ("OK", f"bold {palette.OK}"),
    DiagnosticStatus.WARN: ("WARN", f"bold {palette.WARN}"),
    DiagnosticStatus.FAIL: ("FAIL", f"bold {palette.BAD}"),
}


def _pretty_status(status: DiagnosticStatus) -> Text:
    label, style = _STATUS_STYLES[status]
    return Text(label, style=style)
