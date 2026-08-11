from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, LoadingIndicator

from jaull.domain.enums import DiagnosticStatus
from jaull.domain.model import DiagnosticResult
from jaull.tui.widgets.banner import Banner


class _DoctorFinished(Message):
    def __init__(self, results: list[DiagnosticResult]) -> None:
        super().__init__()
        self.results = results


class DoctorScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._doctor_closing = Event()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner("Environment diagnostics", "Runs the doctor checks and shows their status.")
        yield LoadingIndicator(id="doctor-loading")
        yield VerticalScroll(Vertical(id="doctor-content"))
        yield Footer()

    def on_mount(self) -> None:
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
        table: DataTable[str] = DataTable(zebra_stripes=True)
        table.add_columns("Check", "Status", "Detail")
        for result in results:
            table.add_row(result.name, _pretty_status(result.status), result.detail)
        content = self.query_one("#doctor-content", Vertical)
        content.remove_children()
        content.mount(table)


def _pretty_status(status: DiagnosticStatus) -> str:
    return {
        DiagnosticStatus.OK: "OK",
        DiagnosticStatus.WARN: "WARN",
        DiagnosticStatus.FAIL: "FAIL",
    }[status]
