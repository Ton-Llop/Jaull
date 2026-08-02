from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, LoadingIndicator

from local_ai_check.cli.doctor import collect_diagnostics
from local_ai_check.domain.enums import DiagnosticStatus
from local_ai_check.domain.model import DiagnosticResult
from local_ai_check.tui.widgets.banner import Banner


class DoctorScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner("Environment diagnostics", "Runs the doctor checks and shows their status.")
        yield LoadingIndicator(id="doctor-loading")
        yield VerticalScroll(Vertical(id="doctor-content"))
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._worker, thread=True)

    def _worker(self) -> None:
        results = collect_diagnostics()
        self.app.call_from_thread(self._populate, results)

    def _populate(self, results: list[DiagnosticResult]) -> None:
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
