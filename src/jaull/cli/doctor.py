"""CLI presentation for the doctor command.

Diagnostics collection lives in ``jaull.diagnostics.service`` — this module
only turns the results into a Rich table and picks the exit code.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from jaull.advisor.service import AdvisorService
from jaull.domain.enums import DiagnosticStatus
from jaull.domain.model import DiagnosticResult
from jaull.presentation.console import make_console


def run_doctor(advisor: AdvisorService | None = None) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    results = resolved.diagnostics()
    _render(results, console)
    return 0 if all(r.status is not DiagnosticStatus.FAIL for r in results) else 6


_STATUS_STYLE = {
    DiagnosticStatus.OK: ("[green]OK[/green]", "green"),
    DiagnosticStatus.WARN: ("[yellow]WARN[/yellow]", "yellow"),
    DiagnosticStatus.FAIL: ("[red]FAIL[/red]", "red"),
}


def _render(results: list[DiagnosticResult], console: Console) -> None:
    table = Table(title="Doctor")
    table.add_column("Check", style="bold cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")

    for result in results:
        label, _ = _STATUS_STYLE[result.status]
        table.add_row(result.name, label, result.detail)

    console.print(table)


__all__ = ["run_doctor"]
