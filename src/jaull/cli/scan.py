from __future__ import annotations

from jaull.advisor.service import AdvisorService
from jaull.presentation.console import make_console
from jaull.presentation.hardware_report import render_hardware


def run_scan(advisor: AdvisorService | None = None) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    profile = resolved.scan_hardware()
    render_hardware(profile, console)
    return 0
