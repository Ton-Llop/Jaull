from __future__ import annotations

from local_ai_check.hardware.detector import detect_hardware
from local_ai_check.presentation.console import make_console
from local_ai_check.presentation.hardware_report import render_hardware


def run_scan() -> int:
    console = make_console()
    profile = detect_hardware()
    render_hardware(profile, console)
    return 0
