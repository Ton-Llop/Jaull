from __future__ import annotations

from jaull.hardware.detector import detect_hardware
from jaull.presentation.console import make_console
from jaull.presentation.hardware_report import render_hardware


def run_scan() -> int:
    console = make_console()
    profile = detect_hardware()
    render_hardware(profile, console)
    return 0
