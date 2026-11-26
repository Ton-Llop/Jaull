from __future__ import annotations

import json
import sys

from jaull.advisor.service import AdvisorService
from jaull.presentation.console import make_console
from jaull.presentation.hardware_report import render_hardware


def run_scan(advisor: AdvisorService | None = None, *, as_json: bool = False) -> int:
    resolved = advisor or AdvisorService.default()
    profile = resolved.scan_hardware()
    if as_json:
        payload = {"schema_version": 1, "hardware": profile.model_dump(mode="json")}
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    render_hardware(profile, make_console())
    return 0
