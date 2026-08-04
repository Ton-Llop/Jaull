"""Compatibility shim.

The report generators now live in ``jaull.reporting.recommendation``. This
module re-exports them so existing callers (tests, CLI, TUI) keep working
without an extra edit. Prefer importing from ``jaull.reporting.recommendation``
in new code.
"""

from __future__ import annotations

from jaull.reporting.recommendation import (
    REPORT_SCHEMA_VERSION,
    report_to_dict,
    report_to_json,
    report_to_markdown,
)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "report_to_dict",
    "report_to_json",
    "report_to_markdown",
]
