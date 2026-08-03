"""Byte-identical regression tests for the recommendation report.

The Fase 6 refactor moves ``report_to_json`` and ``report_to_markdown`` from
``recommendation/report.py`` into ``reporting/``. These snapshots pin the
public output of the reports so any accidental change of shape, key order or
line wording fails a test instead of silently altering downstream consumers.

If a report field genuinely needs to change, the fix is to regenerate the
snapshot in ``tests/snapshots/`` — never to relax this test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Reuse the container + fake search client from the orchestrator tests so a
# single fixture defines the "canonical run" we snapshot against.
sys.path.insert(0, str(Path(__file__).parent))

from _workflow_fixtures import answers, hardware
from jaull.workflow import orchestrator
from test_workflow_orchestrator import _container, _search_with

# Kept as absolute imports so the test survives the Fase 6 move without
# editing the imports mid-refactor — after the reporting/ package exists,
# swap the module path here and both the snapshot and the regression stay
# aligned.
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _canonical_state():
    services = _container(_search_with("org/Coder-7B"))
    return orchestrator.run_workflow(answers(), hardware(), services)


def _redact_timestamp_json(text: str) -> str:
    return re.sub(r'"timestamp": "[^"]+"', '"timestamp": "<redacted>"', text)


def _redact_timestamp_markdown(text: str) -> str:
    return re.sub(r"Generated: [^\n]+", "Generated: <redacted>", text)


def test_report_json_is_byte_identical_to_snapshot() -> None:
    from jaull.recommendation.report import report_to_json

    state = _canonical_state()
    actual = _redact_timestamp_json(report_to_json(state))
    expected = (SNAPSHOT_DIR / "report.json").read_text(encoding="utf-8")
    assert actual == expected


def test_report_markdown_is_byte_identical_to_snapshot() -> None:
    from jaull.recommendation.report import report_to_markdown

    state = _canonical_state()
    actual = _redact_timestamp_markdown(report_to_markdown(state))
    expected = (SNAPSHOT_DIR / "report.md").read_text(encoding="utf-8")
    assert actual == expected
