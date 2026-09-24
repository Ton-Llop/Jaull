"""Byte-identical regression tests for the recommendation report.

The Fase 6 refactor moves ``report_to_json`` and ``report_to_markdown`` from
``recommendation/report.py`` into ``reporting/``. These snapshots pin the
public output of the reports so any accidental change of shape, key order or
line wording fails a test instead of silently altering downstream consumers.

If a report field genuinely needs to change, the fix is to regenerate the
snapshot in ``tests/snapshots/`` — never to relax this test.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Reuse the container + fake search client from the orchestrator tests so a
# single fixture defines the "canonical run" we snapshot against.
sys.path.insert(0, str(Path(__file__).parent))

from _workflow_fixtures import answers, hardware
from jaull.domain.estimation import CompatibilityStatus
from jaull.reporting.recommendation import REPORT_SCHEMA_VERSION
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
    payload = json.loads(text)
    payload["timestamp"] = "<redacted>"
    # Timings are intentionally non-deterministic. Their shape is asserted in
    # the dedicated test below; semantic report snapshots stay stable.
    payload.pop("workflow_telemetry", None)
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


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


def test_report_separates_unknown_plan_without_changing_rank_or_score() -> None:
    from jaull.recommendation.report import report_to_dict, report_to_markdown

    state = _canonical_state()
    confirmed = state.recommendations[0]
    assert confirmed.plan is not None
    estimate = confirmed.plan.memory_prediction
    assert estimate is not None
    unknown_assessment = estimate.assessment.model_copy(
        update={
            "status": CompatibilityStatus.UNKNOWN,
            "reasons": ["KV cache could not be estimated."],
        }
    )
    unknown_plan = confirmed.plan.model_copy(
        update={
            "memory_prediction": estimate.model_copy(
                update={"assessment": unknown_assessment}
            )
        }
    )
    unknown = confirmed.model_copy(update={"rank": 2, "plan": unknown_plan})
    mixed = state.model_copy(update={"recommendations": [confirmed, unknown]})

    markdown = report_to_markdown(mixed)
    assert "## Confirmed recommendations" in markdown
    assert "## Unconfirmed alternatives" in markdown
    assert "KV cache could not be estimated." in markdown
    payload = report_to_dict(mixed)
    assert [item["rank"] for item in payload["recommendations"]] == [1, 2]
    assert payload["recommendations"][1]["compatibility"] == "unknown"
    assert payload["recommendations"][1]["score_breakdown"] == payload[
        "recommendations"
    ][0]["score_breakdown"]

    only_unknown = state.model_copy(
        update={"recommendations": [unknown.model_copy(update={"rank": 1})]}
    )
    assert "No confirmed recommendations were found." in report_to_markdown(only_unknown)


def test_report_json_exposes_technical_latency_separately() -> None:
    from jaull.recommendation.report import report_to_dict

    payload = report_to_dict(_canonical_state())
    telemetry = payload["workflow_telemetry"]
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    assert telemetry["wall_seconds"] is not None
    assert isinstance(telemetry["phase_seconds"], dict)
    assert isinstance(telemetry["counters"], dict)
    assert telemetry["candidate_latency"]
