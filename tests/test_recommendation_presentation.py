"""Presentation distinguishes diagnostic scores and independent evidence scopes."""

from __future__ import annotations

import asyncio

from textual.widgets import Static

from jaull.domain.estimation import EstimateSource, EstimationConfidence
from jaull.reporting.recommendation import report_to_dict, report_to_markdown
from jaull.reporting.recommendation_labels import confidence_rows, score_rows
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_results import RecommendationResultsScreen, _assessment_rows
from jaull.tui.widgets.technical_details import TechnicalDetails
from tests.test_reporting_regression import _canonical_state
from tests.test_tui_recommendation_execution import _FakeAdvisor


def _state():
    state = _canonical_state()
    rec = state.recommendations[0]
    estimate = rec.plan.memory_prediction
    component = estimate.runtime_overhead.component.model_copy(update={
        "confidence": EstimationConfidence.LOW,
        "source": EstimateSource.ASSUMED,
        "explanation": "Runtime overhead is a documented heuristic, not a measurement.",
    })
    estimate = estimate.model_copy(update={
        "assessment": estimate.assessment.model_copy(update={
            "confidence": EstimationConfidence.LOW,
        }),
        "runtime_overhead": estimate.runtime_overhead.model_copy(update={"component": component}),
    })
    rec = rec.model_copy(update={
        "confidence": EstimationConfidence.LOW,
        "plan": rec.plan.model_copy(update={"memory_prediction": estimate}),
        "evaluated": rec.evaluated.model_copy(update={"memory_estimate": estimate}),
        "plan_assessment": rec.plan_assessment.model_copy(update={
            "confidence": EstimationConfidence.HIGH,
            "local_benchmark_id": "stored-benchmark",
        }),
    })
    return state.model_copy(update={"recommendations": [rec]})


def test_memory_and_plan_confidence_are_distinct_and_name_the_low_component():
    state = _state()
    rec = state.recommendations[0]
    rows = dict(confidence_rows(rec))
    assert rows["Memory estimation confidence"] == "low"
    assert rows["Plan evidence confidence"] == "high"
    assert "runtime overhead (assumed heuristic)" in rows["Memory estimation limitations"]
    assert "not a measurement" in rows["Memory estimation limitations"]
    assert "stored-benchmark" in rows["Plan evidence basis"]
    assert "Not a change to memory-estimate confidence" in rows["Plan evidence basis"]
    assert rows["Performance evidence"] == "No comparable benchmark for ranking"
    assert rows["Quality evidence"] == "No applicable evaluation"
    markdown = report_to_markdown(state)
    for label, value in rows.items():
        assert f"- {label}: {value}" in markdown
    assert "\n- Confidence:" not in markdown
    assert "plan evidence confidence: high" in markdown
    assert dict(_assessment_rows(rec))["Plan evidence confidence"] == "high"


def test_diagnostic_score_order_does_not_change_rank_or_serialized_data():
    state = _state()
    first = state.recommendations[0]
    first = first.model_copy(update={"score": first.score.model_copy(update={"total": 0.53})})
    second = first.model_copy(update={
        "rank": 2,
        "score": first.score.model_copy(update={"total": 0.64}),
        "evaluated": first.evaluated.model_copy(update={
            "candidate": first.evaluated.candidate.model_copy(update={"repo_id": "org/Second"}),
        }),
    })
    state = state.model_copy(update={"recommendations": [first, second]})
    before = state.model_dump_json()
    payload = report_to_dict(state)
    markdown = report_to_markdown(state)
    assert markdown.index("### 1.") < markdown.index("### 2.")
    assert [r["score"] for r in payload["recommendations"]] == [53, 64]
    assert [r["rank"] for r in payload["recommendations"]] == [1, 2]
    assert "a higher score need not rank first" in markdown
    assert markdown.index("Why this position?") < markdown.index("<details>")
    assert markdown.index("<details>") < markdown.index("Diagnostic composite score: 53/100")
    assert payload["recommendations"][0]["confidence"] == "low"
    axes = payload["recommendations"][0]["ranking"]["criteria"]
    assert next(axis["value"] for axis in axes if axis["axis"] == "plan_confidence") == "high"
    assert state.model_dump_json() == before


def test_legacy_record_without_estimate_does_not_claim_memory_confidence_or_diagnostic_ranking():
    rec = _state().recommendations[0]
    rec = rec.model_copy(update={
        "plan": None, "plan_assessment": None,
        "evaluated": rec.evaluated.model_copy(update={"memory_estimate": None}),
    })
    assert confidence_rows(rec) == [
        ("Metadata confidence", rec.evaluated.candidate.metadata_confidence.value),
    ]
    assert "legacy ranking" in score_rows(rec)[0][0]
    assert "orders this list" in score_rows(rec)[1][1]


def test_results_keep_diagnostics_collapsed_and_do_not_mutate_recommendations():
    async def scenario():
        state = _state()
        before = state.model_dump_json()
        app = JaullApp(advisor=_FakeAdvisor())
        async with app.run_test(size=(120, 50)) as pilot:
            screen = RecommendationResultsScreen(state)
            await app.push_screen(screen)
            await pilot.pause()
            row_text = " ".join(
                str(w.content) for w in screen.query_one("#rec-row-0").query(Static)
            )
            assert "Diagnostic composite score" not in row_text
            diagnostics = next(w for w in screen.query(TechnicalDetails)
                               if w.title == "Confidence and diagnostics")
            assert diagnostics.collapsed
            text = " ".join(str(w.content) for w in diagnostics.query(Static))
            assert "Memory estimation confidence" in text
            assert "Plan evidence confidence" in text
            assert "runtime overhead (assumed heuristic)" in text
            assert "Diagnostic composite score" in text
            assert "This is not what ordered the list" in text
            assert state.model_dump_json() == before

    asyncio.run(scenario())
