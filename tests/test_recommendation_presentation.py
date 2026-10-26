"""Presentation distinguishes diagnostic scores and independent evidence scopes."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App
from textual.widgets import Static

from jaull.domain.estimation import EstimateSource, EstimationConfidence
from jaull.reporting.recommendation import report_to_dict, report_to_markdown
from jaull.reporting.recommendation_labels import (
    confidence_rows,
    memory_confidence_rows,
    score_rows,
)
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_results import (
    RecommendationResultsScreen,
    _assessment_rows,
    _comparison_table,
)
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
    assert rows["Weights estimate"].startswith("high (")
    assert rows["KV cache estimate"].startswith("high (")
    assert rows["Runtime overhead estimate"].startswith("low (assumed heuristic)")
    assert "Weakest component" in rows["Memory confidence aggregation"]
    assert rows["Plan evidence confidence"] == "high"
    assert "runtime overhead (assumed heuristic)" in rows["Memory estimation limitations"]
    assert "not a measurement" in rows["Memory estimation limitations"]
    assert "stored-benchmark" in rows["Plan evidence basis"]
    assert "Not a change to memory-estimate confidence" in rows["Plan evidence basis"]
    assert rows["Performance evidence"] == "No comparable benchmark for ranking"
    assert rows["Quality evidence"] == "No applicable evaluation"
    markdown = report_to_markdown(state)
    main, technical = markdown.split("<details>", maxsplit=1)
    assert "Memory estimation confidence:" not in main
    assert "Runtime overhead estimate: low (assumed heuristic)" in main
    assert "Memory estimation confidence: low" in technical
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
    assert memory_confidence_rows(rec) == []
    assert "legacy ranking" in score_rows(rec)[0][0]
    assert "orders this list" in score_rows(rec)[1][1]


@pytest.mark.parametrize("selected_plan_only", [False, True])
def test_results_keep_diagnostics_collapsed_and_do_not_mutate_recommendations(selected_plan_only):
    async def scenario():
        state = _state()
        if selected_plan_only:
            rec = state.recommendations[0]
            rec = rec.model_copy(update={"evaluated": rec.evaluated.model_copy(update={
                "memory_estimate": None,
            })})
            state = state.model_copy(update={"recommendations": [rec]})
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
            assert "Weights estimate: high" in text
            assert "KV cache estimate: high" in text
            assert "Runtime overhead estimate: low (assumed heuristic)" in text
            assert "Plan evidence confidence" in text
            assert "runtime overhead (assumed heuristic)" in text
            assert "Diagnostic composite score" in text
            assert "This is not what ordered the list" in text
            all_text = " ".join(str(w.content) for w in screen.query(Static))
            assert "Memory estimate basis" in all_text
            assert "Memory breakdown unavailable." not in all_text
            assert state.model_dump_json() == before

    asyncio.run(scenario())


@pytest.mark.parametrize("confidence,source", [
    (EstimationConfidence.MEDIUM, EstimateSource.DERIVED),
    (EstimationConfidence.UNKNOWN, EstimateSource.UNKNOWN),
])
def test_memory_breakdown_keeps_the_selected_plan_component_evidence(confidence, source):
    rec = _state().recommendations[0]
    estimate = rec.plan.memory_prediction
    cache = estimate.kv_cache.component.model_copy(update={
        "confidence": confidence, "source": source, "explanation": "Recorded KV limitation.",
    })
    selected = estimate.model_copy(update={
        "kv_cache": estimate.kv_cache.model_copy(update={"component": cache}),
    })
    rec = rec.model_copy(update={
        "plan": rec.plan.model_copy(update={"memory_prediction": selected}),
    })
    before = rec.model_dump_json()
    rows = dict(memory_confidence_rows(rec, detailed=True))
    assert rows["KV cache estimate"].startswith(confidence.value)
    assert "Recorded KV limitation." in rows["KV cache estimate"]
    assert dict(confidence_rows(rec))["Memory estimation confidence"] == "low"
    assert rec.model_dump_json() == before


def test_comparison_displays_components_instead_of_a_global_low_grade():
    async def scenario():
        app = App()
        rec = _state().recommendations[0]
        before = rec.model_dump_json()
        async with app.run_test():
            table = _comparison_table([rec])
            await app.screen.mount(table)
            assert "Memory estimate basis" in [str(c.label) for c in table.columns.values()]
            cell = table.get_row_at(0)[4]
            assert "Weights estimate: high" in cell
            assert "KV cache estimate: high" in cell
            assert "Runtime overhead estimate: low (assumed heuristic)" in cell
            assert table.rows[next(iter(table.rows))].height == 3
            assert rec.model_dump_json() == before

    asyncio.run(scenario())


def test_the_context_bar_names_a_vulkan_card_that_nvml_does_not_see() -> None:
    """Regression found on the B60 box: the bar showed only CPU and RAM while the plans ran
    on the card, because only NVIDIA (NVML) GPUs were ever added to the summary."""
    from jaull.presentation.plan_labels import hardware_summary
    from tests._intel_fixtures import LINUX_OS, scan

    summary = hardware_summary(scan(LINUX_OS))

    assert "B60" in summary and "24 GiB" in summary  # the fixture's 24 GiB heap
    assert "(R)" not in summary and "(TM)" not in summary
