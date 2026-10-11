"""Run-level uncertainty, not confidence intervals or hardware calibration."""

from dataclasses import replace
from datetime import timedelta

import pytest

from jaull.domain.benchmarks import BenchmarkMeasurementKind
from jaull.recommendation.local_evidence import assess_speed, speed_cohort
from jaull.recommendation.shadow import apply_measured_policy, build_shadow_report
from tests.test_fastest_v2_applicability import _v2
from tests.test_measured_policy_activation import BALANCED, FASTEST, _measured, _order
from tests.test_recommendation_engine_v2 import _requirements
from tests.test_shadow_policies import _pool


def _runs(values):
    plan, template = _v2()
    records = [template.model_copy(update={
        "identity": template.identity.model_copy(update={
            "benchmark_id": f"run-{index}",
            "created_at": template.identity.created_at + timedelta(seconds=index),
        }),
        "observation": template.observation.model_copy(update={
            "measurements": [measurement.model_copy(update={
                "mean_tokens_per_second": value,
            }) if measurement.kind is BenchmarkMeasurementKind.GENERATION else measurement
                for measurement in template.observation.measurements],
        }),
    }) for index, value in enumerate(values)]
    return plan, records


def test_all_valid_runs_are_aggregated_without_rewriting_or_cherry_picking():
    plan, records = _runs([218.6, 227.5, 256.1, 247.1])
    before = [record.model_dump_json() for record in records]
    result = assess_speed(plan, _requirements(), records)
    measured = result.applicable
    assert measured is not None
    assert measured.tokens_per_second == 237.3
    assert measured.run_count == 4 and measured.repetitions == 5
    assert measured.observed_min == 218.6 and measured.observed_max == 256.1
    assert measured.benchmark_ids == ("run-0", "run-1", "run-2", "run-3")
    assert measured.stddev_tokens_per_second is None  # Not an averaged within-run SD.
    assert measured.ranking_ready
    assert assess_speed(plan, _requirements(), list(reversed(records))) == result
    assert assess_speed(plan, _requirements(), [*records, records[0]]) == result
    assert [record.model_dump_json() for record in records] == before


@pytest.mark.parametrize("values", [[10.0], [10.0, 20.0]])
def test_fewer_than_three_calls_are_visible_but_cannot_rank(values):
    plan, records = _runs(values)
    measured = assess_speed(plan, _requirements(), records).applicable
    assert measured is not None and measured.run_count == len(values)
    assert not measured.ranking_ready


def test_blocked_runs_and_other_artifacts_do_not_enter_the_aggregate():
    plan, records = _runs([10.0, 20.0, 30.0, 9999.0])
    blocked = records[-1].model_copy(update={
        "observation": records[-1].observation.model_copy(update={"success": False}),
    })
    other = records[-1].model_copy(update={
        "artifact": records[-1].artifact.model_copy(update={"sha256": "b" * 64}),
    })
    result = assess_speed(plan, _requirements(), [*records[:3], blocked, other])
    assert result.applicable.tokens_per_second == 20.0
    assert result.applicable.run_count == 3
    assert len(result.references) == 4


def test_conflicting_duplicate_id_cannot_select_an_arbitrary_value():
    plan, records = _runs([10.0, 20.0, 30.0, 9999.0])
    conflict = records[-1].model_copy(update={"identity": records[0].identity})
    for stored in ([*records[:3], conflict], [conflict, *reversed(records[:3])]):
        measured = assess_speed(plan, _requirements(), stored)
        assert measured.applicable.run_count == 2
        assert measured.applicable.tokens_per_second == 25.0
        assert not measured.applicable.ranking_ready
        assert any("Conflicting" in blocker
                   for ref in measured.references for blocker in ref.blockers)


def test_only_the_newest_valid_cohort_is_aggregated():
    plan, records = _runs([10.0, 20.0, 30.0, 100.0])
    newest = records[-1]
    command = list(newest.observation.command)
    command[command.index("-p") + 1] = "512,1024"
    newest = newest.model_copy(update={
        "request": newest.request.model_copy(update={"prefill_sizes": (512, 1024)}),
        "observation": newest.observation.model_copy(update={"command": tuple(command)}),
    })
    result = assess_speed(plan, _requirements(), [*records[:3], newest])
    assert result.applicable.cohort == speed_cohort(newest)
    assert result.applicable.tokens_per_second == 100.0
    assert result.applicable.benchmark_ids == ("run-3",)
    assert not result.applicable.ranking_ready


def _ranged(item, low, high, *, quality=None, runs=3):
    item = _measured(item, speed=(low + high) / 2, quality=quality)
    speed = item.assessment.speed
    measured = speed.applicable.model_copy(update={
        "run_count": runs, "observed_min": low, "observed_max": high,
        "benchmark_ids": tuple(f"{item.plan.plan_id}-{i}" for i in range(runs)),
    })
    return replace(item, assessment=item.assessment.model_copy(update={
        "speed": speed.model_copy(update={"applicable": measured}),
    }))


def test_overlapping_speed_never_reorders_fastest():
    req, (a, b) = _pool("A", "B", priority=FASTEST)
    items = [_ranged(a, 222.2, 231.6, quality=0.80),
             _ranged(b, 227.5, 256.1, quality=0.26)]
    assert apply_measured_policy(items, req) == items
    report = build_shadow_report(items, req, limit=5)
    assert any("overlap" in reason for group in report.groups for reason in group.limitations)


def test_balanced_lets_clearly_better_quality_decide_a_speed_tie():
    # The RTX 2060 pair: speeds overlap, IFEval 26 % vs 80 % (disjoint intervals).
    req, (a, b) = _pool("A", "B", priority=BALANCED)
    items = [_ranged(a, 227.5, 256.1, quality=0.26),
             _ranged(b, 222.2, 231.6, quality=0.80)]
    assert _order(items, req) == ["B", "A"]


def test_balanced_keeps_base_order_when_neither_axis_is_distinguishable():
    req, (a, b) = _pool("A", "B", priority=BALANCED)
    items = [_ranged(a, 227.5, 256.1, quality=0.40),
             _ranged(b, 222.2, 231.6, quality=0.45)]  # Wilson intervals overlap.
    assert apply_measured_policy(items, req) == items


def test_balanced_keeps_a_distinguishable_trade_off_in_base_order():
    req, (a, b) = _pool("A", "B", priority=BALANCED)
    items = [_ranged(a, 131.4, 150.9, quality=0.80),   # Clearly better, clearly slower.
             _ranged(b, 222.2, 231.6, quality=0.26)]
    assert apply_measured_policy(items, req) == items


def test_balanced_moves_a_faster_plan_whose_quality_is_not_distinguishable():
    req, (a, b) = _pool("A", "B", priority=BALANCED)
    items = [_ranged(a, 131.4, 150.9, quality=0.45),
             _ranged(b, 222.2, 231.6, quality=0.40)]
    assert _order(items, req) == ["B", "A"]


@pytest.mark.parametrize("priority", [FASTEST, BALANCED])
def test_separate_ranges_can_reorder(priority):
    req, (a, b) = _pool("A", "B", priority=priority)
    items = [_ranged(a, 131.4, 150.9, quality=0.41),
             _ranged(b, 222.2, 231.6, quality=0.80)]
    assert _order(items, req) == ["B", "A"]
    report = build_shadow_report(items, req, limit=5)
    assert len(report.moves[0].evidence_ids) == (4 if priority == BALANCED else 3)


def test_chained_overlap_respects_the_direct_comparison_in_fastest():
    # A~B and B~C overlap, but C is demonstrably faster than A: only that pair moves.
    req, (a, b, c) = _pool("A", "B", "C", priority=FASTEST)
    items = [_ranged(a, 10, 20), _ranged(b, 19, 30), _ranged(c, 29, 40)]
    assert _order(items, req) == ["C", "A", "B"]  # C rises to A's slot; A stays before B.


def test_chained_overlap_with_quality_in_balanced():
    # Speeds chain-overlap; quality is pairwise disjoint (n=100): each pair is decided.
    req, (a, b, c) = _pool("A", "B", "C", priority=BALANCED)
    items = [_ranged(a, 10, 20, quality=0.15), _ranged(b, 19, 30, quality=0.5),
             _ranged(c, 29, 40, quality=0.9)]
    assert _order(items, req) == ["C", "B", "A"]


def test_chained_overlap_never_inverts_a_demonstrable_pair():
    # A~B, B~C, C~D chain; only A<C, A<D and B<D are demonstrable (all faster later).
    req, items = _pool("A", "B", "C", "D", priority=FASTEST)
    ranges = {"A": (10, 20), "B": (19, 30), "C": (25, 35), "D": (31, 45)}
    measured = [_ranged(item, *ranges[name]) for item, name in zip(items, "ABCD", strict=True)]
    order = _order(measured, req)
    for slow, fast in (("A", "C"), ("A", "D"), ("B", "D")):
        assert order.index(fast) < order.index(slow)


@pytest.mark.parametrize("priority", [FASTEST, BALANCED])
def test_touching_ranges_are_unresolved(priority):
    req, (a, b) = _pool("A", "B", priority=priority)
    # Quality overlaps too, so neither axis is distinguishable.
    items = [_ranged(a, 10, 20, quality=0.40), _ranged(b, 20, 30, quality=0.45)]
    assert apply_measured_policy(items, req) == items


@pytest.mark.parametrize("priority", [FASTEST, BALANCED])
def test_insufficient_runs_cannot_reorder(priority):
    req, (a, b) = _pool("A", "B", priority=priority)
    items = [_ranged(a, 10, 20, quality=0.3), _ranged(b, 100, 110, quality=0.9, runs=1)]
    assert apply_measured_policy(items, req) == items
    report = build_shadow_report(items, req, limit=5)
    assert any("three" in reason for fallback in report.fallback for reason in fallback.reasons)


@pytest.mark.parametrize("priority", [FASTEST, BALANCED])
def test_overlap_band_does_not_hide_a_clearly_dominated_plan(priority):
    req, (a, b, c) = _pool("A", "B", "C", priority=priority)
    items = [_ranged(a, 131.4, 150.9, quality=0.2),
             _ranged(b, 222.2, 231.6, quality=0.8),
             _ranged(c, 227.5, 256.1, quality=0.26)]
    assert _order(items, req) == ["B", "C", "A"]


def test_report_exports_the_aggregate_and_every_contributing_id(monkeypatch):
    from jaull.application.recommendation.service import recommend
    from jaull.domain.recommendation import SpeedAssessment
    from jaull.recommendation import engine_v2
    from jaull.recommendation.engine_v2 import PlanRankingContext
    from jaull.reporting.recommendation import report_to_dict, report_to_markdown
    from jaull.workflow.state import RecommendationWorkflowState
    from tests._workflow_fixtures import hardware
    from tests.test_ifeval_applicability import _gguf

    plan, records = _runs([10.0, 20.0, 30.0])
    aggregate = assess_speed(plan, _requirements(), records)
    monkeypatch.setattr(engine_v2, "assess_speed", lambda plan, req, stored:
                        aggregate if plan.artifact.sha256 == "a" * 64 else SpeedAssessment())
    requirements = _requirements(FASTEST)
    pools = []
    shown = recommend([_gguf("org/A-7B-GGUF", "a" * 64)], requirements,
                      hardware=hardware(), plan_context=PlanRankingContext(hardware=hardware()),
                      on_ranked_plans=pools.append)
    state = RecommendationWorkflowState(requirements=requirements, recommendations=shown,
                                        ranked_plans=tuple(pools[0]))
    before = state.model_dump_json()
    policy = report_to_dict(state)["recommendations"][0]["ranking"]["speed_policy"]
    measurement = policy["measurement"]
    assert measurement["benchmark_ids"] == ["run-0", "run-1", "run-2"]
    assert measurement["tokens_per_second"] == 20.0 and measurement["run_count"] == 3
    markdown = report_to_markdown(state)
    assert "median: 20.0" in markdown and "3 calls, 5 repetitions per call" in markdown
    assert "10.0-30.0" in markdown and "not a confidence interval" in markdown
    assert all(identifier in markdown for identifier in measurement["benchmark_ids"])
    assert state.model_dump_json() == before
