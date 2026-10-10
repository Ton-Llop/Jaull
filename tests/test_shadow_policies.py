"""Shadow-v1 policies on synthetic, labelled evidence; not measured results.

Every `Measured` below is invented to exercise one rule. Production attaches no
applicable evidence yet, which the last tests check through the real workflow.
"""

from copy import deepcopy

import pytest

from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.recommendation import AssessmentLevel
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    RankedPlan,
    assess_plan,
    generate_execution_plans,
    rank_execution_plans,
)
from jaull.recommendation.shadow import Measured, PlanEvidence, build_shadow_report
from jaull.workflow import orchestrator
from tests._workflow_fixtures import answers, hardware
from tests.test_quality_eval_records import synthetic_limited_record
from tests.test_recommendation_engine_v2 import _evaluated_gguf, _requirements
from tests.test_workflow_orchestrator import _container, _search_with

Q, B, S, M = (RecommendationPriority.QUALITY, RecommendationPriority.BALANCED,
              RecommendationPriority.SPEED, RecommendationPriority.MEMORY)


def _pool(*names: str, priority: RecommendationPriority = Q):
    """Plans in exactly this base order, one distinct logical model each."""
    requirements = _requirements(priority)
    items = []
    for name in names:
        evaluated = _evaluated_gguf(f"org/{name}-7B-GGUF", priority=priority)
        plan = generate_execution_plans(evaluated, requirements)[0]
        items.append(RankedPlan(evaluated, plan, assess_plan(evaluated, plan, requirements)))
    return requirements, items


def quality(value: float, cohort: str = "ifeval-v1") -> Measured:
    return Measured(cohort=cohort, value=value, label="prompt_level_strict_acc",
                    evidence_ids=(f"q:{cohort}:{value}",))


def speed(value: float, cohort: str = "tg128|machine-1|build-1") -> Measured:
    return Measured(cohort=cohort, value=value, label="tg128 tok/s",
                    evidence_ids=(f"s:{cohort}:{value}",))


def _shadow(items, requirements, **evidence: PlanEvidence):
    by_plan = {items[int(k[1:])].plan.plan_id: v for k, v in evidence.items()}
    return build_shadow_report(items, requirements, limit=5, evidence=by_plan)


def _repos(report, items, field: str = "shadow_order") -> list[str]:
    names = {item.plan.plan_id: item.evaluated.repo_id.split("/")[1][0] for item in items}
    return [names[plan_id] for plan_id in getattr(report, field)]


def test_relevant_quality_outranks_the_prior_inside_its_group() -> None:
    requirements, items = _pool("A", "B", "C")
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(0.5)), p2=PlanEvidence(quality=quality(0.9)))

    # C takes A's slot; B, unmeasured, keeps its own between them.
    assert _repos(report, items) == ["C", "B", "A"]
    assert {move.rule for move in report.moves} == {
        "prompt_level_strict_acc 0.9", "prompt_level_strict_acc 0.5",
    }
    assert report.moves[0].evidence_ids == ("q:ifeval-v1:0.9",)


def test_a_lone_measurement_is_not_promoted_for_existing() -> None:
    requirements, items = _pool("A", "B", "C", priority=S)
    report = _shadow(items, requirements, p2=PlanEvidence(speed=speed(500)))

    assert report.shadow_order == report.base_order
    assert report.groups == () and report.moves == ()


def test_a_slow_measured_plan_gets_no_bonus_over_unmeasured_ones() -> None:
    requirements, items = _pool("A", "B", "C", priority=S)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(speed=speed(5)), p2=PlanEvidence(speed=speed(50)))

    assert _repos(report, items) == ["C", "B", "A"]


def test_different_generation_loads_are_never_compared() -> None:
    requirements, items = _pool("A", "B", priority=S)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(speed=speed(5, "tg128|m|b")),
                     p1=PlanEvidence(speed=speed(50, "tg256|m|b")))

    assert report.shadow_order == report.base_order
    assert report.groups == ()


def test_equal_values_keep_the_base_order() -> None:
    requirements, items = _pool("A", "B", "C")
    report = _shadow(items, requirements, **{
        f"p{i}": PlanEvidence(quality=quality(0.7)) for i in range(3)
    })

    assert report.shadow_order == report.base_order
    assert report.moves == ()


def test_pareto_orders_dominated_plans_and_keeps_trade_offs() -> None:
    requirements, items = _pool("C", "B", "A", priority=B)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(0.5), speed=speed(5)),    # C: dominated
                     p1=PlanEvidence(quality=quality(0.6), speed=speed(20)),   # B: faster
                     p2=PlanEvidence(quality=quality(0.8), speed=speed(10)))   # A: better

    # A and B are both on the first front; neither is dropped for "barely" winning.
    assert _repos(report, items) == ["B", "A", "C"]
    rules = {move.repo_id.split("/")[1][0]: move.rule for move in report.moves}
    assert rules["C"].startswith("Pareto front 2")
    assert rules["A"].startswith("Pareto front 1")


def test_balanced_infers_no_dominance_from_one_known_axis() -> None:
    requirements, items = _pool("A", "B", priority=B)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(0.1), speed=speed(1)),
                     p1=PlanEvidence(quality=quality(0.9)))

    assert report.shadow_order == report.base_order
    assert report.groups == ()


def test_evidence_never_carries_a_plan_across_a_stratum() -> None:
    requirements, items = _pool("A", "B", "C")
    weak = items[1].assessment.model_copy(update={"suitability": AssessmentLevel.WEAK})
    items[1] = RankedPlan(items[1].evaluated, items[1].plan, weak)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(0.1)), p2=PlanEvidence(quality=quality(0.9)))

    # A and C share a stratum, but B sits between them in another one.
    assert report.shadow_order == report.base_order
    assert report.groups == ()


@pytest.mark.parametrize("priority", [Q, S, B])
def test_evidence_cannot_cross_the_confirmed_placement_partition(priority) -> None:
    requirements, items = _pool("A", "B", priority=priority)
    item = items[1]
    prediction = item.plan.memory_prediction
    assert prediction is not None
    unknown = item.plan.model_copy(update={"memory_prediction": prediction.model_copy(update={
        "assessment": prediction.assessment.model_copy(update={
            "status": CompatibilityStatus.UNKNOWN,
        }),
    })})
    items[1] = RankedPlan(
        item.evaluated, unknown, assess_plan(item.evaluated, unknown, requirements),
    )
    assert not items[1].assessment.rejected
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(.1), speed=speed(1)),
                     p1=PlanEvidence(quality=quality(.9), speed=speed(9)))
    assert report.shadow_order == report.base_order
    assert not report.groups


def test_balanced_cohort_pairs_are_not_concatenated_into_an_ambiguous_key() -> None:
    requirements, items = _pool("A", "B", priority=B)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(.1, "a+b"), speed=speed(1, "c")),
                     p1=PlanEvidence(quality=quality(.9, "a"), speed=speed(9, "b+c")))
    assert report.shadow_order == report.base_order
    assert not report.groups


@pytest.mark.parametrize("priority", [Q, S, B])
def test_moves_cite_only_evidence_used_by_the_selected_policy(priority) -> None:
    requirements, items = _pool("A", "B", priority=priority)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(.1), speed=speed(1)),
                     p1=PlanEvidence(quality=quality(.9), speed=speed(9)))
    assert report.moves
    for move in report.moves:
        kinds = {evidence_id.split(":")[0] for evidence_id in move.evidence_ids}
        assert kinds == ({"q"} if priority is Q else {"s"} if priority is S else {"q", "s"})


def test_memory_priority_keeps_the_active_order() -> None:
    requirements, items = _pool("A", "B", priority=M)
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(0.1), speed=speed(1)),
                     p1=PlanEvidence(quality=quality(0.9), speed=speed(9)))

    assert not report.applied
    assert report.shadow_order == report.base_order


def test_groups_are_disjoint() -> None:
    requirements, items = _pool("A", "B", "C", "D")
    report = _shadow(items, requirements,
                     p0=PlanEvidence(quality=quality(0.1, "x")),
                     p1=PlanEvidence(quality=quality(0.2, "y")),
                     p2=PlanEvidence(quality=quality(0.3, "x")),
                     p3=PlanEvidence(quality=quality(0.4, "y")))

    seen = [plan_id for group in report.groups for plan_id in group.plan_ids]
    assert len(seen) == len(set(seen)) == 4
    assert _repos(report, items) == ["C", "D", "A", "B"]


def test_top5_changes_are_reported_apart_from_moves() -> None:
    requirements, items = _pool("A", "B", "C", "D", "E", "F", "G")
    report = _shadow(items, requirements,
                     p4=PlanEvidence(quality=quality(0.2)), p6=PlanEvidence(quality=quality(0.9)))

    assert _repos(report, items, "base_top5") == ["A", "B", "C", "D", "E"]
    assert _repos(report, items, "shadow_top5") == ["A", "B", "C", "D", "G"]
    changes = {(c.plan_id, c.change, c.moved) for c in report.top5_changes}
    assert changes == {(items[6].plan.plan_id, "entered", True),
                       (items[4].plan.plan_id, "left", True)}


def test_a_hellaswag_record_never_orders_a_coding_request() -> None:
    """Production path: a complete scoring record stays a reference, so nothing moves."""
    requirements = _requirements(Q)
    record = synthetic_limited_record()
    evaluated = []
    for name in ("A", "B"):
        candidate = _evaluated_gguf(f"org/{name}-7B-GGUF", priority=Q)
        assert candidate.analysis is not None
        for variant in candidate.analysis.classification.gguf_variants:
            for index, file in enumerate(variant.files):
                variant.files[index] = file.model_copy(
                    update={"sha256": record["identity"]["artifact_sha256"]},
                )
        evaluated.append(candidate)
    ranked = rank_execution_plans(evaluated, requirements,
                                  context=PlanRankingContext(quality_records=[record]))

    report = build_shadow_report(ranked, requirements, limit=5)

    assert all(item.assessment.quality.local_references for item in ranked)
    assert report.shadow_order == report.base_order
    assert all("references only" in fallback.reasons[0] for fallback in report.fallback)


def test_without_applicable_evidence_every_plan_falls_back_with_reasons() -> None:
    requirements, items = _pool("A", "B")
    report = build_shadow_report(items, requirements, limit=5)

    assert report.shadow_order == report.base_order
    assert [f.plan_id for f in report.fallback] == list(report.base_order)
    assert all("SHA256 unavailable" in f.reasons[0] for f in report.fallback)
    assert all(any("no evidence for this artifact" in r for r in f.reasons)
               for f in report.fallback)
    assert all(f.repo_id and f.artifact_label for f in report.fallback)
    assert all("SHA256 unavailable" in " ".join(f.reasons) for f in report.fallback)


@pytest.mark.parametrize("separation", ["protocol", "stratum", "singleton"])
def test_fallback_explains_why_applicable_evidence_cannot_form_a_group(separation) -> None:
    requirements, items = _pool("A", "B")
    evidence = {items[0].plan.plan_id: PlanEvidence(quality=quality(0.5))}
    if separation != "singleton":
        cohort = "different-protocol" if separation == "protocol" else "ifeval-v1"
        evidence[items[1].plan.plan_id] = PlanEvidence(quality=quality(0.9, cohort))
    if separation == "stratum":
        items[1] = RankedPlan(items[1].evaluated, items[1].plan,
                              items[1].assessment.model_copy(update={
                                  "suitability": AssessmentLevel.WEAK,
                              }))
    report = build_shadow_report(items, requirements, limit=5, evidence=evidence)
    assert not report.groups and not report.moves
    expected = {
        "protocol": "no matching comparison protocol",
        "stratum": "stratum boundary",
        "singleton": "No second eligible plan",
    }[separation]
    assert expected in " ".join(report.fallback[0].reasons)


def test_the_shadow_is_independent_of_candidate_input_order() -> None:
    requirements = _requirements(B)
    evaluated = [_evaluated_gguf(f"org/{n}-7B-GGUF") for n in ("A", "B", "C")]

    forward = build_shadow_report(rank_execution_plans(evaluated, requirements),
                                  requirements, limit=5)
    backward = build_shadow_report(rank_execution_plans(evaluated[::-1], requirements),
                                   requirements, limit=5)

    assert forward == backward


def test_the_workflow_keeps_its_result_and_reports_the_same_top5() -> None:
    services = _container(_search_with("org/Coder-7B", "org/Coder-13B"))
    state = orchestrator.run_workflow(answers(), hardware(), services)

    assert state.shadow is not None
    assert list(state.shadow.base_top5) == [r.plan.plan_id for r in state.recommendations]
    assert tuple(item.plan.plan_id for item in state.ranked_plans) == state.shadow.base_order
    # Nothing applicable in production yet: the proposal equals the active order.
    assert state.shadow.shadow_order == state.shadow.base_order


def test_a_failing_shadow_never_costs_the_real_result(monkeypatch) -> None:
    services = _container(_search_with("org/Coder-7B", "org/Coder-13B"))
    expected = orchestrator.run_workflow(answers(), hardware(), services)

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic shadow failure")

    monkeypatch.setattr(orchestrator, "build_shadow_report", broken)
    state = orchestrator.run_workflow(answers(), hardware(), deepcopy(services))

    assert state.shadow is None
    assert state.ranked_plans
    assert [r.plan.plan_id for r in state.recommendations] == (
        [r.plan.plan_id for r in expected.recommendations]
    )


def test_a_cancelled_new_search_does_not_retain_the_previous_shadow() -> None:
    services = _container(_search_with("org/Coder-7B"))
    previous = orchestrator.run_workflow(answers(), hardware(), services)
    assert previous.shadow is not None
    state = orchestrator.run_workflow(
        answers(), hardware(), services, state=previous, is_cancelled=lambda: True,
    )
    assert state.shadow is None
    assert state.ranked_plans == ()
    assert state.errors == ["Search cancelled."]
    assert previous.shadow is not None


@pytest.mark.parametrize("priority", [Q, B, S])
def test_the_report_names_the_policy_it_applied(priority) -> None:
    requirements, items = _pool("A", "B", priority=priority)
    report = build_shadow_report(items, requirements, limit=5)

    assert report.applied and report.priority is priority
    assert report.policy_version == "shadow-v1"
