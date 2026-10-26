from __future__ import annotations

from dataclasses import replace

import pytest

from jaull.application.recommendation.service import recommend
from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.recommendation import AssessmentLevel
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.selection import weak_plans_last
from jaull.recommendation.shadow import apply_active_policy, build_shadow_report
from jaull.reporting.recommendation import report_to_dict, report_to_markdown
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import GIB, hardware
from tests.test_measured_policy_activation import _measured
from tests.test_recommendation_engine_v2 import _evaluated_gguf, _requirements
from tests.test_shadow_policies import _pool


@pytest.mark.parametrize(
    "status",
    [
        CompatibilityStatus.UNKNOWN,
        CompatibilityStatus.INSUFFICIENT,
        CompatibilityStatus.OFFLOADING_REQUIRED,
    ],
)
def test_capability_cannot_move_an_unconfirmed_or_degraded_plan_ahead_of_a_runnable_one(status):
    req = _requirements(RecommendationPriority.BALANCED)
    small = _evaluated_gguf("org/Coder-135M-GGUF", total_bytes=GIB)
    big = _evaluated_gguf("org/Coder-7B-GGUF", status=status)
    recs = recommend([small, big], req, hardware=hardware())
    assert recs[0].repo_id == small.repo_id


def test_viable_capable_plan_is_preferred_and_selection_is_traceable_everywhere():
    req = _requirements(RecommendationPriority.BALANCED)
    small = _evaluated_gguf("org/Coder-135M-GGUF", total_bytes=GIB)
    big = _evaluated_gguf("org/Coder-7B-GGUF", license_value=None)
    base = []
    recs = recommend(
        [small, big],
        req,
        hardware=hardware(),
        on_ranked_plans=lambda rows: base.extend(rows),
    )
    assert recs[0].repo_id == big.repo_id
    assert base[0].evaluated.repo_id == small.repo_id
    state = RecommendationWorkflowState(
        hardware=hardware(),
        requirements=req,
        recommendations=recs,
        ranked_plans=tuple(base),
    )
    before = state.model_dump_json()
    payload = report_to_dict(state)
    ranking = payload["recommendations"][0]["ranking"]
    selection = ranking["selection_policy"]
    assert selection["base_pool_position"] == 2
    assert selection["selected_pool_position"] == 1
    assert selection["stage"] == "before_measured_policy"
    assert ranking["balanced_policy"]["base_pool_position"] == 1
    assert ranking["balanced_policy"]["balanced_pool_position"] == 1
    markdown = report_to_markdown(state)
    assert markdown.index("**Balanced selection**") < markdown.index("**Measured-balanced policy**")
    assert "Selection order: pool #2 -> #1" in markdown
    assert "selection-adjusted slot" in markdown
    shadow = build_shadow_report(base, req, limit=len(recs))
    assert list(shadow.shadow_top5) == [rec.plan.plan_id for rec in recs]
    assert state.model_dump_json() == before


def test_unknown_capability_does_not_overtake_weak_capability():
    _, (a, b) = _pool("A", "B", priority=RecommendationPriority.BALANCED)
    weak = replace(
        a, assessment=a.assessment.model_copy(update={"capability": AssessmentLevel.WEAK})
    )
    unknown = replace(
        b, assessment=b.assessment.model_copy(update={"capability": AssessmentLevel.UNKNOWN})
    )
    assert weak_plans_last([weak, unknown], RecommendationPriority.BALANCED) == [weak, unknown]


def test_measured_evidence_can_override_the_metadata_selection_preference():
    req, (a, b) = _pool("A", "B", priority=RecommendationPriority.BALANCED)
    weak = replace(
        a, assessment=a.assessment.model_copy(update={"capability": AssessmentLevel.WEAK})
    )
    weak = _measured(weak, quality=0.9, speed=100)
    strong = _measured(b, quality=0.3, speed=50)
    assert weak_plans_last([weak, strong], req.priority) == [strong, weak]
    assert apply_active_policy([weak, strong], req) == [weak, strong]
    assert build_shadow_report([weak, strong], req, limit=2).shadow_order == (
        weak.plan.plan_id,
        strong.plan.plan_id,
    )
