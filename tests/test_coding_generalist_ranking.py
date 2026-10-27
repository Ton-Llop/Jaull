from __future__ import annotations

import pytest

from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.recommendation import AssessmentLevel
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.recommendation.engine_v2 import rank_execution_plans, ranking_criteria, ranking_tie_key
from tests._workflow_fixtures import GIB, size_driven_estimator
from tests.test_recommendation_engine_v2 import (
    _evaluated_gguf,
    _ready_llama_context,
    _requirements,
)


@pytest.mark.parametrize("priority", [
    RecommendationPriority.QUALITY, RecommendationPriority.BALANCED,
])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_capable_generalist_can_compete_with_explicit_coders(priority, reverse):
    generalist = _evaluated_gguf(
        "org/General-27B-Instruct-GGUF", priority=priority,
        tags=["conversational", "instruct", "gguf"],
        quantizations=("Q4_K_M",), base_bytes=20 * GIB,
    )
    coder = _evaluated_gguf(
        "org/Coder-8B-Instruct-GGUF", priority=priority,
        quantizations=("Q8_0",), base_bytes=10 * GIB,
    )
    candidates = [generalist, coder]
    if reverse:
        candidates.reverse()
    ranked = rank_execution_plans(
        candidates, _requirements(priority), context=_ready_llama_context(),
        estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
    )
    assert ranked[0].evaluated.repo_id == generalist.repo_id
    assert ranked[0].plan.memory_prediction is not None
    assert ranked[0].plan.memory_prediction.assessment.status is CompatibilityStatus.COMPATIBLE
    # Ranking eligibility does not turn a heuristic into confirmed coding evidence.
    assert ranked[0].assessment.suitability is AssessmentLevel.ADEQUATE
    assert ranked[1].assessment.suitability is AssessmentLevel.STRONG
    suitability = [next(c for c in ranking_criteria(item, priority) if c.axis == "suitability")
                   for item in ranked]
    assert suitability[0].rank == suitability[1].rank
    assert [c.rank for c in ranking_criteria(ranked[0], priority)] < [
        c.rank for c in ranking_criteria(ranked[1], priority)
    ]


@pytest.mark.parametrize("tags,languages", [
    (["text-generation"], ["en"]),
    (["instruct", "translation"], ["en"]),
    (["instruct"], ["ja"]),
])
def test_ineligible_or_language_mismatched_profiles_do_not_get_the_coding_tie(tags, languages):
    generalist = _evaluated_gguf(
        "org/General-27B-GGUF", tags=tags, languages=languages,
        quantizations=("Q4_K_M",), base_bytes=20 * GIB,
    )
    coder = _evaluated_gguf("org/Coder-8B-GGUF", quantizations=("Q8_0",), base_bytes=10 * GIB)
    ranked = rank_execution_plans(
        [generalist, coder], _requirements(RecommendationPriority.QUALITY),
        context=_ready_llama_context(), estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
    )
    assert ranked[0].evaluated.repo_id == coder.repo_id


def test_speed_and_other_use_cases_keep_their_suitability_order():
    from dataclasses import replace

    generalist = _evaluated_gguf("org/General-27B-Instruct-GGUF", tags=["instruct"])
    coder = _evaluated_gguf("org/Coder-8B-GGUF")
    ranked = rank_execution_plans([generalist, coder], _requirements())
    inferred, explicit = sorted(ranked, key=lambda item: item.assessment.suitability.value)[:2]
    assert inferred.assessment.suitability is AssessmentLevel.ADEQUATE
    assert explicit.assessment.suitability is AssessmentLevel.STRONG
    assert ranking_tie_key(explicit, RecommendationPriority.SPEED) < ranking_tie_key(
        inferred, RecommendationPriority.SPEED,
    )
    assert ranking_tie_key(replace(explicit, use_case=UseCase.GENERAL_CHAT),
                           RecommendationPriority.QUALITY) < ranking_tie_key(
        replace(inferred, use_case=UseCase.GENERAL_CHAT), RecommendationPriority.QUALITY,
    )


def test_coding_reports_explain_the_shared_rank_without_upgrading_the_assessment():
    from jaull.application.recommendation.service import recommend
    from jaull.reporting.recommendation import report_to_dict, report_to_markdown
    from jaull.workflow.state import RecommendationWorkflowState
    from tests._workflow_fixtures import hardware

    req = _requirements(RecommendationPriority.QUALITY)
    candidates = [
        _evaluated_gguf("org/General-27B-Instruct-GGUF", tags=["instruct"]),
        _evaluated_gguf("org/Coder-8B-GGUF"),
    ]
    recommendations = recommend(candidates, req, hardware=hardware())
    state = RecommendationWorkflowState(requirements=req, recommendations=recommendations)
    payload = report_to_dict(state)
    generalist = next(r for r in payload["recommendations"] if "General" in r["repo_id"])
    assert "Strong and adequate coding suitability share one eligible-task rank" in (
        generalist["ranking"]["note"]
    )
    assert {c["axis"]: c["value"] for c in generalist["ranking"]["criteria"]}[
        "suitability"
    ] == "adequate"
    assert "Strong and adequate coding suitability share one eligible-task rank" in (
        report_to_markdown(state)
    )
