from dataclasses import replace

import pytest

from jaull.application.recommendation.service import recommend
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    generate_execution_plans,
    rank_execution_plans,
)
from tests._workflow_fixtures import hardware
from tests.test_recommendation_engine_v2 import (
    _benchmark_record,
    _evaluated_gguf,
    _experiment_record,
    _requirements,
    _selection,
)


@pytest.mark.parametrize("priority", [
    RecommendationPriority.BALANCED, RecommendationPriority.QUALITY,
])
@pytest.mark.parametrize("kind", ["benchmark", "validation"])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_stored_run_does_not_upgrade_confidence_or_reorder_the_base_pool(priority, kind, reverse):
    requirements = _requirements(priority)
    context = PlanRankingContext(hardware=hardware(), backend_selection=_selection())
    candidates = [_evaluated_gguf(f"org/Coder-{size}B-Instruct-GGUF") for size in (7, 14)]
    candidates = [item.model_copy(update={
        "memory_estimate": item.memory_estimate.model_copy(update={
            "assessment": item.memory_estimate.assessment.model_copy(update={
                "confidence": EstimationConfidence.LOW,
            }),
        }),
    }) for item in candidates]
    plan = generate_execution_plans(candidates[0], requirements, context=context)[0]
    if kind == "benchmark":
        record = _benchmark_record(
            plan.artifact.to_model_artifact(), plan.runtime, machine=hardware(), tps=9.3,
            peak_ram_bytes=1234,
        )
        updated = replace(context, benchmark_records=[record])
        identifier = record.identity.benchmark_id
    else:
        experiment = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
        updated = replace(context, experiment_records=[experiment])
        identifier = experiment.identity.experiment_id
    if reverse:
        candidates.reverse()
    before = rank_execution_plans(candidates, requirements, context=context)
    after = rank_execution_plans(candidates, requirements, context=updated)
    assert [(item.plan.plan_id, item.assessment.confidence) for item in after] == [
        (item.plan.plan_id, item.assessment.confidence) for item in before
    ]
    observed = next(item.assessment for item in after if item.plan.plan_id == plan.plan_id)
    assert observed.confidence is EstimationConfidence.LOW
    # Retain the actual observations and IDs, without treating a tg64/v1 run as eligible speed.
    assert observed.measured_memory_bytes is not None
    assert observed.measured_generation_tokens_per_second is None
    assert identifier in {observed.local_benchmark_id, observed.local_experiment_id}
    assert any(e.observed and e.provenance for e in observed.evidence)
    # Exercise measured policy and diversity too, not just the base ranking key.
    shown_before = recommend(candidates, requirements, hardware=hardware(), plan_context=context)
    shown_after = recommend(candidates, requirements, hardware=hardware(), plan_context=updated)
    assert [(item.repo_id, item.plan.artifact.quantization) for item in shown_after] == [
        (item.repo_id, item.plan.artifact.quantization) for item in shown_before
    ]
