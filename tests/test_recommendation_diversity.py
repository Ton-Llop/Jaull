from __future__ import annotations

from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution_plans import (
    ModelIdentity,
    ModelIdentityEvidence,
    ModelIdentityEvidenceKind,
    model_identity_key,
)
from jaull.domain.inference import WeightPrecision
from jaull.domain.recommendation import AssessmentLevel, RecommendationPosition
from jaull.domain.requirements import RecommendationPriority
from jaull.domain.runtime import RuntimeName
from jaull.execution_plans import execution_plan_for_recommendation
from jaull.recommendation.diversity import diversify_ranked_plans
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    RankedPlan,
    rank_execution_plans,
)
from jaull.workflow import policies, ranking
from tests._workflow_fixtures import GIB, hardware, size_driven_estimator
from tests.test_recommendation_engine_v2 import (
    _benchmark_record,
    _evaluated_gguf,
    _evaluated_transformers,
    _requirements,
)


def test_max_recommendations_is_five() -> None:
    assert policies.MAX_RECOMMENDATIONS == 5


def test_eight_valid_candidates_return_five_recommendations() -> None:
    recs = ranking.recommend(
        [_evaluated_gguf(repo_id=f"org/Model-{index}-7B-GGUF") for index in range(8)],
        _requirements(),
        hardware=hardware(),
    )

    assert len(recs) == 5


def test_four_valid_candidates_return_four_recommendations() -> None:
    recs = ranking.recommend(
        [_evaluated_gguf(repo_id=f"org/Model-{index}-7B-GGUF") for index in range(4)],
        _requirements(),
        hardware=hardware(),
    )

    assert len(recs) == 4


def test_rejected_plans_do_not_fill_top_five() -> None:
    valid = [_evaluated_gguf(repo_id=f"org/Valid-{index}-7B-GGUF") for index in range(4)]
    rejected = [
        _evaluated_gguf(
            repo_id=f"org/Rejected-{index}-70B-GGUF",
            status=CompatibilityStatus.INSUFFICIENT,
        )
        for index in range(4)
    ]

    recs = ranking.recommend(valid + rejected, _requirements(), hardware=hardware())

    assert len(recs) == 4
    assert all(rec.plan_assessment is not None for rec in recs)
    assert all(not rec.plan_assessment.rejected for rec in recs if rec.plan_assessment)


def test_same_model_quantizations_consume_one_slot_with_alternatives() -> None:
    evaluated = _evaluated_gguf()
    ranked = rank_execution_plans(
        [evaluated],
        _requirements(RecommendationPriority.QUALITY),
        estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
    )

    diversified = diversify_ranked_plans(ranked, limit=5)

    assert len(diversified) == 1
    assert diversified[0].primary.plan.artifact.quantization == "Q6_K"
    assert {item.plan.artifact.quantization for item in diversified[0].alternatives} >= {
        "Q4_K_M",
        "Q5_K_M",
    }


def test_base_repo_gguf_repo_and_precision_variants_share_one_slot() -> None:
    base = _evaluated_transformers(repo_id="Qwen/Qwen2.5-1.5B-Instruct")
    int8 = _with_precision(base, WeightPrecision.INT8)
    gguf = _evaluated_gguf(repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")

    recs = ranking.recommend([base, gguf, int8], _requirements(), hardware=hardware())

    assert len(recs) == 1
    plans = [recs[0].plan, *recs[0].alternative_plans]
    labels = {
        plan.artifact.quantization or plan.artifact.precision
        for plan in plans
        if plan is not None
    }
    assert {"Q5_K_M", "float16", "int8"} <= labels


def test_model_identity_key_ignores_artifact_and_evidence_fields() -> None:
    gguf_identity = ModelIdentity(
        canonical_repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        family="qwen2.5",
        model_name="Qwen2.5-1.5B-Instruct-GGUF",
        parameter_count=1_500_000_000,
        variant="gguf",
        architecture="Qwen2ForCausalLM",
        confidence=EstimationConfidence.LOW,
        evidence=[
            ModelIdentityEvidence(
                kind=ModelIdentityEvidenceKind.NAME_HEURISTIC,
                source="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                value="Qwen/Qwen2.5-1.5B-Instruct",
                confidence=EstimationConfidence.LOW,
            )
        ],
    )
    transformers_identity = ModelIdentity(
        canonical_repo_id="qwen/qwen2.5-1.5b-instruct",
        family="Qwen",
        model_name="Qwen2.5 1.5B Instruct",
        parameter_count=1_500_000_000,
        variant="instruct",
    )

    assert gguf_identity != transformers_identity
    assert model_identity_key(gguf_identity) == model_identity_key(
        transformers_identity
    )


def test_gguf_suffix_is_removed_from_model_identity() -> None:
    ranked = rank_execution_plans(
        [_evaluated_gguf(repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")],
        _requirements(),
    )

    identity = ranked[0].plan.model_identity

    assert identity.canonical_repo_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert identity.model_name == "Qwen2.5-1.5B-Instruct"


def test_primary_execution_plan_comes_from_v2_priority_policy() -> None:
    evaluated = _evaluated_gguf()

    quality = diversify_ranked_plans(
        rank_execution_plans(
            [evaluated],
            _requirements(RecommendationPriority.QUALITY),
            estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
        ),
        limit=5,
    )
    speed = diversify_ranked_plans(
        rank_execution_plans(
            [evaluated],
            _requirements(RecommendationPriority.SPEED),
            estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
        ),
        limit=5,
    )

    assert quality[0].primary.plan.artifact.quantization == "Q6_K"
    assert speed[0].primary.plan.artifact.quantization == "Q4_K_M"


def test_recommendation_model_preserves_alternative_plans() -> None:
    recs = ranking.recommend(
        [_evaluated_gguf()],
        _requirements(RecommendationPriority.QUALITY),
        hardware=hardware(),
    )

    assert recs[0].plan is not None
    assert recs[0].alternative_plans == []


def test_number_one_does_not_change_for_diversity() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("gemma", "Gemma", 2_000_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=3)

    assert diversified[0].primary is ranked[0]


def test_diversity_can_reorder_comparable_later_slots() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("gemma", "Gemma", 2_000_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=3)

    assert [item.primary.plan.model_identity.family for item in diversified] == [
        "qwen",
        "gemma",
        "qwen",
    ]


def test_diversity_does_not_promote_clearly_worse_candidate() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("gemma", "Gemma", 2_000_000_000),
    )
    worse_gemma = ranked[2].assessment.model_copy(
        update={"suitability": AssessmentLevel.WEAK}
    )
    ranked[2] = RankedPlan(
        evaluated=ranked[2].evaluated,
        plan=ranked[2].plan,
        assessment=worse_gemma,
    )

    diversified = diversify_ranked_plans(ranked, limit=3)

    assert diversified[1].primary.plan.model_identity.family == "qwen"


def test_multiple_models_from_same_family_are_allowed() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("qwen", "Qwen-C", 4_000_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=5)

    assert len(diversified) == 3
    assert {item.primary.plan.model_identity.family for item in diversified} == {"qwen"}


def test_no_family_is_required_for_any_slot() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 4_000_000_000),
        ("qwen", "Qwen-C", 7_000_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=5)

    assert len(diversified) == 3


def test_unknown_family_still_works() -> None:
    ranked = _ranked_with_identities(
        (None, "Unknown-A", 1_500_000_000),
        (None, "Unknown-B", 2_000_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=5)

    assert len(diversified) == 2
    assert all(item.primary.plan.model_identity.family is None for item in diversified)


def test_same_family_size_and_runtime_is_more_redundant() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("liquidai", "LiquidAI", 1_200_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=3)

    assert diversified[1].primary.plan.model_identity.family == "liquidai"


def test_different_runtime_can_help_but_does_not_guarantee_promotion() -> None:
    llama = _ranked_with_identities(("qwen", "Qwen-A", 1_500_000_000))[0]
    qwen_llama = _ranked_with_identities(("qwen", "Qwen-B", 1_600_000_000))[0]
    raw_transformers = _ranked_transformers("gemma", "Gemma", 2_000_000_000)
    transformers = RankedPlan(
        evaluated=raw_transformers.evaluated,
        plan=raw_transformers.plan,
        assessment=qwen_llama.assessment,
    )

    diversified = diversify_ranked_plans([llama, qwen_llama, transformers], limit=3)
    assert diversified[1].primary.plan.runtime_family is RuntimeName.TRANSFORMERS

    worse_transformers = RankedPlan(
        evaluated=transformers.evaluated,
        plan=transformers.plan,
        assessment=transformers.assessment.model_copy(
            update={"executability": AssessmentLevel.WEAK}
        ),
    )
    conservative = diversify_ranked_plans(
        [llama, qwen_llama, worse_transformers],
        limit=3,
    )
    assert conservative[1].primary.plan.runtime_family is RuntimeName.LLAMA_CPP


def test_different_parameter_tier_can_add_diversity() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("qwen", "Qwen-C", 7_000_000_000),
    )

    diversified = diversify_ranked_plans(ranked, limit=3)

    assert diversified[1].primary.plan.model_identity.parameter_count == 7_000_000_000


def test_local_benchmark_effect_survives_through_assessment() -> None:
    evaluated = _evaluated_gguf()
    ranked = rank_execution_plans([evaluated], _requirements())
    benchmark = _benchmark_record(
        ranked[0].plan.artifact.to_model_artifact(),
        ranked[0].plan.runtime,
        machine=hardware(),
        tps=88.0,
    )
    measured = rank_execution_plans(
        [evaluated],
        _requirements(),
        context=PlanRankingContext(hardware=hardware(), benchmark_records=[benchmark]),
    )

    diversified = diversify_ranked_plans(measured, limit=5)

    assert diversified[0].primary.assessment.local_benchmark_id is not None
    assert diversified[0].position is RecommendationPosition.BEST_OVERALL


def test_diversity_does_not_modify_plan_assessment() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("gemma", "Gemma", 2_000_000_000),
    )
    before = [item.assessment for item in ranked]

    diversify_ranked_plans(ranked, limit=5)

    assert [item.assessment for item in ranked] == before


def test_diversity_is_deterministic() -> None:
    ranked = _ranked_with_identities(
        ("qwen", "Qwen-A", 1_500_000_000),
        ("qwen", "Qwen-B", 1_600_000_000),
        ("gemma", "Gemma", 2_000_000_000),
        ("liquidai", "LiquidAI", 1_200_000_000),
    )

    first = diversify_ranked_plans(ranked, limit=5)
    second = diversify_ranked_plans(ranked, limit=5)

    assert [item.primary.plan.plan_id for item in first] == [
        item.primary.plan.plan_id for item in second
    ]


def test_tui_accepts_five_recommendations() -> None:
    recs = ranking.recommend(
        [_evaluated_gguf(repo_id=f"org/Model-{index}-7B-GGUF") for index in range(5)],
        _requirements(),
        hardware=hardware(),
    )

    assert [rec.rank for rec in recs] == [1, 2, 3, 4, 5]


def test_execution_workflows_use_primary_plan() -> None:
    rec = ranking.recommend(
        [_evaluated_gguf()],
        _requirements(),
        hardware=hardware(),
    )[0]

    assert rec.plan is not None
    assert execution_plan_for_recommendation(rec) is rec.plan


def _ranked_with_identities(
    *items: tuple[str | None, str, int | None],
) -> list[RankedPlan]:
    return [
        _with_identity(
            rank_execution_plans(
                [_evaluated_gguf(repo_id=f"org/{model_name}-GGUF")],
                _requirements(),
            )[0],
            family=family,
            model_name=model_name,
            params=params,
        )
        for family, model_name, params in items
    ]


def _ranked_transformers(
    family: str,
    model_name: str,
    params: int,
) -> RankedPlan:
    ranked = rank_execution_plans(
        [_evaluated_transformers(repo_id=f"org/{model_name}")],
        _requirements(),
    )[0]
    return _with_identity(ranked, family=family, model_name=model_name, params=params)


def _with_identity(
    ranked: RankedPlan,
    *,
    family: str | None,
    model_name: str,
    params: int | None,
) -> RankedPlan:
    identity = ranked.plan.model_identity.model_copy(
        update={
            "canonical_repo_id": f"org/{model_name}",
            "family": family,
            "model_name": model_name,
            "parameter_count": params,
        }
    )
    artifact = ranked.plan.artifact.model_copy(update={"model_identity": identity})
    plan = ranked.plan.model_copy(
        update={
            "model_identity": identity,
            "artifact": artifact,
            "plan_id": f"{ranked.plan.plan_id}-{model_name.lower()}",
        }
    )
    return RankedPlan(
        evaluated=ranked.evaluated,
        plan=plan,
        assessment=ranked.assessment,
    )


def _with_precision(evaluated, precision: WeightPrecision):
    config = evaluated.selected_configuration
    assert config is not None
    estimate = evaluated.memory_estimate
    assert estimate is not None
    updated_config = config.model_copy(update={"precision": precision, "quantization": None})
    return evaluated.model_copy(
        update={
            "selected_configuration": updated_config,
            "memory_estimate": estimate.model_copy(
                update={"inference_configuration": updated_config}
            ),
        }
    )
