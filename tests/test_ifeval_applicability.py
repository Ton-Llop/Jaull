"""Complete IFEval records as applicable shadow evidence. Synthetic records, not measurements."""

import copy
import hashlib
from pathlib import Path

import pytest
from pilot.quality_eval import ifeval
from pilot.quality_eval.evaluate import EVALUATOR_COMMIT, SERVER_SHA256

from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.evaluation.quality_records import describe_record, digest
from jaull.recommendation.engine_v2 import PlanRankingContext, rank_execution_plans
from jaull.recommendation.quality import IFEVAL_SOURCES_SHA256, IFEVAL_SUITE_SHA256
from jaull.recommendation.shadow import build_shadow_report
from tests.test_quality_generation_records import FULL, SMOKE, record
from tests.test_recommendation_engine_v2 import _evaluated_gguf, _requirements

Q = RecommendationPriority.QUALITY


def _full_record(*, stronger=False) -> dict:
    """Synthetic outcomes with audited metadata; not actual dataset bytes or scores."""
    value = record()
    identity = value["identity"]
    ids = list(range(ifeval.DATASET_SIZE))
    originals = value["result"]["samples"][FULL]
    samples = [copy.deepcopy(originals[i % 2]) for i in ids]
    for i, sample in enumerate(samples):
        sample["doc_id"] = i
        if stronger:
            for mode in ("strict", "loose"):
                sample[f"prompt_level_{mode}_acc"] = True
                sample[f"inst_level_{mode}_acc"] = [True] * len(
                    sample["doc"]["instruction_id_list"],
                )
    value["result"]["samples"][FULL] = samples
    identity["samples"] = [
        {key: sample[key] for key in identity["samples"][0]} for sample in samples
    ]
    suite_sha = hashlib.sha256(
        Path("pilot/quality_eval/suite_ifeval.yaml").read_bytes(),
    ).hexdigest()
    identity["suite"]["sha256"] = suite_sha
    identity["dataset"] = {
        "repo": ifeval.DATASET_REPO, "revision": ifeval.DATASET_REVISION,
        "sha256": ifeval.DATASET_SHA256, "sample_ids": ids,
        "total_samples": ifeval.DATASET_SIZE,
    }
    identity["evaluator"].update({
        "suite_sha256": suite_sha, "sample_ids": ids, "evaluator_commit": EVALUATOR_COMMIT,
        "source_sha256": dict(ifeval.SOURCE_SHA256),
    })
    identity["runtime"]["server_sha256"] = SERVER_SHA256
    value["result"]["n-samples"][FULL] = {"original": len(ids), "effective": len(ids)}
    for key in value["result"]["results"][FULL]:
        name = key.removesuffix(",none")
        flat = [outcome for sample in samples for outcome in (
            sample[name] if name.startswith("inst_") else [sample[name]]
        )]
        value["result"]["results"][FULL][key] = sum(flat) / len(flat)
    value["outcome"]["responses"] = len(ids)
    if stronger:
        identity["artifact_sha256"] = "9" * 64
        identity["evaluator"]["chat_template_sha256"] = "8" * 64
    value["identity_sha256"] = digest(identity)
    return value


def _gguf(repo: str, sha: str):
    evaluated = _evaluated_gguf(repo, priority=Q)
    assert evaluated.analysis is not None
    for variant in evaluated.analysis.classification.gguf_variants:
        for index, file in enumerate(variant.files):
            variant.files[index] = file.model_copy(update={"sha256": sha})
    return evaluated


def _ranked(records, *, use_case=UseCase.GENERAL_CHAT, languages=None, priority=Q):
    requirements = _requirements(priority, use_case=use_case)
    if languages is not None:
        requirements = requirements.model_copy(update={"languages": languages})
    candidates = [_gguf("org/A-7B-GGUF", "a" * 64), _gguf("org/B-7B-GGUF", "9" * 64)]
    ranked = rank_execution_plans(candidates, requirements,
                                  context=PlanRankingContext(quality_records=records))
    return requirements, ranked


def _repos(report, ranked):
    names = {item.plan.plan_id: item.evaluated.repo_id for item in ranked}
    return [names[plan_id] for plan_id in report.shadow_order]


def test_complete_ifeval_records_order_the_shadow_and_never_the_active_ranking() -> None:
    weaker, stronger = _full_record(), _full_record(stronger=True)
    requirements, ranked = _ranked([weaker, stronger])
    _, unmeasured = _ranked([])

    # The active order is exactly what it is without any record.
    assert [i.plan.plan_id for i in ranked] == [i.plan.plan_id for i in unmeasured]
    applied = {i.evaluated.repo_id: i.assessment.quality.applicable for i in ranked}
    assert applied["org/A-7B-GGUF"].value == 271 / 541
    assert applied["org/B-7B-GGUF"].value == 1.0
    assert applied["org/A-7B-GGUF"].cohort == applied["org/B-7B-GGUF"].cohort

    report = build_shadow_report(ranked, requirements, limit=5)
    assert _repos(report, ranked)[:2] == ["org/B-7B-GGUF", "org/A-7B-GGUF"]

    # Swap the measurements and the proposal follows the evidence, not the names.
    swapped_a = _full_record(stronger=True)
    swapped_a["identity"]["artifact_sha256"] = "a" * 64
    swapped_b = _full_record()
    swapped_b["identity"]["artifact_sha256"] = "9" * 64
    for item in (swapped_a, swapped_b):
        item["identity_sha256"] = digest(item["identity"])
    requirements, ranked = _ranked([swapped_a, swapped_b])
    report = build_shadow_report(ranked, requirements, limit=5)
    assert _repos(report, ranked)[:2] == ["org/A-7B-GGUF", "org/B-7B-GGUF"]


@pytest.mark.parametrize("kwargs", [
    {"use_case": UseCase.CODING},           # IFEval says nothing about code.
    {"languages": ["es"]},                  # An English benchmark says nothing about Spanish.
    {"languages": ["en", "es"]},
])
def test_ifeval_applies_only_to_english_chat(kwargs) -> None:
    _, ranked = _ranked([_full_record(), _full_record(stronger=True)], **kwargs)
    assert all(item.assessment.quality.applicable is None for item in ranked)


def test_a_smoke_never_applies() -> None:
    smoke = record(SMOKE, "plumbing")
    _, ranked = _ranked([smoke])
    assert all(item.assessment.quality.applicable is None for item in ranked)


def test_repeated_runs_with_the_same_answers_count_as_one() -> None:
    first = _full_record()
    second = copy.deepcopy(first)
    second["result"]["date"] = 1  # Same identity and answers; only lm-eval's run date moved.
    _, ranked = _ranked([second, first])
    measured = next(i for i in ranked if i.evaluated.repo_id == "org/A-7B-GGUF")
    applied = measured.assessment.quality.applicable
    assert applied is not None
    # Deterministic choice: the same record stands for the repeats in any store order.
    assert applied.record_sha256 == min(digest(first), digest(second))
    assert any("gave identical answers; they count as one" in item
               for item in measured.assessment.quality.limitations)


def test_repeated_runs_that_disagree_abstain_instead_of_picking() -> None:
    first = _full_record()
    second = copy.deepcopy(first)
    sample = second["result"]["samples"][FULL][0]
    sample["resps"] = [["A different reply."]]
    sample["filtered_resps"] = ["A different reply."]
    _, ranked = _ranked([first, second])
    measured = next(i for i in ranked if i.evaluated.repo_id == "org/A-7B-GGUF")
    assert measured.assessment.quality.applicable is None
    assert any("different results; no rule selects one" in item
               for item in measured.assessment.quality.limitations)


def test_two_prompts_with_the_full_suite_name_are_only_a_reference() -> None:
    value = record()
    assert describe_record(value).reusable  # Valid for its declared two-sample identity.
    _, ranked = _ranked([value])
    assessed = next(i.assessment.quality for i in ranked if i.evaluated.repo_id == "org/A-7B-GGUF")
    assert assessed.applicability == "reference_only"
    assert assessed.applicable is None
    assert assessed.local_references


@pytest.mark.parametrize("section,key,changed", [
    ("suite", "sha256", "0" * 64),
    ("dataset", "repo", "other/Instructions"),
    ("dataset", "revision", "0" * 40),
    ("dataset", "sha256", "0" * 64),
    ("evaluator", "suite_sha256", "0" * 64),
    ("evaluator", "evaluator_commit", "0" * 40),
    ("evaluator", "source_sha256", {"unknown.checker": "0" * 64}),
    ("evaluator", "context", 2048),
    ("evaluator", "langdetect_seed", 1),
    ("runtime", "server_sha256", "0" * 64),
    ("runtime", "fingerprint", "other-build"),
])
def test_unaudited_profile_drift_stays_reference_only(section, key, changed) -> None:
    value = _full_record()
    value["identity"][section][key] = changed
    if (key == "sha256" and section == "suite") or key == "suite_sha256":
        value["identity"]["suite"]["sha256"] = changed
        value["identity"]["evaluator"]["suite_sha256"] = changed
    value["identity_sha256"] = digest(value["identity"])
    assert describe_record(value).reusable
    requirements, ranked = _ranked([value])
    assessed = next(i.assessment.quality for i in ranked if i.evaluated.repo_id == "org/A-7B-GGUF")
    assert assessed.applicability == "reference_only"
    assert assessed.applicable is None
    assert assessed.local_references
    assert any("audited pins" in reason for reason in assessed.local_references[0].blockers)
    report = build_shadow_report(ranked, requirements, limit=5)
    assert report.shadow_order == report.base_order


def test_consumer_audit_pins_match_the_producer_sources_and_suite() -> None:
    assert digest(ifeval.SOURCE_SHA256) == IFEVAL_SOURCES_SHA256
    assert hashlib.sha256(Path("pilot/quality_eval/suite_ifeval.yaml").read_bytes()).hexdigest() \
        == IFEVAL_SUITE_SHA256


def test_quality_priority_now_orders_the_active_list_by_measured_quality() -> None:
    """Step 6 activation: Quality follows comparable measurements; nothing else moves."""
    from jaull.recommendation.shadow import apply_quality_policy

    weaker, stronger = _full_record(), _full_record(stronger=True)
    requirements, ranked = _ranked([weaker, stronger])
    base = [i.evaluated.repo_id for i in ranked]
    active = [i.evaluated.repo_id for i in apply_quality_policy(ranked, requirements)]
    assert base[:2] == ["org/A-7B-GGUF", "org/B-7B-GGUF"]
    assert active[:2] == ["org/B-7B-GGUF", "org/A-7B-GGUF"]  # B measured 100 %, A 50 %.

    # Without applicable evidence the active order is exactly the base order.
    _, unmeasured = _ranked([])
    assert apply_quality_policy(unmeasured, requirements) == list(unmeasured)
    # Speed and Balanced stay in shadow until speed evidence applies.
    for priority in (RecommendationPriority.SPEED, RecommendationPriority.BALANCED):
        other = requirements.model_copy(update={"priority": priority})
        assert apply_quality_policy(ranked, other) == list(ranked)


def test_recommend_shows_the_quality_order_and_keeps_the_base_pool_for_recompare() -> None:
    from jaull.application.recommendation.service import recommend
    from tests._workflow_fixtures import hardware

    records = [_full_record(), _full_record(stronger=True)]
    requirements = _requirements(Q, use_case=UseCase.GENERAL_CHAT)
    candidates = [_gguf("org/A-7B-GGUF", "a" * 64), _gguf("org/B-7B-GGUF", "9" * 64)]
    pools: list = []
    shown = recommend(
        candidates, requirements, hardware=hardware(),
        plan_context=PlanRankingContext(hardware=hardware(), quality_records=records),
        on_ranked_plans=pools.append,
    )
    assert [rec.repo_id for rec in shown][:2] == ["org/B-7B-GGUF", "org/A-7B-GGUF"]
    # The saved pool is the base order: Recompare and the fallback start from it.
    assert [item.evaluated.repo_id for item in pools[0]][:2] == [
        "org/A-7B-GGUF", "org/B-7B-GGUF",
    ]
