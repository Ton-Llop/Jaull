"""Synthetic offline quality references; not measured benchmark results."""

import json
from copy import deepcopy

import pytest

from jaull.advisor.service import AdvisorService
from jaull.domain.recommendation import ExternalEvaluationEvidence, PlanAssessment
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.evaluation.quality_records import digest
from jaull.evaluation.quality_storage import QualityEvidenceStore
from jaull.recommendation import capability_catalog
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    assess_plan,
    generate_execution_plans,
    rank_execution_plans,
    ranking_criteria,
)
from jaull.recommendation.quality import assess_quality
from tests.test_quality_eval_records import synthetic_full_record, synthetic_limited_record
from tests.test_quality_evidence_store import _services
from tests.test_recommendation_engine_v2 import (
    _capability_evaluation,
    _evaluated_gguf,
    _requirements,
    _with_confirmed_lineage,
)


@pytest.mark.parametrize("priority", list(RecommendationPriority))
def test_quality_references_are_exact_historical_and_do_not_change_ranking(priority) -> None:
    evaluated = _evaluated_gguf(priority=priority)
    requirements = _requirements(priority)
    record = synthetic_limited_record()
    # Set published artifact metadata, not model-family inheritance.
    analysis = evaluated.analysis
    assert analysis is not None
    for variant in analysis.classification.gguf_variants:
        for index, file in enumerate(variant.files):
            variant.files[index] = file.model_copy(update={
                "sha256": record["identity"]["artifact_sha256"],
            })
    baseline = rank_execution_plans([evaluated], requirements)
    with_records = rank_execution_plans([evaluated], requirements, context=PlanRankingContext(
        quality_records=[record, {"status": "failed"}],
    ))
    assert [item.plan.plan_id for item in baseline] == [item.plan.plan_id for item in with_records]
    for before, after in zip(baseline, with_records, strict=True):
        assert before.assessment.model_dump(exclude={"quality"}) == after.assessment.model_dump(
            exclude={"quality"},
        )
        assert ranking_criteria(before, requirements) == ranking_criteria(after, requirements)
        quality = after.assessment.quality
        assert quality.use_case == requirements.use_case
        assert quality.metadata_prior == before.assessment.capability
        assert "invalid" in quality.limitations[-1]
        reference, = quality.local_references
        assert reference.samples_used == 100
        assert reference.record_sha256 == digest(record)
        assert reference.classification == "limited"
        assert "audited generation contract" in reference.blockers[-1]
        assert quality.applicability == "reference_only"
        assert quality.requested_profile == "humaneval-python-en-v1"
        restored = PlanAssessment.model_validate_json(after.assessment.model_dump_json())
        assert restored.quality == quality

    other = deepcopy(record)
    other["identity"]["artifact_sha256"] = "0" * 64
    other["identity_sha256"] = digest(other["identity"])
    unrelated = rank_execution_plans([evaluated], requirements, context=PlanRankingContext(
        quality_records=[other],
    ))
    assert all(not item.assessment.quality.local_references for item in unrelated)


def test_different_full_runs_remain_references_without_cherry_picking_or_conflict_claim() -> None:
    evaluated = _evaluated_gguf(priority=RecommendationPriority.QUALITY)
    plan = generate_execution_plans(evaluated, _requirements(RecommendationPriority.QUALITY))[0]
    record = synthetic_full_record()
    plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": record["identity"]["artifact_sha256"],
    })})
    repeated = deepcopy(record)
    task = record["identity"]["suite"]["name"]
    repeated["result"]["samples"][task][0].update(acc=0, acc_norm=0)
    repeated["result"]["results"][task].update({"acc,none": 0., "acc_norm,none": 0.})
    requirements = _requirements(use_case=UseCase.CODING)
    first = assess_quality(plan, requirements, PlanAssessment(plan_id="x").capability,
                           [record, repeated])
    second = assess_quality(plan, requirements, first.metadata_prior, [repeated, record])
    assert first == second
    assert {ref.metrics["acc"] for ref in first.local_references} == {0., 1.}
    assert len({ref.record_sha256 for ref in first.local_references}) == 2
    assert all(ref.blockers for ref in first.local_references)
    assert first.applicability == "reference_only"
    unknown_artifact = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": None,
    })})
    assert not assess_quality(
        unknown_artifact, requirements, first.metadata_prior, [record],
    ).local_references
    assert PlanAssessment.model_validate({"plan_id": "old"}).quality.local_references == ()


@pytest.mark.parametrize("use_case,profile", [
    (UseCase.GENERAL_CHAT, "ifeval-instructions-en-v1"),
    (UseCase.CODING, "humaneval-python-en-v1"),
    (UseCase.REASONING, None),
    (UseCase.DOCUMENT_QA, None),
    (UseCase.SUMMARIZATION_EXTRACTION, None),
    (UseCase.WRITING_TRANSLATION, None),
])
def test_quality_target_depends_on_request_not_available_benchmark(use_case, profile) -> None:
    evaluated = _evaluated_gguf()
    requirements = _requirements(use_case=use_case)
    plan = generate_execution_plans(evaluated, requirements)[0]
    record = synthetic_full_record()
    plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": record["identity"]["artifact_sha256"],
    })})
    empty = assess_quality(plan, requirements, PlanAssessment(plan_id="x").capability, [])
    historical = assess_quality(plan, requirements, empty.metadata_prior, [record])
    assert empty.requested_profile == historical.requested_profile == profile
    assert historical.languages == ("en",)
    assert empty.applicability == "absent"
    assert historical.applicability == "reference_only"
    reference, = historical.local_references
    assert reference.classification == "full"
    assert reference.blockers
    if profile == "ifeval-instructions-en-v1":
        # A complete HellaSwag record is a reference; only the IFEval chat suite applies.
        assert "one complete jaull-ifeval-chat-v1 run" in reference.blockers[-1]
        assert historical.applicable is None
    elif profile:
        assert "generation contract" in reference.blockers[-1]
    else:
        assert "requested task and languages" in reference.blockers[-1]


@pytest.mark.parametrize("languages", [[], ["es"], ["ca"], ["en", "es"]])
def test_unknown_or_uncovered_language_does_not_default_to_english(languages) -> None:
    requirements = _requirements().model_copy(update={"languages": languages})
    plan = generate_execution_plans(_evaluated_gguf(), requirements)[0]
    result = assess_quality(plan, requirements, PlanAssessment(plan_id="x").capability, [])
    assert result.requested_profile is None
    assert result.languages == tuple(languages)
    assert "No validated quality profile" in result.limitations[-1]


@pytest.mark.parametrize("renamed", ["ifeval-instructions-en-v1", "jaull-ifeval-chat-v1"])
def test_suite_name_alone_cannot_promote_scoring_to_instruction_quality(renamed) -> None:
    # The second name is the suite that *can* apply; a loglikelihood record wearing it
    # must still be a reference, and must not break looking for its primary metric.
    record = synthetic_full_record()
    old = record["identity"]["suite"]["name"]
    record["identity"]["suite"]["name"] = renamed
    for field in ("results", "configs", "n-samples", "samples"):
        record["result"][field][renamed] = record["result"][field].pop(old)
    record["result"]["configs"][renamed]["task"] = renamed
    record["identity_sha256"] = digest(record["identity"])
    requirements = _requirements(use_case=UseCase.GENERAL_CHAT)
    plan = generate_execution_plans(_evaluated_gguf(), requirements)[0]
    plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": record["identity"]["artifact_sha256"],
    })})
    before = deepcopy(record)
    result = assess_quality(plan, requirements, PlanAssessment(plan_id="x").capability, [record])
    reference, = result.local_references  # Valid scoring record, not a rejected fixture.
    assert reference.suite == renamed
    assert result.applicability == "reference_only"
    assert result.applicable is None
    assert "one complete jaull-ifeval-chat-v1 run" in reference.blockers[-1]
    assert record == before


def test_published_reference_does_not_override_local_or_authorize_quality_ranking() -> None:
    requirements = _requirements()
    plan = generate_execution_plans(_evaluated_gguf(), requirements)[0]
    record = synthetic_full_record()
    plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": record["identity"]["artifact_sha256"],
    })})
    # Already-attributed reference; the engine's subject filter is tested separately.
    published = ExternalEvaluationEvidence(
        benchmark="synthetic-coding", task="coding", value=1., metric="pass@1",
        source="synthetic publisher", verified=True,
    )
    prior = PlanAssessment(plan_id="x").capability
    only = assess_quality(plan, requirements, prior, [], published=[published])
    both = assess_quality(plan, requirements, prior, [record], published=[published])
    assert only.applicability == both.applicability == "reference_only"
    assert only.local_references == ()
    local_only = assess_quality(plan, requirements, prior, [record])
    assert both.local_references == local_only.local_references
    assert "representation and protocol" in both.limitations[-1]
    invalid = assess_quality(plan, requirements, prior, [{"status": "failed"}])
    assert invalid.applicability == "absent"
    assert "invalid" in invalid.limitations[-1]


@pytest.mark.parametrize("matching_subject", [False, True])
def test_quality_reference_state_uses_only_published_evidence_attributed_by_engine(
    matching_subject,
) -> None:
    requirements = _requirements()
    evaluated = _evaluated_gguf()
    plan = _with_confirmed_lineage(generate_execution_plans(evaluated, requirements)[0])
    subject = plan.model_identity.canonical_repo_id if matching_subject else "other/Unrelated-7B"
    published = _capability_evaluation(subject)
    result = assess_plan(evaluated, plan, requirements, context=PlanRankingContext(
        external_evaluations=[published],
    ))
    assert result.quality.applicability == ("reference_only" if matching_subject else "absent")
    assert result.external_evaluations == ([published] if matching_subject else [])
    assert result.quality.local_references == ()
    assert result.quality.requested_profile == "humaneval-python-en-v1"


def test_new_search_context_reads_new_local_records_and_catalog_data(tmp_path, monkeypatch) -> None:
    catalog = json.loads(capability_catalog.default_catalog_path().read_text())
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(capability_catalog, "default_catalog_path", lambda: path)
    store = QualityEvidenceStore(tmp_path / "quality")
    advisor = AdvisorService(services=_services(), quality_store=store)
    # Avoid runtime probes: this test checks data freshness, not readiness.
    monkeypatch.setattr(AdvisorService, "_plan_backend_selection", lambda *args: None)
    monkeypatch.setattr(AdvisorService, "_plan_runtime_readiness", lambda *args, **kwargs: None)
    monkeypatch.setattr(AdvisorService, "_stored_benchmark_records", lambda *args: [])
    monkeypatch.setattr(AdvisorService, "_stored_experiment_records", lambda *args: [])
    from tests._workflow_fixtures import hardware

    before = advisor._plan_ranking_context(hardware())
    assert before.quality_records == []
    catalog["evaluations"][0]["value"] = 0.0
    path.write_text(json.dumps(catalog), encoding="utf-8")
    store.save(synthetic_full_record())
    after = advisor._plan_ranking_context(hardware())
    assert len(after.quality_records) == 1
    assert after.external_evaluations[0].value == 0.0
    assert before.external_evaluations[0].value != 0.0
    path.write_text("not JSON", encoding="utf-8")
    invalid = advisor._plan_ranking_context(hardware())
    assert invalid.external_evaluations == ()
    assert len(invalid.quality_records) == 1
    assert advisor.capability_catalog().diagnostic
    def unavailable():
        raise OSError("Quality directory unavailable")

    monkeypatch.setattr(store, "records", unavailable)
    assert advisor._plan_ranking_context(hardware()).quality_records == []


def test_records_are_validated_once_per_ranking_not_once_per_plan(monkeypatch) -> None:
    from jaull.recommendation import quality as quality_module

    calls = 0
    real = quality_module.describe_record

    def counted(record):
        nonlocal calls
        calls += 1
        return real(record)

    monkeypatch.setattr(quality_module, "describe_record", counted)
    records = [synthetic_limited_record(), synthetic_full_record()]
    candidates = [_evaluated_gguf("org/A-7B-GGUF"), _evaluated_gguf("org/B-7B-GGUF")]

    ranked = rank_execution_plans(candidates, _requirements(), context=PlanRankingContext(
        quality_records=records,
    ))

    assert len(ranked) >= 2
    assert calls == len(records)
