from __future__ import annotations

import pytest

from jaull.domain import policies
from jaull.domain.requirements import (
    CommercialUse,
    ConcurrencyLevel,
    DocumentScale,
    RecommendationPriority,
    UseCase,
    UserAnswers,
    UserRequirements,
    WorkloadMode,
    WorkloadProfile,
)
from jaull.workflow.requirements import (
    build_requirements,
    normalize_language,
    normalize_languages,
)
from tests._workflow_fixtures import answers, hardware


@pytest.mark.parametrize(
    "use_case", [case for case in UseCase if case is not UseCase.BATCH_PROCESSING]
)
def test_every_use_case_maps_through(use_case: UseCase) -> None:
    req = build_requirements(answers(use_case=use_case), hardware())
    assert req.use_case is use_case
    assert req.pipeline_tag == policies.TEXT_GENERATION_PIPELINE


def test_legacy_batch_is_a_general_chat_task_with_batch_mode() -> None:
    old = answers().model_dump(mode="json")
    old["use_case"] = "batch_processing"
    old.pop("workload_mode", None)
    loaded = UserAnswers.model_validate(old)
    assert loaded.use_case is UseCase.GENERAL_CHAT
    assert loaded.workload_mode is WorkloadMode.BATCH
    req = build_requirements(loaded, hardware())
    assert req.use_case is UseCase.GENERAL_CHAT
    assert req.workload_mode is WorkloadMode.BATCH
    old_req = req.model_dump(mode="json")
    old_req["use_case"] = "batch_processing"
    old_req.pop("workload_mode", None)
    restored = UserRequirements.model_validate(old_req)
    assert restored.use_case is UseCase.GENERAL_CHAT
    assert restored.workload_mode is WorkloadMode.BATCH


def test_coding_task_match_is_independent_of_workload_mode() -> None:
    interactive = build_requirements(answers(use_case=UseCase.CODING), hardware())
    batch = interactive.model_copy(update={"workload_mode": WorkloadMode.BATCH})
    from jaull.discovery.query_builder import build_queries
    from jaull.recommendation.scoring import task_match
    from tests._workflow_fixtures import candidate

    model = candidate(repo_id="org/Coder-7B", tags=["coder", "instruct"])
    assert task_match(model, interactive) == task_match(model, batch)
    assert build_queries(interactive) == build_queries(batch)
    assert batch.workload_profile.mode is WorkloadMode.BATCH


def test_workload_profile_optional_slos_and_serialization() -> None:
    minimal = WorkloadProfile(context_length=4096)
    assert minimal.concurrent_users == 1
    assert minimal.min_generation_tps is None
    assert minimal.max_ttft_ms is None
    assert WorkloadProfile.model_validate_json(minimal.model_dump_json()) == minimal

    req = build_requirements(answers(use_case=UseCase.CODING), hardware())
    enriched = req.model_copy(
        update={
            "expected_input_tokens": 512,
            "expected_output_tokens": 128,
            "min_generation_tps": 20.0,
            "max_ttft_ms": 1000.0,
        }
    )
    profile = enriched.workload_profile
    assert profile.context_length == enriched.desired_context
    assert profile.concurrent_users == enriched.concurrent_users
    assert profile.min_generation_tps == 20.0
    assert profile.max_ttft_ms == 1000.0
    assert UserRequirements.model_validate_json(enriched.model_dump_json()) == enriched


@pytest.mark.parametrize("priority", list(RecommendationPriority))
def test_every_priority_maps_through(priority: RecommendationPriority) -> None:
    req = build_requirements(answers(priority=priority), hardware())
    assert req.priority is priority


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Spanish", "es"),
        ("español", "es"),
        ("Catalan", "ca"),
        ("English", "en"),
        ("Portuguese", "pt"),
        ("fr", "fr"),
        ("Spanish (Spain)", "es"),
        ("  english  ", "en"),
        ("klingon", None),
        ("", None),
    ],
)
def test_language_normalisation(value: str, expected: str | None) -> None:
    assert normalize_language(value) == expected


def test_language_normalisation_deduplicates_and_reports_rejects() -> None:
    codes, rejected = normalize_languages(["Spanish", "es", "English", "zzzzz"])
    assert codes == ["es", "en"]
    assert rejected == ["zzzzz"]


def test_no_languages_falls_back_to_english_with_an_assumption() -> None:
    req = build_requirements(answers(languages=[]), hardware())
    assert req.languages == ["en"]
    assert any("English" in note for note in req.assumptions)


@pytest.mark.parametrize(
    ("level", "expected_users"),
    [
        (ConcurrencyLevel.SINGLE, 1),
        (ConcurrencyLevel.SMALL, 3),
        (ConcurrencyLevel.MEDIUM, 8),
        (ConcurrencyLevel.LARGE, 15),
    ],
)
def test_concurrency_buckets(level: ConcurrencyLevel, expected_users: int) -> None:
    req = build_requirements(answers(concurrency=level), hardware())
    assert req.concurrent_users == expected_users
    # The original range is preserved so reports never imply false precision.
    assert req.concurrency_range == policies.CONCURRENCY_BUCKETS[level.value][1]


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (DocumentScale.SHORT, 4096),
        (DocumentScale.MEDIUM, 8192),
        (DocumentScale.LONG, 16384),
        (DocumentScale.COLLECTION, 32768),
    ],
)
def test_document_context_mapping(scale: DocumentScale, expected: int) -> None:
    req = build_requirements(
        answers(use_case=UseCase.DOCUMENT_QA, document_scale=scale), hardware()
    )
    assert req.desired_context == expected
    assert any("not the size of a document collection" in a for a in req.assumptions)


def test_document_question_is_ignored_for_other_use_cases() -> None:
    """Answering question 5 for a non-document use case must not change context."""
    req = build_requirements(
        answers(use_case=UseCase.CODING, document_scale=DocumentScale.COLLECTION),
        hardware(),
    )
    assert req.desired_context == policies.DEFAULT_CONTEXT_TOKENS


def test_document_use_case_without_scale_uses_the_default() -> None:
    req = build_requirements(
        answers(use_case=UseCase.DOCUMENT_QA, document_scale=None), hardware()
    )
    assert req.desired_context == policies.DEFAULT_CONTEXT_TOKENS


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (CommercialUse.YES, True),
        (CommercialUse.NO, False),
        (CommercialUse.NO_PREFERENCE, False),
        (CommercialUse.NOT_SURE, None),
    ],
)
def test_commercial_use_mapping(answer: CommercialUse, expected: bool | None) -> None:
    req = build_requirements(answers(commercial=answer), hardware())
    assert req.commercial_use_required is expected


def test_not_sure_records_that_licenses_will_not_exclude_models() -> None:
    req = build_requirements(answers(commercial=CommercialUse.NOT_SURE), hardware())
    assert any("not used to exclude" in note for note in req.assumptions)


def test_low_vram_prefers_gguf() -> None:
    req = build_requirements(answers(), hardware(vram_gib=6))
    assert req.preferred_formats[0] == "gguf"


def test_large_vram_puts_safetensors_first() -> None:
    req = build_requirements(answers(), hardware(vram_gib=24))
    assert req.preferred_formats[0] == "safetensors"


def test_no_gpu_prefers_gguf_and_says_why() -> None:
    req = build_requirements(answers(), hardware(vram_gib=None))
    assert req.preferred_formats[0] == "gguf"
    assert any("No CUDA-capable NVIDIA GPU" in note for note in req.assumptions)


def test_concurrency_above_one_records_a_no_throughput_caveat() -> None:
    req = build_requirements(answers(concurrency=ConcurrencyLevel.MEDIUM), hardware())
    assert any("does not model real throughput" in a for a in req.assumptions)
