from __future__ import annotations

import pytest

from local_ai_check.workflow import policies
from local_ai_check.workflow.models import (
    CommercialUse,
    ConcurrencyLevel,
    DocumentScale,
    RecommendationPriority,
    UseCase,
)
from local_ai_check.workflow.requirements import (
    build_requirements,
    normalize_language,
    normalize_languages,
)
from tests._workflow_fixtures import answers, hardware


@pytest.mark.parametrize("use_case", list(UseCase))
def test_every_use_case_maps_through(use_case: UseCase) -> None:
    req = build_requirements(answers(use_case=use_case), hardware())
    assert req.use_case is use_case
    assert req.pipeline_tag == policies.TEXT_GENERATION_PIPELINE


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
        (ConcurrencyLevel.MEDIUM, 10),
        (ConcurrencyLevel.LARGE, 25),
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
    assert any("No NVIDIA GPU" in note for note in req.assumptions)


def test_concurrency_above_one_records_a_no_throughput_caveat() -> None:
    req = build_requirements(answers(concurrency=ConcurrencyLevel.MEDIUM), hardware())
    assert any("does not model real throughput" in a for a in req.assumptions)
