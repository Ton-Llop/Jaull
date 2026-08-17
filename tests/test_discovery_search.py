"""Query building, the search client mapping and preliminary filtering.

Every test is offline: ``HfApi`` is replaced by a stub whose ``list_models``
returns canned objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from huggingface_hub.errors import HfHubHTTPError

from jaull.discovery import query_builder
from jaull.discovery.candidate_filter import (
    deduplicate,
    filter_candidates,
    parameter_count_hint,
    shortlist,
)
from jaull.discovery.search_client import (
    HfSearchClient,
    candidate_from_model_info,
)
from jaull.domain.candidates import SearchQuery
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.policies import TEXT_GENERATION_PIPELINE
from jaull.domain.requirements import CommercialUse, UseCase
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.workflow import policies
from jaull.workflow.requirements import build_requirements
from tests._workflow_fixtures import answers, candidate, hardware


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------
def _requirements(use_case: UseCase = UseCase.CODING) -> Any:
    return build_requirements(answers(use_case=use_case), hardware())


@pytest.mark.parametrize(
    ("use_case", "expected_phrase"),
    [
        (UseCase.GENERAL_CHAT, "instruct"),
        (UseCase.CODING, "coder instruct"),
        (UseCase.DOCUMENT_QA, "instruct long context"),
        (UseCase.SUMMARIZATION_EXTRACTION, "summarization instruct"),
        (UseCase.REASONING, "reasoning instruct"),
        (UseCase.BATCH_PROCESSING, "fast instruct"),
        (UseCase.WRITING_TRANSLATION, "multilingual instruct"),
    ],
)
def test_each_use_case_produces_its_own_phrases(
    use_case: UseCase, expected_phrase: str
) -> None:
    queries = query_builder.build_queries(_requirements(use_case))
    assert any(q.search == expected_phrase for q in queries)
    assert all(q.pipeline_tag == TEXT_GENERATION_PIPELINE for q in queries)


def test_queries_are_not_a_single_literal_string() -> None:
    """Several angles, not one clever query."""
    queries = query_builder.build_queries(_requirements())
    assert len({q.search for q in queries}) > 1


def test_format_queries_are_generated_for_preferred_formats() -> None:
    queries = query_builder.build_queries(_requirements())
    tags = {tag for q in queries for tag in q.filter_tags}
    assert "gguf" in tags


def test_non_english_languages_get_their_own_query() -> None:
    queries = query_builder.build_queries(_requirements())
    # The fixture asks for Spanish + English; only 'es' needs a tagged query.
    assert any("es" in q.filter_tags for q in queries)
    assert not any("en" in q.filter_tags for q in queries)


def test_every_query_respects_the_per_query_limit() -> None:
    queries = query_builder.build_queries(_requirements())
    assert all(q.limit == policies.SEARCH_RESULTS_PER_QUERY for q in queries)


def test_queries_are_deduplicated() -> None:
    queries = query_builder.build_queries(_requirements())
    keys = [q.cache_key() for q in queries]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Search client mapping
# ---------------------------------------------------------------------------
@dataclass
class _Info:
    id: str
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = "text-generation"
    library_name: str | None = "transformers"
    downloads: int | None = 500
    likes: int | None = 10
    private: bool = False
    gated: bool | str = False
    card_data: dict[str, Any] = field(default_factory=dict)
    sha: str | None = None
    last_modified: datetime | str | None = None


@dataclass
class _StubApi:
    results: list[_Info] = field(default_factory=list)
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def list_models(self, **kwargs: Any) -> list[_Info]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


def _query(label: str = "q") -> SearchQuery:
    return SearchQuery(label=label, search="instruct", pipeline_tag="text-generation")


def test_normal_response_is_mapped_to_candidates() -> None:
    api = _StubApi(
        results=[
            _Info(
                id="org/model",
                card_data={"license": "apache-2.0", "language": ["es", "en"]},
                tags=["text-generation"],
            )
        ]
    )
    client = HfSearchClient(api=api)  # type: ignore[arg-type]
    results = client.search(_query())
    assert len(results) == 1
    assert results[0].repo_id == "org/model"
    assert results[0].license == "apache-2.0"
    assert results[0].languages == ["es", "en"]
    assert results[0].source_queries == ["q"]


def test_gated_repositories_are_excluded_server_side() -> None:
    api = _StubApi(results=[])
    HfSearchClient(api=api).search(_query())  # type: ignore[arg-type]
    assert api.calls[0]["gated"] is False


def test_model_without_card_data_gets_low_confidence() -> None:
    info = _Info(id="org/bare", card_data={})
    result = candidate_from_model_info(info, "q")  # type: ignore[arg-type]
    assert result.metadata_confidence is EstimationConfidence.LOW
    assert result.license is None


def test_model_without_license_still_becomes_a_candidate() -> None:
    info = _Info(id="org/x", card_data={"language": "es"})
    result = candidate_from_model_info(info, "q")  # type: ignore[arg-type]
    assert result.license is None
    assert result.languages == ["es"]


def test_license_given_as_a_list_takes_the_first_entry() -> None:
    info = _Info(id="org/x", card_data={"license": ["mit", "apache-2.0"]})
    assert candidate_from_model_info(info, "q").license == "mit"  # type: ignore[arg-type]


def test_base_model_dict_shape_is_understood() -> None:
    info = _Info(id="org/x-GGUF", card_data={"base_model": {"finetune": "org/x"}})
    assert candidate_from_model_info(info, "q").base_model_repo_id == "org/x"  # type: ignore[arg-type]


def test_revision_and_last_modified_are_preserved_from_search_metadata() -> None:
    modified = datetime(2026, 1, 2, tzinfo=UTC)
    info = _Info(id="org/x", sha="abc123", last_modified=modified)

    result = candidate_from_model_info(info, "q")  # type: ignore[arg-type]

    assert result.revision_hint == "abc123"
    assert result.last_modified == modified


@dataclass
class _Response:
    """Just enough of a requests.Response for HfHubHTTPError to construct."""

    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    request: Any = None


def test_rate_limit_is_reported_as_unavailable() -> None:
    error = HfHubHTTPError("429", response=_Response(429))  # type: ignore[arg-type]
    client = HfSearchClient(api=_StubApi(error=error))  # type: ignore[arg-type]
    with pytest.raises(HuggingFaceUnavailableError, match="rate limit"):
        client.search(_query())


def test_other_http_errors_are_reported_as_unavailable() -> None:
    error = HfHubHTTPError("500", response=_Response(500))  # type: ignore[arg-type]
    client = HfSearchClient(api=_StubApi(error=error))  # type: ignore[arg-type]
    with pytest.raises(HuggingFaceUnavailableError, match="HTTP 500"):
        client.search(_query())


def test_transport_failure_is_reported_as_unavailable() -> None:
    client = HfSearchClient(api=_StubApi(error=OSError("timed out")))  # type: ignore[arg-type]
    with pytest.raises(HuggingFaceUnavailableError):
        client.search(_query())


# ---------------------------------------------------------------------------
# Deduplication and filtering
# ---------------------------------------------------------------------------
def test_deduplication_merges_query_labels() -> None:
    first = candidate(repo_id="org/a", queries=["q1"])
    second = candidate(repo_id="org/a", queries=["q2"], downloads=50_000)
    merged = deduplicate([first, second])
    assert len(merged) == 1
    assert merged[0].source_queries == ["q1", "q2"]
    assert merged[0].downloads == 50_000


def test_deduplication_preserves_first_seen_order() -> None:
    merged = deduplicate(
        [candidate(repo_id="org/b"), candidate(repo_id="org/a"), candidate(repo_id="org/b")]
    )
    assert [c.repo_id for c in merged] == ["org/b", "org/a"]


def test_private_and_gated_are_rejected() -> None:
    req = _requirements()
    outcome = filter_candidates(
        [
            candidate(repo_id="org/private", private=True),
            candidate(repo_id="org/gated", gated=True),
        ],
        req,
    )
    assert outcome.kept == []
    assert {repo for repo, _ in outcome.rejected} == {"org/private", "org/gated"}


def test_wrong_pipeline_is_rejected() -> None:
    outcome = filter_candidates(
        [candidate(repo_id="org/embed", pipeline_tag="feature-extraction")],
        _requirements(),
    )
    assert outcome.kept == []
    assert "text generation" in outcome.rejected[0][1]


def test_multimodal_models_are_rejected() -> None:
    outcome = filter_candidates(
        [candidate(repo_id="org/vlm", tags=["image-text-to-text"])], _requirements()
    )
    assert outcome.kept == []
    assert "multimodal" in outcome.rejected[0][1]


def test_adapter_without_base_model_is_rejected() -> None:
    outcome = filter_candidates(
        [candidate(repo_id="org/lora", tags=["peft", "lora"])], _requirements()
    )
    assert outcome.kept == []
    assert "Adapter" in outcome.rejected[0][1]


def test_adapter_with_base_model_survives() -> None:
    outcome = filter_candidates(
        [candidate(repo_id="org/lora", tags=["peft"], base_model="org/base")],
        _requirements(),
    )
    assert len(outcome.kept) == 1


def test_restricted_license_is_rejected_when_commercial_use_is_required() -> None:
    outcome = filter_candidates(
        [candidate(repo_id="org/nc", license_value="cc-by-nc-4.0")], _requirements()
    )
    assert outcome.kept == []
    assert "commercial" in outcome.rejected[0][1]


def test_restricted_license_survives_when_commercial_use_is_not_required() -> None:
    req = build_requirements(answers(commercial=CommercialUse.NO), hardware())
    outcome = filter_candidates(
        [candidate(repo_id="org/nc", license_value="cc-by-nc-4.0")], req
    )
    assert len(outcome.kept) == 1


def test_incomplete_metadata_is_penalised_not_rejected() -> None:
    """A thin model card is the norm on the Hub; it must not remove a candidate."""
    outcome = filter_candidates(
        [
            candidate(
                repo_id="org/thin",
                license_value=None,
                languages=[],
                pipeline_tag=None,
                tags=["text-generation"],
            )
        ],
        _requirements(),
    )
    assert len(outcome.kept) == 1
    kept = outcome.kept[0]
    assert kept.penalties
    assert kept.metadata_confidence is not EstimationConfidence.HIGH


def test_shortlist_spends_the_inspection_budget_on_plausible_repositories() -> None:
    """A 19-download experiment must not displace a widely used release."""
    req = _requirements()
    junk = candidate(repo_id="someone/gpt2_experiment", downloads=19, likes=0)
    real = candidate(repo_id="Qwen/Qwen2.5-Coder-7B-Instruct", downloads=2_000_000)
    picked = shortlist([junk, real], req, limit=1)
    assert [c.repo_id for c in picked] == ["Qwen/Qwen2.5-Coder-7B-Instruct"]


def test_shortlist_prefers_the_format_this_machine_can_run() -> None:
    req = build_requirements(answers(), hardware(vram_gib=6))
    assert req.preferred_formats[0] == "gguf"
    gguf = candidate(repo_id="org/model-GGUF", downloads=1000, tags=["gguf"])
    safet = candidate(repo_id="org/model", downloads=1000, tags=["safetensors"])
    picked = shortlist([safet, gguf], req, limit=1)
    assert picked[0].repo_id == "org/model-GGUF"


def test_shortlist_rewards_candidates_found_by_several_queries() -> None:
    req = _requirements()
    once = candidate(repo_id="org/a", downloads=1000, queries=["q1"])
    thrice = candidate(repo_id="org/b", downloads=1000, queries=["q1", "q2", "q3"])
    picked = shortlist([once, thrice], req, limit=1)
    assert picked[0].repo_id == "org/b"


@pytest.mark.parametrize(
    ("repo_id", "expected"),
    [
        ("Qwen/Qwen2.5-Coder-7B-Instruct", 7.0),
        ("org/Model-1.5B-Instruct", 1.5),
        ("org/Qwen3-Coder-30B-A3B-Instruct", 30.0),
        ("org/plain-model", None),
        ("org/model-v2", None),
    ],
)
def test_parameter_count_hint(repo_id: str, expected: float | None) -> None:
    assert parameter_count_hint(repo_id) == expected


def test_shortlist_deprioritises_models_that_cannot_possibly_fit() -> None:
    """A 6 GiB GPU should not spend its inspection budget on 32B models."""
    req = _requirements()
    huge = candidate(repo_id="Qwen/Qwen2.5-Coder-32B-Instruct", downloads=2_000_000)
    small = candidate(repo_id="Qwen/Qwen2.5-Coder-3B-Instruct", downloads=50_000)
    picked = shortlist([huge, small], req, limit=1, budget_bytes=6 * 1024**3)
    assert picked[0].repo_id == "Qwen/Qwen2.5-Coder-3B-Instruct"


def test_shortlist_without_a_budget_ignores_the_size_hint() -> None:
    req = _requirements()
    huge = candidate(repo_id="org/Model-32B", downloads=2_000_000)
    small = candidate(repo_id="org/Model-3B", downloads=50_000)
    picked = shortlist([huge, small], req, limit=1)
    assert picked[0].repo_id == "org/Model-32B"


def test_size_hint_never_reaches_a_reported_number() -> None:
    """The heuristic orders the queue; it must not become a memory estimate."""
    req = _requirements()
    picked = shortlist(
        [candidate(repo_id="org/Model-7B")], req, limit=1, budget_bytes=24 * 1024**3
    )
    # The candidate is unchanged: no size was written onto it.
    assert picked[0].repo_id == "org/Model-7B"
    assert not hasattr(picked[0], "parameter_count")


def test_shortlist_is_deterministic_and_respects_the_limit() -> None:
    req = _requirements()
    pool = [candidate(repo_id=f"org/m{i}", downloads=1000) for i in range(20)]
    first = [c.repo_id for c in shortlist(pool, req, limit=5)]
    second = [c.repo_id for c in shortlist(list(reversed(pool)), req, limit=5)]
    assert first == second
    assert len(first) == 5


def test_valid_gguf_and_transformers_repositories_survive() -> None:
    outcome = filter_candidates(
        [
            candidate(repo_id="org/gguf", tags=["text-generation", "gguf"]),
            candidate(repo_id="org/tf", tags=["text-generation", "safetensors"]),
        ],
        _requirements(),
    )
    assert {c.repo_id for c in outcome.kept} == {"org/gguf", "org/tf"}
