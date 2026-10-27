"""Query building, the search client mapping and preliminary filtering.

Every test is offline: ``HfApi`` is replaced by a stub whose ``list_models``
returns canned objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from jaull.application.recommendation import policies
from jaull.application.requirements import build_requirements
from jaull.discovery import candidate_filter, query_builder
from jaull.discovery.candidate_filter import (
    coarse_placement_hint,
    collapse_quantized_reuploads,
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
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import EstimationConfidence, HardwareFitMode
from jaull.domain.policies import TEXT_GENERATION_PIPELINE
from jaull.domain.requirements import CommercialUse, UseCase
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.recommendation.scoring import targets_another_task
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


@pytest.mark.parametrize("use_case", [UseCase.DOCUMENT_QA, UseCase.REASONING])
@pytest.mark.parametrize("max_billions", [None, 2.0, 30.0])
def test_documents_and_reasoning_include_general_instruct_discovery(
    use_case: UseCase, max_billions: float | None,
) -> None:
    requirements = _requirements(use_case)
    queries = query_builder.build_queries(requirements, max_billions=max_billions)
    broad = next(q for q in queries if q.label == f"{use_case.value}:instruct")
    assert broad.search == "instruct"
    assert broad.pipeline_tag == TEXT_GENERATION_PIPELINE
    assert broad.max_billions is None
    assert broad.filter_tags == ()

    # Every filtered angle uses a name that ordinary Instruct repositories can
    # actually match, not the task's multi-word description.
    for suffix in ("gguf", "safetensors", "lang-es", "trending"):
        matches = [q for q in queries if q.label == f"{use_case.value}:{suffix}"]
        assert len(matches) == 1
        assert matches[0].search == "instruct"
        assert matches[0].pipeline_tag == TEXT_GENERATION_PIPELINE
    # One named query per size band when the machine holds sizes beyond the small ones.
    bands = [
        q for q in queries
        if q.label.startswith(f"{use_case.value}:scale-") and q.search is not None
    ]
    assert len(bands) == (0 if max_billions is None else 2)
    for band in bands:
        assert band.search == "instruct"
        assert band.pipeline_tag == TEXT_GENERATION_PIPELINE

    # One additional page only; shared inspection and candidate budgets do not change.
    baseline = query_builder.build_queries(_requirements(), max_billions=max_billions)
    assert len(queries) == len(baseline) + 1
    assert all(q.limit == policies.SEARCH_RESULTS_PER_QUERY for q in queries)
    assert len({q.cache_key() for q in queries}) == len(queries)


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


def test_the_sizes_that_fit_beyond_the_small_models_are_searched_band_by_band() -> None:
    """The plain queries sort by downloads, so the very popular small models fill every page and
    the 7B-32B range this machine can hold is never listed. Each size bucket the machine can hold
    gets its own pair of queries, so the popular 9B models cannot crowd out the 14B-32B ones."""
    requirements = _requirements()
    queries = query_builder.build_queries(requirements, max_billions=30)
    scale = [q for q in queries if q.max_billions is not None]
    named = [q for q in scale if q.search is not None]
    open_ = [q for q in scale if q.search is None]

    bands = [(q.min_billions, q.max_billions) for q in named]
    assert bands == [(candidate_filter.LARGER_THAN_SMALL_BILLIONS, 14.0), (14.0, 30)]
    assert [(q.min_billions, q.max_billions) for q in open_] == bands
    for query in scale:
        assert query.filter_tags == (requirements.preferred_formats[0],)
    assert all(q.pipeline_tag == TEXT_GENERATION_PIPELINE for q in named)
    # Qwen3.8-27B's GGUF repository declares no pipeline and has neither "instruct" nor "coder" in
    # its name, so no named query can return it; the open ones are limited by size and format only.
    assert all(q.pipeline_tag is None for q in open_)


def test_a_machine_that_holds_very_large_models_gets_a_band_for_them() -> None:
    queries = query_builder.build_queries(_requirements(), max_billions=70)
    bands = [(q.min_billions, q.max_billions) for q in queries
             if q.max_billions is not None and q.search is not None]
    assert bands == [
        (candidate_filter.LARGER_THAN_SMALL_BILLIONS, 14.0), (14.0, 32.0), (32.0, 70),
    ]


@pytest.mark.parametrize("max_billions", [None, 0.0, -1.0])
def test_no_scale_query_without_a_positive_planning_ceiling(
    max_billions: float | None,
) -> None:
    queries = query_builder.build_queries(_requirements(), max_billions=max_billions)
    assert all(q.max_billions is None for q in queries)


def test_the_largest_fitting_size_follows_the_machine() -> None:
    req = _requirements()
    big = candidate_filter.largest_fitting_billions(hardware(vram_gib=24, ram_gib=16), req)
    small = candidate_filter.largest_fitting_billions(hardware(vram_gib=4, ram_gib=8), req)
    assert big is not None and small is not None
    assert 14 < big < 60  # a 24 GiB card holds a 14B model, and not a 70B one
    assert small < candidate_filter.LARGER_THAN_SMALL_BILLIONS * 2
    assert small < big


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


@pytest.mark.parametrize("error", [
    httpx.ConnectError("DNS unavailable"), httpx.ReadTimeout("Hub timed out"),
])
def test_transport_failure_uses_the_workflows_recoverable_search_error(error: Exception) -> None:
    client = HfSearchClient(api=_StubApi(error=error))  # type: ignore[arg-type]
    with pytest.raises(HuggingFaceUnavailableError, match="Unable to reach") as caught:
        client.search(_query())
    assert caught.value.__cause__ is error


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


def test_known_adapter_is_rejected_even_with_a_declared_base_model() -> None:
    adapter = candidate(repo_id="org/lora", base_model="org/base").model_copy(
        update={"repository_type": RepositoryType.ADAPTER}
    )

    outcome = filter_candidates([adapter], _requirements())

    assert outcome.kept == []
    assert "standalone model" in outcome.rejected[0][1]


def test_declared_adapter_relation_does_not_spend_an_inspection_slot() -> None:
    # Reproduces yusifnuri/*_code_generation in the 2026-10-07 code captures: the
    # Hub tagged them as adapters of a named base, the filter kept them because a
    # base existed, and inspection then rejected them - 2 of 12 deep-inspection
    # slots in every coding scenario, spent on candidates that could never run.
    tags = ["peft", "lora", "base_model:adapter:mistralai/Mistral-7B-v0.3",
            "base_model:mistralai/Mistral-7B-v0.3"]
    outcome = filter_candidates(
        [candidate(repo_id="org/coder-lora", tags=tags, base_model="mistralai/Mistral-7B-v0.3")],
        _requirements(),
    )
    assert outcome.kept == []
    assert "standalone model" in outcome.rejected[0][1]


def test_a_finetune_relation_is_not_mistaken_for_an_adapter() -> None:
    # A merged fine-tune names its base too, and may still carry a `lora` tag.
    tags = ["lora", "base_model:finetune:org/base", "base_model:org/base"]
    outcome = filter_candidates(
        [candidate(repo_id="org/merged", tags=tags, base_model="org/base")], _requirements(),
    )
    assert len(outcome.kept) == 1


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


def test_undeclared_languages_are_unknown_not_a_penalty() -> None:
    """A model card that does not list its languages says nothing about them: most cards omit the
    field. That is unknown, which the language fit, the requirement check and the explanation
    already report; it must not also lower the candidate's metadata confidence or its place in
    the inspection budget as if the card were doubtful."""
    complete = candidate(repo_id="org/complete", languages=["en"])
    no_languages = candidate(repo_id="org/no-languages", languages=[])
    no_pipeline = candidate(repo_id="org/no-pipeline", languages=["en"], pipeline_tag=None)
    nothing_declared = candidate(repo_id="org/nothing", languages=[], pipeline_tag=None)

    kept = {
        item.repo_id: item
        for item in filter_candidates(
            [complete, no_languages, no_pipeline, nothing_declared], _requirements(),
        ).kept
    }

    assert kept["org/no-languages"].penalties == []
    assert kept["org/no-languages"].metadata_confidence == kept["org/complete"].metadata_confidence
    # An absent pipeline tag leaves a visible warning but does not lower the confidence: it says
    # what the task is, which the suitability already weighs, not the format or the runtime that
    # the inspection confirms. A missing license is a different matter and still does.
    assert len(kept["org/no-pipeline"].penalties) == 1
    assert kept["org/no-pipeline"].metadata_confidence == kept["org/complete"].metadata_confidence
    assert len(kept["org/nothing"].penalties) == 1
    assert kept["org/nothing"].metadata_confidence == kept["org/complete"].metadata_confidence
    unlicensed = filter_candidates(
        [candidate(repo_id="org/unlicensed", license_value=None)], _requirements(),
    ).kept[0]
    assert unlicensed.metadata_confidence is not kept["org/complete"].metadata_confidence


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


def test_coarse_shortlist_marks_gpu_resident_candidates() -> None:
    req = _requirements()
    hint = coarse_placement_hint(
        candidate(repo_id="org/Assistant-3B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
        req,
        hardware(vram_gib=8, ram_gib=32),
    )
    assert hint.mode is HardwareFitMode.GPU_RESIDENT
    assert hint.gpu_weight_bytes > 0
    assert hint.ram_weight_bytes == 0


def test_coarse_shortlist_marks_gpu_offload_without_vram_ram_pooling() -> None:
    req = _requirements()
    hint = coarse_placement_hint(
        candidate(repo_id="org/Assistant-14B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
        req,
        hardware(vram_gib=8, ram_gib=32),
    )
    assert hint.mode is HardwareFitMode.GPU_OFFLOAD
    assert hint.gpu_weight_bytes > 0
    assert hint.ram_weight_bytes > 0
    assert hint.gpu_required_bytes is not None
    assert hint.gpu_required_bytes <= 8 * 1024**3


def test_coarse_shortlist_uses_cpu_ram_when_gpu_placement_is_not_viable() -> None:
    req = _requirements()
    hint = coarse_placement_hint(
        candidate(repo_id="org/Assistant-7B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
        req,
        hardware(vram_gib=1, ram_gib=32),
    )
    assert hint.mode is HardwareFitMode.CPU_RAM
    assert hint.gpu_weight_bytes == 0
    assert hint.ram_weight_bytes > 0


def test_coarse_shortlist_marks_too_large_candidates() -> None:
    req = _requirements()
    hint = coarse_placement_hint(
        candidate(repo_id="org/Assistant-70B-Q8_0-GGUF", tags=["gguf", "Q8_0"]),
        req,
        hardware(vram_gib=6, ram_gib=16),
    )
    assert hint.mode is HardwareFitMode.TOO_LARGE


def test_hardware_aware_shortlist_includes_offload_candidate_on_8gb_gpu() -> None:
    req = _requirements()
    pool = [
        candidate(repo_id=f"Qwen/Qwen2.5-{index}B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
        for index in (0.5, 1.5, 3, 7)
    ]
    pool.extend(
        [
            candidate(repo_id="Qwen/Qwen2.5-14B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
            candidate(repo_id="Huge/Huge-70B-Q8_0-GGUF", tags=["gguf", "Q8_0"]),
        ]
    )
    picked = shortlist(pool, req, limit=5, hardware=hardware(vram_gib=8, ram_gib=32))
    modes = {
        item.repo_id: coarse_placement_hint(item, req, hardware(vram_gib=8, ram_gib=32)).mode
        for item in picked
    }
    assert len(picked) == 5
    assert "Qwen/Qwen2.5-14B-Q4_K_M-GGUF" in modes
    assert modes["Qwen/Qwen2.5-14B-Q4_K_M-GGUF"] is HardwareFitMode.GPU_OFFLOAD
    assert "Huge/Huge-70B-Q8_0-GGUF" not in modes


def test_hardware_aware_shortlist_expands_scale_on_larger_gpu() -> None:
    req = _requirements()
    pool = [
        candidate(repo_id=f"org/Model-{size}B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
        for size in (1, 3, 7, 14, 32, 70)
    ]
    picked = shortlist(pool, req, limit=4, hardware=hardware(vram_gib=24, ram_gib=64))
    picked_sizes = {
        parameter_count_hint(item.repo_id)
        for item in picked
        if parameter_count_hint(item.repo_id) is not None
    }
    assert len(picked) == 4
    assert any(size is not None and size >= 14 for size in picked_sizes)


def test_hardware_aware_shortlist_handles_cpu_only() -> None:
    req = _requirements()
    pool = [
        candidate(repo_id="org/Small-3B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
        candidate(repo_id="org/Huge-70B-Q8_0-GGUF", tags=["gguf", "Q8_0"]),
    ]
    picked = shortlist(pool, req, limit=2, hardware=hardware(vram_gib=None, ram_gib=32))
    assert picked[0].repo_id == "org/Small-3B-Q4_K_M-GGUF"


def test_hardware_aware_shortlist_rejects_fake_offload_with_tiny_ram() -> None:
    req = _requirements()
    hint = coarse_placement_hint(
        candidate(repo_id="org/Assistant-14B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
        req,
        hardware(vram_gib=8, ram_gib=2),
    )
    assert hint.mode is not HardwareFitMode.GPU_OFFLOAD


def test_hardware_aware_shortlist_is_quantization_sensitive() -> None:
    req = _requirements()
    q4 = candidate(repo_id="org/Assistant-14B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
    q8 = candidate(repo_id="org/Assistant-14B-Q8_0-GGUF", tags=["gguf", "Q8_0"])
    profile = hardware(vram_gib=8, ram_gib=10)
    assert coarse_placement_hint(q4, req, profile).mode is HardwareFitMode.GPU_OFFLOAD
    assert coarse_placement_hint(q8, req, profile).mode is HardwareFitMode.TOO_LARGE


def test_hardware_aware_shortlist_uses_family_diversity_without_a_quota() -> None:
    req = _requirements()
    qwen = [
        candidate(repo_id=f"Qwen/Qwen2.5-{index}B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
        for index in (1, 2, 3, 4, 5, 6)
    ]
    others = [
        candidate(repo_id="LiquidAI/LFM2-3B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
        candidate(repo_id="Google/Gemma-3B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]),
    ]
    picked = shortlist(qwen + others, req, limit=5, hardware=hardware(vram_gib=8))
    families = {item.repo_id.split("/", maxsplit=1)[0] for item in picked}
    assert {"Qwen", "LiquidAI", "Google"} <= families
    assert sum(item.repo_id.startswith("Qwen/") for item in picked) > 1


def test_the_coarse_hint_only_gives_a_gpu_to_formats_that_can_run_on_it() -> None:
    """Regression found on a B60 box: nine of twelve inspections went to safetensors repos that
    were rated GPU resident on a machine whose only accelerator is Vulkan, where Jaull's
    Transformers loader cannot run; they were also sized as if 4-bit."""
    from tests._intel_fixtures import LINUX_OS, scan

    vulkan_only = scan(LINUX_OS).model_copy(update={"memory": hardware().memory})
    req = _requirements()

    gguf = coarse_placement_hint(
        candidate("org/Model-7B-Instruct-GGUF", tags=["gguf", "Q4_K_M"]), req, vulkan_only,
    )
    plain = coarse_placement_hint(candidate("org/Model-7B-Instruct"), req, vulkan_only)
    awq = coarse_placement_hint(candidate("org/Model-7B-Instruct-AWQ"), req, vulkan_only)
    on_cuda = coarse_placement_hint(candidate("org/Model-7B-Instruct"), req, hardware(vram_gib=24))

    assert gguf.mode is HardwareFitMode.GPU_RESIDENT
    assert plain.mode is not HardwareFitMode.GPU_RESIDENT  # Transformers cannot use Vulkan
    assert plain.quantization_hint == "float16"  # no 4-bit assumption without a hint
    assert plain.estimated_weight_bytes > 13 * 1024**3  # 7B at two bytes per parameter
    assert awq.mode is HardwareFitMode.TOO_LARGE  # AWQ needs CUDA, even on CPU
    assert on_cuda.mode in {HardwareFitMode.GPU_RESIDENT, HardwareFitMode.GPU_OFFLOAD}


def test_candidates_that_can_use_the_accelerator_are_not_displaced_by_cpu_only_ones() -> None:
    """Regression found on a B60 box: the shortlist went round every placement class, so
    when slots were scarce small CPU-only checkpoints took places from models that run
    on the card."""
    from tests._intel_fixtures import LINUX_OS, scan

    vulkan_only = scan(LINUX_OS).model_copy(update={"memory": hardware(ram_gib=64).memory})
    gguf = ["gguf", "Q4_K_M"]
    on_the_card = [
        candidate(f"org/Chat-{size}B-Instruct-GGUF", tags=gguf, downloads=100_000)
        for size in (7, 8, 9, 12)
    ]
    cpu_only = [
        candidate(f"vendor{index}/Tiny-3B-Instruct", downloads=5_000_000)
        for index in range(4)
    ]

    picked = shortlist([*cpu_only, *on_the_card], _requirements(), limit=4, hardware=vulkan_only)

    assert {item.repo_id for item in picked} == {item.repo_id for item in on_the_card}


def test_family_diversity_does_not_beat_a_far_better_score() -> None:
    """Regression found on a B60 box: with `qwen2` already picked, the shortlist skipped
    every official Qwen coder (score ~22) for a 39-download repo (score ~7) whose name
    started with a family it had not seen yet."""
    req = _requirements()
    gguf = ["gguf", "Q4_K_M"]
    popular = {"downloads": 1_000_000, "likes": 5_000}
    pool = [
        candidate("Acme/Coder-1.5B-Instruct-GGUF", tags=gguf, **popular),
        candidate("Acme/Coder-7B-Instruct-GGUF", tags=gguf, **popular),
        candidate("Acme/Coder-14B-Instruct-GGUF", tags=gguf, **popular),
        candidate("nobody/zebra-14b-gguf", tags=gguf, downloads=39, likes=0),
    ]

    picked = shortlist(pool, req, limit=2, hardware=hardware(vram_gib=24, ram_gib=16))

    assert "nobody/zebra-14b-gguf" not in {item.repo_id for item in picked}
    assert {item.repo_id for item in picked} == {
        "Acme/Coder-7B-Instruct-GGUF", "Acme/Coder-14B-Instruct-GGUF",
    }


def test_unknown_size_candidates_get_one_round_pick_and_do_not_crowd_out_known_ones() -> None:
    req = _requirements()
    gguf = ["gguf", "Q4_K_M"]
    popular = {"downloads": 1_000_000, "likes": 5_000}
    known = [
        candidate(f"Acme/Coder-{size}B-Instruct-GGUF", tags=gguf, **popular)
        for size in (0.5, 0.6, 0.8, 1.5, 3, 4, 7, 8, 14)
    ]
    relevant_unknown = candidate("bigco/Wizard-Coder-GGUF", tags=gguf, **popular)
    junk = [
        candidate(
            f"misc{index}/code_generation_model{index}", tags=gguf, downloads=40 + index, likes=0,
        )
        for index in range(4)
    ]

    picked = shortlist(
        [*known, relevant_unknown, *junk],
        req,
        limit=8,
        hardware=hardware(vram_gib=24, ram_gib=16),
    )

    names = {item.repo_id for item in picked}
    unknown = [item for item in picked if parameter_count_hint(item.repo_id) is None]
    assert len(unknown) == 1
    assert "bigco/Wizard-Coder-GGUF" in names  # a relevant unknown still gets its slot


def test_reuploads_of_one_model_take_a_single_candidate_slot() -> None:
    """Found on a B60 box: twelve of the hundred most downloaded 20-32B GGUF repositories were
    quantizations of the same Qwen3.8-27B, and a search page holds twenty. Hugging Face declares
    that relation with a `base_model:quantized:` tag, so they are merged before inspection. A
    fine-tune of the model, or a different model, is not a re-upload."""
    base = "Qwen/Qwen3.8-27B"
    quantized = [
        candidate(repo_id=f"{owner}/Qwen3.8-27B-GGUF", downloads=downloads,
                  tags=["gguf", f"base_model:quantized:{base}", f"base_model:{base}"])
        for owner, downloads in (("unsloth", 5_900_000), ("ggml-org", 2_200_000), ("byte", 10))
    ]
    finetune = candidate(
        repo_id="huihui/Qwen3.8-27B-abliterated-GGUF", downloads=2_600_000,
        tags=["gguf", "base_model:quantized:huihui/Qwen3.8-27B-abliterated"],
    )
    other = candidate(repo_id="google/gemma-4-26B-it-GGUF", downloads=1_000_000, tags=["gguf"])
    unlinked = candidate(repo_id="acme/Plain-7B-GGUF", downloads=5, tags=["gguf"])
    # Found right after: Qwen's 1M-download AWQ build of the same base absorbed its GGUF repository.
    # Only GGUF copies merge with each other; another quantization format is a different artifact.
    awq = candidate(
        repo_id="Qwen/Qwen3.8-27B-AWQ", downloads=9_000_000,
        tags=["awq", f"base_model:quantized:{base}"],
    )

    reuploads = collapse_quantized_reuploads([awq, *quantized, finetune, other, unlinked])
    kept = reuploads.kept

    assert [item.repo_id for item in kept] == [
        "Qwen/Qwen3.8-27B-AWQ",
        "unsloth/Qwen3.8-27B-GGUF",
        "huihui/Qwen3.8-27B-abliterated-GGUF",
        "google/gemma-4-26B-it-GGUF",
        "acme/Plain-7B-GGUF",
    ]
    # The copies that were merged are not lost: they wait, most downloaded first, to be
    # inspected if the kept one turns out to have nothing that fits.
    assert [item.repo_id for item in reuploads.alternatives["unsloth/qwen3.8-27b-gguf"]] == [
        "ggml-org/Qwen3.8-27B-GGUF",
        "byte/Qwen3.8-27B-GGUF",
    ]
    assert "huihui/qwen3.8-27b-abliterated-gguf" not in reuploads.alternatives


def test_small_models_only_fill_what_larger_ones_that_fit_leave() -> None:
    """Regression found on a 24 GiB box: five very popular sub-4B models took inspection
    slots that no 7B-32B model fitting the same placement class got, so a whole size range
    was never evaluated. On a machine where only small models fit, they still qualify."""
    req = _requirements()
    gguf = ["gguf", "Q4_K_M"]
    tiny = [
        candidate(f"Acme/Chat-{size}B-Instruct-GGUF", tags=gguf, downloads=5_000_000, likes=900)
        for size in (0.5, 1.5, 3)
    ]
    larger = [
        candidate(f"Acme/Chat-{size}B-Instruct-GGUF", tags=gguf, downloads=40_000, likes=10)
        for size in (7, 8, 14, 32)
    ]
    pool = [*tiny, *larger]

    big_box = shortlist(pool, req, limit=5, hardware=hardware(vram_gib=24, ram_gib=16))
    sizes = {parameter_count_hint(item.repo_id) for item in big_box}
    assert {7, 8, 14, 32} <= sizes  # every larger one, before any small one

    small_box = shortlist(pool, req, limit=3, hardware=hardware(vram_gib=4, ram_gib=8))
    small_sizes = {parameter_count_hint(item.repo_id) for item in small_box}
    assert {0.5, 1.5, 3} <= small_sizes


def test_hardware_aware_shortlist_is_order_deterministic() -> None:
    req = _requirements()
    pool = [
        candidate(repo_id=f"org/Model-{index}B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
        for index in range(1, 18)
    ]
    profile = hardware(vram_gib=8, ram_gib=32)
    first = [item.repo_id for item in shortlist(pool, req, limit=12, hardware=profile)]
    second = [
        item.repo_id for item in shortlist(list(reversed(pool)), req, limit=12, hardware=profile)
    ]
    assert first == second
    assert len(first) == 12


def test_hardware_aware_shortlist_changes_with_context_and_concurrency() -> None:
    req = _requirements()
    profile = hardware(vram_gib=8, ram_gib=32)
    model = candidate(repo_id="org/Assistant-7B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
    short_context = coarse_placement_hint(model, req, profile)
    long_busy = coarse_placement_hint(
        model,
        req.model_copy(update={"desired_context": 32768, "concurrent_users": 4}),
        profile,
    )
    assert short_context.mode is HardwareFitMode.GPU_RESIDENT
    assert long_busy.mode is not HardwareFitMode.GPU_RESIDENT


def test_hardware_aware_shortlist_does_not_let_impossible_models_flood_the_budget() -> None:
    req = _requirements()
    impossible = [
        candidate(repo_id=f"Huge/Huge-{70 + index}B-Q8_0-GGUF", tags=["gguf", "Q8_0"])
        for index in range(30)
    ]
    plausible = [
        candidate(repo_id=f"Small/Small-{index}B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
        for index in (1, 3, 7, 14)
    ]
    picked = shortlist(
        impossible + plausible,
        req,
        limit=12,
        hardware=hardware(vram_gib=8, ram_gib=32),
    )
    assert {item.repo_id for item in plausible} <= {item.repo_id for item in picked}
    assert sum(item.repo_id.startswith("Huge/") for item in picked) <= 2


def test_hardware_aware_shortlist_keeps_borderline_plausible_candidate() -> None:
    req = _requirements()
    borderline = candidate(
        repo_id="org/Borderline-14B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"]
    )
    picked = shortlist(
        [borderline, candidate(repo_id="org/Impossible-70B-Q8_0-GGUF", tags=["gguf", "Q8_0"])],
        req,
        limit=1,
        hardware=hardware(vram_gib=8, ram_gib=32),
    )
    assert picked == [borderline]


def test_valid_gguf_and_transformers_repositories_survive() -> None:
    outcome = filter_candidates(
        [
            candidate(repo_id="org/gguf", tags=["text-generation", "gguf"]),
            candidate(repo_id="org/tf", tags=["text-generation", "safetensors"]),
        ],
        _requirements(),
    )
    assert {c.repo_id for c in outcome.kept} == {"org/gguf", "org/tf"}


def test_the_client_sends_the_size_window_only_when_the_query_has_one() -> None:
    api = _StubApi()
    client = HfSearchClient(api=api)  # type: ignore[arg-type]
    client.search(_query())
    client.search(SearchQuery(label="scale", min_billions=4, max_billions=30.5))
    assert "num_parameters" not in api.calls[0]
    assert api.calls[1]["num_parameters"] == "min:4B,max:30.5B"


# The candidates the B60 preselection of "general chat / balanced" had to choose from, with the
# numbers it saw: (repository, downloads, likes, queries that returned it, penalties).
_SAVED_CHAT_POOL = (
    ("unsloth/Qwen3.8-27B-GGUF", 6_009_562, 5364, 1, 2),
    ("ornith-ai/Ornith-1.5-9B-GGUF", 5_217_055, 600, 1, 1),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", 466_321, 1179, 3, 1),
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", 631_948, 203, 3, 0),
    ("ornith-ai/Ornith-1.0-9B-GGUF", 2_411_812, 697, 1, 1),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", 306_996, 593, 3, 0),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", 512_555, 426, 2, 0),
    ("ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF", 1_528_033, 447, 1, 1),
    ("bartowski/Qwen2.5-32B-Instruct-GGUF", 926_783, 92, 2, 0),
    ("Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", 177_825, 265, 3, 0),
    ("tencent/Hy-MT2-7B-GGUF", 1_003_884, 96, 1, 2),
    ("bartowski/Qwen2.5-14B-Instruct-GGUF", 159_000, 75, 2, 0),
    ("TheBloke/Llama-2-7B-Chat-GGUF", 61_995, 517, 1, 0),
    ("LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF", 86_800, 216, 1, 0),
    ("JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q8_0", 140_000, 9, 2, 0),
    ("MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF", 67_000, 103, 1, 1),
    ("bartowski/Qwen2.5-Coder-14B-Instruct-abliterated-GGUF", 42_000, 70, 1, 0),
    ("openthaigpt/openthaigpt1.5-7b-instruct", 69_000, 17, 1, 0),
    ("bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF", 69_000, 30, 1, 2),
    ("MaziyarPanahi/Qwen2-7B-Instruct-GGUF", 62_000, 10, 1, 2),
    ("MaziyarPanahi/Yi-Coder-9B-Chat-GGUF", 49_000, 10, 1, 2),
    ("MaziyarPanahi/Yi-1.5-6B-Chat-GGUF", 47_000, 9, 1, 2),
)


def _saved_chat_candidates() -> list[Any]:
    return [
        candidate(repo, downloads=downloads, likes=likes, tags=["gguf", "text-generation"])
        .model_copy(update={
            "source_queries": [f"q{index}" for index in range(queries)],
            "penalties": [f"penalty {index}" for index in range(penalties)],
        })
        for repo, downloads, likes, queries, penalties in _SAVED_CHAT_POOL
    ]


def _chat_shortlist(limit: int = 12, pool: list[Any] | None = None) -> list[str]:
    requirements = _requirements(UseCase.GENERAL_CHAT)
    picked = shortlist(
        pool if pool is not None else _saved_chat_candidates(), requirements, limit,
        hardware=hardware(vram_gib=24, ram_gib=16),
        targets_other_task=lambda item: targets_another_task(item, requirements),
    )
    return [item.repo_id for item in picked]


def test_a_chat_candidate_does_not_lose_every_slot_to_specialists_of_another_task() -> None:
    """Found on a B60 box: the 14B chat model missed the last inspection slot by 0.21 points while
    three code models took slots in a general-chat search, because three queries had returned
    them. A repository that advertises another task (a `coder` for chat) fills a slot only after
    the others; popularity, likes and the number of queries order what is left."""
    picked = _chat_shortlist()

    assert "bartowski/Qwen2.5-14B-Instruct-GGUF" in picked
    assert not [repo for repo in picked if "coder" in repo.lower()]
    assert len(picked) == 12


def test_a_specialist_of_another_task_still_fills_a_slot_nothing_else_wants() -> None:
    pool = [item for item in _saved_chat_candidates()
            if item.repo_id in {"Qwen/Qwen2.5-7B-Instruct-GGUF",
                                "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"}]

    assert sorted(_chat_shortlist(limit=5, pool=pool)) == sorted(item.repo_id for item in pool)


def test_two_versions_of_one_model_do_not_take_two_early_slots_but_stay_as_reserve() -> None:
    """The second Ornith of the same size waits behind other candidates; it is not merged into the
    first, and it is inspected if the budget is not used up. A different size, a different family
    or a fine-tune of another family is a different option and is not held back."""
    picked = _chat_shortlist()
    assert "ornith-ai/Ornith-1.5-9B-GGUF" in picked
    assert "ornith-ai/Ornith-1.0-9B-GGUF" not in picked  # reserve: enough others to fill 12

    pool = [item for item in _saved_chat_candidates() if item.repo_id in {
        "ornith-ai/Ornith-1.5-9B-GGUF", "ornith-ai/Ornith-1.0-9B-GGUF",
        "unsloth/Qwen3.8-27B-GGUF", "ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF",
        "bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen/Qwen2.5-7B-Instruct-GGUF",
    }]
    everyone = _chat_shortlist(limit=6, pool=pool)
    assert len(everyone) == 6  # the reserve is used when nothing else is left
    four = _chat_shortlist(limit=4, pool=pool)
    assert "ornith-ai/Ornith-1.0-9B-GGUF" not in four
    assert {"unsloth/Qwen3.8-27B-GGUF", "ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF"} <= set(
        _chat_shortlist(limit=5, pool=pool)
    )


def test_a_translation_or_summarization_model_is_another_task_unless_that_is_the_task() -> None:
    """Found on a B60 box: a translation model (declared by its base repository) took a slot of a
    general-chat search. Tasks of a different kind count as another task, except when they are
    what was asked for; a coder still counts for chat as before."""
    chat = _requirements(UseCase.GENERAL_CHAT)
    translating = _requirements(UseCase.WRITING_TRANSLATION)
    summarizing = _requirements(UseCase.SUMMARIZATION_EXTRACTION)

    translator = candidate("org/MT-7B-GGUF", tags=["gguf", "conversational", "translation"])
    summarizer = candidate("org/Sum-7B-GGUF", tags=["gguf", "instruct", "summarization"])
    coder = candidate("org/Coder-7B-GGUF", tags=["gguf", "instruct"])
    plain = candidate("org/Plain-7B-Instruct-GGUF", tags=["gguf", "instruct"])

    assert targets_another_task(translator, chat) and targets_another_task(summarizer, chat)
    assert targets_another_task(coder, chat)
    assert not targets_another_task(plain, chat)
    assert not targets_another_task(translator, translating)
    assert targets_another_task(translator, summarizing)
    assert not targets_another_task(summarizer, summarizing)
