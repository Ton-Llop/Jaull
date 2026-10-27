"""Build the Hugging Face search queries for a set of requirements.

Several complementary queries beat one clever string: each use case gets a few
angles (task wording, format, language), and the results are deduplicated later.
Every query is a declarative :class:`SearchQuery`, which is what makes this
testable without touching the network.
"""

from __future__ import annotations

from jaull.discovery.candidate_filter import LARGER_THAN_SMALL_BILLIONS
from jaull.domain.candidates import SearchQuery
from jaull.domain.requirements import UseCase, UserRequirements

# Default when the caller does not pass a limit. Kept generous — the actual
# guided run overrides it from ``jaull.application.recommendation.policies``.
DEFAULT_RESULTS_PER_QUERY = 20

# Search phrases per use case. list_models documents search over model IDs,
# not semantic task search over model cards; naming can limit recall.
_USE_CASE_PHRASES: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("instruct", "chat", "multilingual instruct"),
    UseCase.CODING: ("coder instruct", "code generation", "programming assistant"),
    UseCase.DOCUMENT_QA: (
        "instruct long context",
        "multilingual instruct",
        "question answering instruct",
    ),
    UseCase.SUMMARIZATION_EXTRACTION: (
        "summarization instruct",
        "information extraction instruct",
        "document assistant",
    ),
    UseCase.REASONING: (
        "reasoning instruct",
        "math reasoning",
        "problem solving instruct",
    ),
    UseCase.WRITING_TRANSLATION: (
        "multilingual instruct",
        "translation instruct",
        "writing assistant",
    ),
}

# These tasks can use general instruction-tuned models whose repository names
# do not advertise the task. Suitability is assessed after discovery.
_GENERAL_INSTRUCT_USE_CASES = frozenset({UseCase.DOCUMENT_QA, UseCase.REASONING})

# Tag filters worth adding as their own query so quantized builds surface even
# when the plain-text search is dominated by the original repositories.
_FORMAT_TAGS: dict[str, str] = {
    "gguf": "gguf",
    "safetensors": "safetensors",
}


def build_queries(
    requirements: UserRequirements,
    limit: int = DEFAULT_RESULTS_PER_QUERY,
    *,
    max_billions: float | None = None,
) -> list[SearchQuery]:
    """Return the ordered, deduplicated set of queries for these requirements.

    ``limit`` is the ``list_models`` page size per query. The guided orchestrator
    supplies its own value from ``jaull.application.recommendation.policies``;
    callers that only want to build the queries can omit it.

    ``max_billions`` is the largest size that fits the machine. Every plain query sorts
    by downloads, so the most popular small models fill every page; when sizes beyond
    the small ones fit, one more query is restricted to that range.
    """
    queries: list[SearchQuery] = []
    seen: set[object] = set()

    def add(query: SearchQuery) -> None:
        key = query.cache_key()
        if key in seen:
            return
        seen.add(key)
        queries.append(query)

    phrases = _USE_CASE_PHRASES[requirements.use_case]
    primary_phrase = (
        "instruct" if requirements.use_case in _GENERAL_INSTRUCT_USE_CASES else phrases[0]
    )

    for phrase in (*phrases, primary_phrase):
        add(
            SearchQuery(
                label=f"{requirements.use_case.value}:{phrase}",
                search=phrase,
                pipeline_tag=requirements.pipeline_tag,
                sort="downloads",
                limit=limit,
            )
        )

    # Format queries must also reach general Instruct builds for documents and
    # reasoning, without requiring the task wording in the repository name.
    for fmt in requirements.preferred_formats:
        tag = _FORMAT_TAGS.get(fmt)
        if tag is None:
            continue
        add(
            SearchQuery(
                label=f"{requirements.use_case.value}:{fmt}",
                search=primary_phrase,
                pipeline_tag=requirements.pipeline_tag,
                filter_tags=(tag,),
                sort="downloads",
                limit=limit,
            )
        )

    # A language-tagged query for every non-English language, since multilingual
    # models are often only discoverable through their language tag.
    for language in requirements.languages:
        if language == "en":
            continue
        add(
            SearchQuery(
                label=f"{requirements.use_case.value}:lang-{language}",
                search=primary_phrase,
                pipeline_tag=requirements.pipeline_tag,
                filter_tags=(language,),
                sort="downloads",
                limit=limit,
            )
        )

    if max_billions is not None and max_billions > LARGER_THAN_SMALL_BILLIONS:
        fmt_tag = next(
            (_FORMAT_TAGS[fmt] for fmt in requirements.preferred_formats if fmt in _FORMAT_TAGS),
            None,
        )
        add(
            SearchQuery(
                label=f"{requirements.use_case.value}:scale",
                search=primary_phrase,
                pipeline_tag=requirements.pipeline_tag,
                filter_tags=(fmt_tag,) if fmt_tag else (),
                sort="downloads",
                limit=limit,
                min_billions=LARGER_THAN_SMALL_BILLIONS,
                max_billions=max_billions,
            )
        )
        if fmt_tag is not None:
            # No phrase and no pipeline: repositories such as Qwen3.8-27B's GGUF declare neither
            # and do not carry "instruct" in their name. The client still drops multimodal and
            # non-text candidates, and the ranking decides suitability from their metadata.
            add(
                SearchQuery(
                    label=f"{requirements.use_case.value}:scale-open",
                    filter_tags=(fmt_tag,),
                    sort="downloads",
                    limit=limit,
                    min_billions=LARGER_THAN_SMALL_BILLIONS,
                    max_billions=max_billions,
                )
            )

    # A trending query gives recent releases a chance against the all-time
    # download leaders, which otherwise dominate every `sort="downloads"` page.
    add(
        SearchQuery(
            label=f"{requirements.use_case.value}:trending",
            search=primary_phrase,
            pipeline_tag=requirements.pipeline_tag,
            sort="trending_score",
            limit=limit,
        )
    )

    return queries


__all__ = ["build_queries"]
