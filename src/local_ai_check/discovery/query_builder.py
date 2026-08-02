"""Build the Hugging Face search queries for a set of requirements.

Several complementary queries beat one clever string: each use case gets a few
angles (task wording, format, language), and the results are deduplicated later.
Every query is a declarative :class:`SearchQuery`, which is what makes this
testable without touching the network.
"""

from __future__ import annotations

from local_ai_check.discovery.models import SearchQuery
from local_ai_check.workflow import policies
from local_ai_check.workflow.models import UseCase, UserRequirements

# Search phrases per use case. Deliberately plain: the Hub's search is a
# full-text match over repo names and cards, so jargon returns less than the
# words maintainers actually put in their model names.
_USE_CASE_PHRASES: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("instruct", "chat", "multilingual instruct"),
    UseCase.CODING: ("coder instruct", "code generation", "programming assistant"),
    UseCase.DOCUMENT_QA: (
        "instruct long context",
        "multilingual instruct",
        "question answering instruct",
    ),
    UseCase.WRITING_TRANSLATION: (
        "multilingual instruct",
        "translation instruct",
        "writing assistant",
    ),
}

# Tag filters worth adding as their own query so quantized builds surface even
# when the plain-text search is dominated by the original repositories.
_FORMAT_TAGS: dict[str, str] = {
    "gguf": "gguf",
    "safetensors": "safetensors",
}


def build_queries(requirements: UserRequirements) -> list[SearchQuery]:
    """Return the ordered, deduplicated set of queries for these requirements."""
    queries: list[SearchQuery] = []
    seen: set[tuple[str | None, str | None, tuple[str, ...], str, int]] = set()

    def add(query: SearchQuery) -> None:
        key = query.cache_key()
        if key in seen:
            return
        seen.add(key)
        queries.append(query)

    phrases = _USE_CASE_PHRASES[requirements.use_case]

    for phrase in phrases:
        add(
            SearchQuery(
                label=f"{requirements.use_case.value}:{phrase}",
                search=phrase,
                pipeline_tag=requirements.pipeline_tag,
                sort="downloads",
                limit=policies.SEARCH_RESULTS_PER_QUERY,
            )
        )

    # One query per preferred format, using the primary phrase. This is what
    # surfaces GGUF conversions, which rarely rank highly on the plain search.
    primary_phrase = phrases[0]
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
                limit=policies.SEARCH_RESULTS_PER_QUERY,
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
                limit=policies.SEARCH_RESULTS_PER_QUERY,
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
            limit=policies.SEARCH_RESULTS_PER_QUERY,
        )
    )

    return queries


__all__ = ["build_queries"]
