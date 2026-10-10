"""Offline stage attribution, checked against the actual guided workflow."""

from __future__ import annotations

import pytest
from scripts.audit_recommendation_pool import audit_pool, capture

from jaull.application.recommendation import policies
from jaull.application.requirements import build_requirements
from jaull.discovery.query_builder import build_queries
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import UseCase
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.workflow.orchestrator import run_workflow
from tests._workflow_fixtures import (
    GIB,
    FakeSearchClient,
    answers,
    candidate,
    gguf_analysis,
    hardware,
)
from tests.test_workflow_orchestrator import _container


@pytest.mark.parametrize("use_case", [UseCase.GENERAL_CHAT, UseCase.CODING])
@pytest.mark.parametrize("profile", [
    hardware(vram_gib=6, name="Synthetic RTX 2060"),
    hardware(vram_gib=8, name="Synthetic RTX 4060"),
    hardware(vram_gib=None, name="Synthetic CPU-only"),
])
def test_pool_trace_matches_workflow_and_attributes_each_exclusion(
    use_case: UseCase, profile: HardwareProfile,
) -> None:
    request = answers(use_case=use_case, languages=["English"])
    queries = build_queries(build_requirements(request, profile), policies.SEARCH_RESULTS_PER_QUERY)
    results = {}
    for index, query in enumerate(queries):
        # A duplicate sighting and a rejected repo appear before eligible models.
        results[query.label] = [
            candidate("org/Shared-Coder-3B-Instruct-GGUF", queries=[query.label]),
            candidate("org/Private-Coder-3B", private=True, queries=[query.label]),
            *[candidate(f"org/Group{index}-Coder-{item + 1}B-Instruct-GGUF",
                        queries=[query.label]) for item in range(18)],
        ]
    observations = capture(queries, FakeSearchClient(results=results))
    report = audit_pool(request, profile, observations)
    counts = report["counts"]
    assert counts["sightings"] == len(queries) * 20
    assert counts["unique"] == 2 + len(queries) * 18
    assert counts["metadata_rejected"] == 1
    assert counts["budgeted"] == policies.MAX_UNIQUE_CANDIDATES
    assert counts["unique_budget_dropped"] == counts["eligible"] - counts["budgeted"]
    assert counts["shortlisted"] == policies.MAX_DEEP_INSPECTION
    assert counts["shortlist_dropped"] == counts["budgeted"] - counts["shortlisted"]
    rows = {row["repo_id"]: row for row in report["candidates"]}
    assert rows["org/Private-Coder-3B"]["stage"] == "metadata_filter"
    assert "private" in rows["org/Private-Coder-3B"]["reason"]
    assert set(rows["org/Shared-Coder-3B-Instruct-GGUF"]["source_queries"]) == {
        query.label for query in queries
    }
    assert {row["stage"] for row in rows.values()} == {
        "metadata_filter", "unique_budget", "shortlist_selection", "shortlisted",
    }

    analyses = {repo: gguf_analysis(repo, quantizations=("Q4_K_M",), base_bytes=GIB)
                for repo in rows}
    state = run_workflow(request, profile, _container(
        FakeSearchClient(results=results), analyses=analyses, vram_budget=6 * GIB,
    ))
    assert state.completed
    assert [item.repo_id for item in state.candidates] == report["budgeted_order"]
    assert [item.repo_id for item in state.evaluated_candidates] == report["shortlist_order"]
    assert len(state.recommendations) == policies.MAX_RECOMMENDATIONS
    assert all(rec.repo_id in report["shortlist_order"] for rec in state.recommendations)


def test_failed_hub_queries_are_unavailable_not_empty_recall() -> None:
    request, profile = answers(), hardware()
    queries = build_queries(build_requirements(request, profile), policies.SEARCH_RESULTS_PER_QUERY)
    observations = capture(queries, FakeSearchClient(raises=HuggingFaceUnavailableError("offline")))
    report = audit_pool(request, profile, observations)
    assert report["status"] == "unavailable"
    assert report["counts"]["query_failures"] == len(queries)
    assert all(row["error"] == "offline" for row in report["queries"])


def test_replay_refuses_other_queries_or_over_budget_results() -> None:
    request, profile = answers(), hardware()
    queries = build_queries(build_requirements(request, profile), policies.SEARCH_RESULTS_PER_QUERY)
    observations = capture(queries, FakeSearchClient())
    with pytest.raises(ValueError, match="queries differ"):
        audit_pool(request, profile, observations[:-1])
    observations[0]["candidates"] = [candidate().model_dump(mode="json")] * 21
    with pytest.raises(ValueError, match="per-query limit"):
        audit_pool(request, profile, observations)
