"""Capture/replay the real discovery prefix without inspecting or downloading weights."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jaull.application.recommendation import policies
from jaull.application.requirements import build_requirements
from jaull.discovery import candidate_filter, query_builder
from jaull.discovery.search_client import HfSearchClient, ModelSearchClient
from jaull.domain.candidates import ModelCandidate, SearchQuery
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import UserAnswers
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.workflow.orchestrator import _interleave, _memory_budget


def capture(queries: list[SearchQuery], client: ModelSearchClient) -> list[dict[str, Any]]:
    observations = []
    for query in queries:
        error = None
        try:
            candidates = client.search(query)
        except HuggingFaceUnavailableError as exc:
            candidates = []
            error = str(exc)
        observations.append({
            "query": query.model_dump(mode="json"),
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "error": error,
        })
    return observations


def audit_pool(
    answers: UserAnswers,
    hardware: HardwareProfile,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use production filtering/budgets; later inspection/ranking is explicitly unobserved."""
    requirements = build_requirements(answers, hardware)
    expected = query_builder.build_queries(requirements, limit=policies.SEARCH_RESULTS_PER_QUERY)
    queries = [SearchQuery.model_validate(row["query"]) for row in observations]
    if queries != expected:
        raise ValueError("Capture queries differ from the current request; recapture required")
    results = [
        [ModelCandidate.model_validate(item) for item in row["candidates"]]
        for row in observations
    ]
    if any(len(items) > query.limit for query, items in zip(queries, results, strict=True)):
        raise ValueError("Capture exceeds its per-query limit")
    if any(row["error"] is not None and row["candidates"] for row in observations):
        raise ValueError("A failed query cannot claim successful results")
    unique = candidate_filter.deduplicate(_interleave(results))
    filtered = candidate_filter.filter_candidates(unique, requirements)
    budgeted = filtered.kept[:policies.MAX_UNIQUE_CANDIDATES]
    shortlist = candidate_filter.shortlist(
        budgeted, requirements, policies.MAX_DEEP_INSPECTION,
        budget_bytes=_memory_budget(hardware), hardware=hardware,
    )
    rejected = dict(filtered.rejected)
    in_budget = {item.repo_id for item in budgeted}
    shortlisted = {item.repo_id for item in shortlist}
    rows = []
    for item in unique:
        hint = candidate_filter.coarse_placement_hint(item, requirements, hardware)
        if item.repo_id in rejected:
            stage, reason = "metadata_filter", rejected[item.repo_id]
        elif item.repo_id not in in_budget:
            stage, reason = "unique_budget", "Eligible but beyond the unique-candidate budget"
        elif item.repo_id not in shortlisted:
            stage = "shortlist_selection"
            reason = "Not selected by the coarse hardware/diversity budget"
        else:
            stage = "shortlisted"
            reason = "Deep inspection pending; fit/artifact/readiness not verified"
        rows.append({
            "repo_id": item.repo_id, "source_queries": item.source_queries,
            "stage": stage, "reason": reason,
            "coarse_mode": hint.mode.value if hint.mode is not None else None,
            "coarse_reason": hint.reason,
        })
    failures = sum(row["error"] is not None for row in observations)
    return {
        "schema_version": 1,
        "status": (
            "unavailable" if failures == len(queries) else "partial" if failures else "captured"
        ),
        "mode": "metadata_only",
        "input": {"answers": answers.model_dump(mode="json"),
                  "hardware": hardware.model_dump(mode="json")},
        "requirements": requirements.model_dump(mode="json"),
        "budgets": {"per_query": policies.SEARCH_RESULTS_PER_QUERY,
                    "unique": policies.MAX_UNIQUE_CANDIDATES,
                    "deep_inspection": policies.MAX_DEEP_INSPECTION},
        "queries": observations,
        "counts": {
            "query_failures": failures,
            "sightings": sum(map(len, results)), "unique": len(unique),
            "metadata_rejected": len(filtered.rejected), "eligible": len(filtered.kept),
            "unique_budget_dropped": len(filtered.kept) - len(budgeted),
            "budgeted": len(budgeted), "shortlisted": len(shortlist),
            "shortlist_dropped": len(budgeted) - len(shortlist),
        },
        "budgeted_order": [item.repo_id for item in budgeted],
        "shortlist_order": [item.repo_id for item in shortlist],
        "candidates": rows,
        "limitations": [
            "One bounded search sample, not exhaustive Hub recall or model quality.",
            "No deep inspection, artifact availability, HFA, readiness or final ranking observed.",
            "Coarse placement hints are not verified fits or guaranteed shortlist explanations.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--hardware-kind", choices=("measured", "synthetic"), required=True)
    parser.add_argument("--replay", type=Path, help="Read a saved audit instead of calling HF")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    answers = UserAnswers.model_validate_json(args.answers.read_text(encoding="utf-8"))
    hardware = HardwareProfile.model_validate_json(args.hardware.read_text(encoding="utf-8"))
    queries = query_builder.build_queries(
        build_requirements(answers, hardware), limit=policies.SEARCH_RESULTS_PER_QUERY,
    )
    started = datetime.now(tz=UTC).isoformat()
    if args.replay:
        previous = json.loads(args.replay.read_text(encoding="utf-8"))
        observations = previous["queries"]
        source = {"kind": "replay", "captured_at": previous["capture_started_at"]}
    else:
        # This audit uses only public metadata; no saved credentials are needed.
        from huggingface_hub import HfApi
        observations = capture(queries, HfSearchClient(api=HfApi(token=False)))
        source = {"kind": "live", "captured_at": started}
    report = audit_pool(answers, hardware, observations)
    report.update({"capture_started_at": source["captured_at"], "generated_at":
                   datetime.now(tz=UTC).isoformat(), "source": source,
                   "hardware_kind": args.hardware_kind})
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
