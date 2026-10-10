"""Audit a measured GGUF pair and a captured search; metadata only, no evaluation.

The pair is an explicitly pinned, controlled candidate pool, NOT normal discovery.
The second workflow replays ordinary discovery and audits optional GGUF paths
separately; discovering a path never changes the saved search or selected plan.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.audit_recommendation_tail import ReplaySearchClient

from jaull.advisor.quality_candidates import _gguf_path
from jaull.advisor.service import AdvisorService
from jaull.artifacts.service import ArtifactService
from jaull.domain.candidates import ModelCandidate, SearchQuery
from jaull.domain.execution_plans import ArtifactVariantFormat, IdentityMatchStatus
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import RecommendationPriority, UserAnswers
from jaull.evaluation.quality_comparison import compare_quality_records
from jaull.evaluation.quality_records import load_record
from jaull.recommendation.shadow import recompare_quality
from jaull.workflow.orchestrator import run_workflow
from jaull.workflow.state import RecommendationWorkflowState


def require(condition: bool, message: str) -> None:
    """An audit claim, checked even under ``python -O`` (which strips asserts)."""
    if not condition:
        raise RuntimeError(message)


class PinnedPairSearch:
    def __init__(self, analyses: dict[str, ModelAnalysis]) -> None:
        self.analyses = analyses

    def search(self, query: SearchQuery) -> list[ModelCandidate]:
        return [ModelCandidate(
            repo_id=repo, license=a.repo.license, tags=a.repo.tags,
            pipeline_tag=a.repo.pipeline_tag, library_name=a.repo.library_name,
            downloads=a.repo.downloads or 0, likes=a.repo.likes or 0,
            source_queries=[query.label], repository_type=a.classification.primary_type,
        ) for repo, a in self.analyses.items()]


def summarize(state: RecommendationWorkflowState, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not state.completed or state.requirements is None or not state.ranked_plans:
        raise ValueError(
            f"Workflow did not produce a pool: {state.errors}, {state.no_results_reason}",
        )
    before = state.model_dump_json()
    proposal = recompare_quality(state.ranked_plans, state.requirements, records, limit=5)
    require(before == state.model_dump_json(), "Recompare mutated the original search")
    require(proposal == recompare_quality(
        state.ranked_plans, state.requirements, list(reversed(records)), limit=5,
    ), "Store order changed the proposal")
    return {
        "top5": [rec.repo_id for rec in state.recommendations],
        "plans": [{"repo": p.evaluated.repo_id, "plan_id": p.plan.plan_id,
                   "artifact": p.plan.artifact.model_dump(mode="json"),
                   "assessment": p.assessment.model_dump(mode="json")}
                  for p in state.ranked_plans],
        "warnings": state.warnings, "errors": state.errors,
        "shadow": proposal.model_dump(mode="json"),
        "original_unchanged": True, "store_order_invariant": True,
    }


def audit(advisor: AdvisorService, capture: dict[str, Any],
          pair: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pair) != 2:
        raise ValueError("Choose exactly two completed records")
    manifests = [json.loads(r["provenance"]["evidence"]["verified-artifact.json"]) for r in pair]
    if len({m["repo_id"] for m in manifests}) != 2:
        raise ValueError("The controlled pair must name two different repositories")
    hardware = HardwareProfile.model_validate(pair[0]["provenance"]["hardware"])
    answers = UserAnswers.model_validate(capture["input"]["answers"]).model_copy(
        update={"priority": RecommendationPriority.QUALITY},
    )
    records = advisor._stored_quality_records()
    context = replace(advisor._plan_ranking_context(hardware), quality_records=())
    analyses = {}
    for manifest, record in zip(manifests, pair, strict=True):
        analysis = advisor.inspect_model(manifest["repo_id"])
        exact = [v for v in analysis.classification.gguf_variants
                 if v.sha256 == record["identity"]["artifact_sha256"]
                 and len(v.files) == 1 and v.files[0].path == manifest["filename"]
                 and v.total_bytes == manifest["size_bytes"]]
        if len(exact) != 1:
            raise ValueError(
                f"Metadata does not identify the measured artifact: {manifest['repo_id']}",
            )
        analyses[manifest["repo_id"]] = analysis.model_copy(update={
            "classification": analysis.classification.model_copy(update={"gguf_variants": exact}),
        })

    def inspect(repo_id: str, **_: Any) -> ModelAnalysis:
        return analyses[repo_id]

    services = replace(advisor.services, search_client=PinnedPairSearch(analyses),
                       inspect_model=inspect, model_analysis_cache=None)
    controlled = run_workflow(answers, hardware, services, plan_context=context)
    controlled_report = summarize(controlled, records)
    require({p.plan.artifact.sha256 for p in controlled.ranked_plans} == {
        r["identity"]["artifact_sha256"] for r in pair
    }, "The controlled workflow selected different bytes")

    services = replace(advisor.services, search_client=ReplaySearchClient(capture["queries"]))
    ordinary = run_workflow(answers, hardware, services, plan_context=context)
    ordinary_report = summarize(ordinary, records)
    before = ordinary.model_dump_json()
    paths = []
    for rec in ordinary.recommendations:
        require(rec.plan is not None, f"{rec.repo_id} has no plan")
        assert rec.plan is not None  # For the type checker; require() raised otherwise.
        saved = _gguf_path(rec.plan, ordinary.ranked_plans)
        row: dict[str, Any] = {
            "repo": rec.repo_id, "selected": rec.plan.artifact.label,
            "saved_gguf": saved.artifact.model_dump(mode="json") if saved else None,
        }
        try:
            variants = advisor.discover_artifact_variants(recommendation=rec, limit=10)
            row["optional_confirmed_gguf"] = [v.model_dump(mode="json") for v in variants
                if v.format is ArtifactVariantFormat.GGUF and v.sha256 and v.file_count == 1
                and v.identity_match is IdentityMatchStatus.CONFIRMED]
        except Exception as exc:
            row["lookup_error"] = f"{type(exc).__name__}: {exc}"
        paths.append(row)
    require(ordinary.model_dump_json() == before, "Variant discovery changed the search")
    ordinary_report["paths"] = paths
    return {
        "at": datetime.now(tz=UTC).isoformat(), "mode": "metadata_only",
        "hardware": hardware.model_dump(mode="json"),
        "hardware_source": "historical measured hardware from the first record",
        "answers": answers.model_dump(mode="json"),
        "record_ids": [r["identity_sha256"] for r in pair],
        "comparison": compare_quality_records(pair[0], pair[1],
            left_source=manifests[0]["filename"], right_source=manifests[1]["filename"]),
        "controlled_pinned_pair": controlled_report,
        "ordinary_captured_discovery": ordinary_report,
        "capture_date": capture.get("capture_started_at"),
        "limitations": [
            "Controlled selection is not normal discovery or a general quality verdict.",
            "Ordinary discovery is a frozen sample; subsequent metadata may be cached or live.",
            "Optional paths are not selected plans; fit and IFEval eligibility are unverified.",
            "Quality remains shadow-only; no speed or Balanced activation is implied.",
        ],
    }


def forbidden(*_: Any, **__: Any) -> None:
    raise AssertionError("This audit must not download weights or execute an evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs=2, type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    pair = [load_record(p) for p in args.records]
    with patch.object(ArtifactService, "download", forbidden), patch.object(
        AdvisorService, "run_quality_evaluation", forbidden,
    ):
        report = audit(AdvisorService.default(), capture, pair)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "controlled_groups": len(
        report["controlled_pinned_pair"]["shadow"]["groups"]), "controlled_moves": len(
        report["controlled_pinned_pair"]["shadow"]["moves"]),
        "ordinary_top5": report["ordinary_captured_discovery"]["top5"]}, indent=2))


if __name__ == "__main__":
    main()
