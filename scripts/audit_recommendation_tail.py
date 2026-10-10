"""Follow a saved discovery capture through inspection, plans, ranking and top 5.

`audit_recommendation_pool` stops at the shortlist on purpose. This picks up
from the same capture: search results are replayed, so the pool is frozen, and
everything after search runs exactly as in production through `run_workflow`.

That part reads Hub metadata (model info, config, safetensors metadata, GGUF
header ranges). It never downloads weights, starts a runtime or evaluates.

Two checks guard the reconstruction against inventing an explanation:
the workflow's 40 budgeted candidates must equal the pool audit's, and the top
5 rebuilt here from the workflow's own evaluated candidates must equal the
workflow's recommendations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.audit_recommendation_pool import audit_pool

from jaull.application.recommendation import policies
from jaull.application.recommendation.service import enrich_candidate_features
from jaull.bootstrap.container import ServiceContainer
from jaull.domain.candidates import ModelCandidate, SearchQuery
from jaull.domain.execution_plans import model_identity_key
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import UserAnswers
from jaull.recommendation.diversity import diversify_ranked_plans
from jaull.recommendation.engine_v2 import PlanRankingContext, rank_execution_plans
from jaull.workflow.orchestrator import run_workflow


class ReplaySearchClient:
    """Serve the captured response for each query; an unknown query is an error."""

    def __init__(self, observations: list[dict[str, Any]]) -> None:
        self._answers = {
            json.dumps(row["query"], sort_keys=True): row for row in observations
        }

    def search(self, query: SearchQuery) -> list[ModelCandidate]:
        key = json.dumps(query.model_dump(mode="json"), sort_keys=True)
        row = self._answers.get(key)
        if row is None:
            raise ValueError(f"Query not in capture: {query.label!r}")
        return [ModelCandidate.model_validate(item) for item in row["candidates"]]


def trace(
    answers: UserAnswers, hardware: HardwareProfile, observations: list[dict[str, Any]],
) -> dict[str, Any]:
    pool = audit_pool(answers, hardware, observations)
    services = dataclasses.replace(
        ServiceContainer.default(), search_client=ReplaySearchClient(observations),
    )
    state = run_workflow(answers, hardware, services)
    if [item.repo_id for item in state.candidates] != pool["budgeted_order"]:
        raise ValueError("Replay did not reproduce the audited 40-candidate budget")

    # Exactly what `recommend` does with hardware present, on the same inputs.
    assert state.requirements is not None
    enriched = enrich_candidate_features(
        state.evaluated_candidates, state.requirements,
        capability_analyzer=services.capability_analyzer,
    )
    ranked = rank_execution_plans(
        enriched, state.requirements, context=PlanRankingContext(hardware=hardware),
    )
    diversified = diversify_ranked_plans(ranked, limit=policies.MAX_RECOMMENDATIONS)
    top = [item.primary.evaluated.repo_id for item in diversified]
    if top != [item.evaluated.repo_id for item in state.recommendations]:
        raise ValueError("Rebuilt top 5 differs from the workflow's recommendations")
    if state.shadow is None or list(state.shadow.base_top5) != [
        item.plan.plan_id for item in state.recommendations if item.plan is not None
    ]:
        raise ValueError("Shadow report missing or not built from the shown result")

    # Logical-model buckets in ranked order: one main slot per identity.
    bucket_order: list[str] = []
    bucket_owner: dict[str, str] = {}
    for plan in ranked:
        if plan.assessment.rejected:
            continue
        key = model_identity_key(plan.plan.model_identity)
        if key not in bucket_owner:
            bucket_order.append(key)
            bucket_owner[key] = plan.evaluated.repo_id
    selected_keys = [model_identity_key(item.primary.plan.model_identity) for item in diversified]

    rows = []
    for evaluated in state.evaluated_candidates:
        repo = evaluated.repo_id
        plans = [plan for plan in ranked if plan.evaluated.repo_id == repo]
        viable = [plan for plan in plans if not plan.assessment.rejected]
        row: dict[str, Any] = {
            "repo_id": repo,
            "plans": len(plans),
            "viable_plans": len(viable),
        }
        if evaluated.failed:
            row |= {"stage": "inspection_failed", "detail": evaluated.warnings}
        elif not plans:
            row |= {"stage": "no_plans",
                    "detail": evaluated.warnings or ["No execution plan was generated."]}
        elif not viable:
            codes = sorted({c.code.value for p in plans for c in p.assessment.hard_constraints})
            row |= {"stage": "all_plans_rejected", "detail": codes,
                    "messages": sorted({c.message for p in plans
                                        for c in p.assessment.hard_constraints})}
        else:
            key = model_identity_key(viable[0].plan.model_identity)
            owner = bucket_owner[key]
            bucket_rank = bucket_order.index(key) + 1
            row |= {"identity": key, "bucket_rank": bucket_rank,
                    "best_plan_rank": ranked.index(viable[0]) + 1}
            if key in selected_keys:
                position = selected_keys.index(key) + 1
                if owner == repo:
                    row |= {"stage": "top5", "position": position,
                            "promoted_by_diversity": bucket_rank > len(selected_keys)}
                else:
                    row |= {"stage": "merged_into", "detail": owner, "position": position}
            elif owner != repo:
                row |= {"stage": "merged_into", "detail": owner}
            elif bucket_rank <= len(selected_keys):
                row |= {"stage": "displaced_by_diversity"}
            else:
                row |= {"stage": "below_cut"}
        rows.append(row)

    return {
        "schema_version": 1,
        "mode": "metadata_inspection",
        "hardware": hardware.model_dump(mode="json"),
        "answers": answers.model_dump(mode="json"),
        "pool_counts": pool["counts"],
        "evaluated": len(state.evaluated_candidates),
        "ranked_plans": len(ranked),
        "logical_models": len(bucket_order),
        "top5": top,
        "shadow": state.shadow.model_dump(mode="json"),
        "candidates": rows,
        "workflow_warnings": state.warnings,
        "workflow_errors": state.errors,
        "no_results_reason": state.no_results_reason,
        "limitations": [
            "One frozen search sample; not Hub recall, model quality or speed.",
            "Inspection used live Hub metadata on the trace date, not the capture date.",
            "Synthetic hardware profiles are scenarios, not physical validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--hardware-kind", choices=("measured", "synthetic"), required=True)
    parser.add_argument("--capture", type=Path, required=True,
                        help="A pool audit JSON written by audit_recommendation_pool")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    answers = UserAnswers.model_validate_json(args.answers.read_text(encoding="utf-8"))
    hardware = HardwareProfile.model_validate_json(args.hardware.read_text(encoding="utf-8"))
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    report = trace(answers, hardware, capture["queries"])
    report |= {"hardware_kind": args.hardware_kind,
               "capture_started_at": capture["capture_started_at"],
               "traced_at": datetime.now(tz=UTC).isoformat()}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    summary: dict[str, int] = {}
    for row in report["candidates"]:
        summary[row["stage"]] = summary.get(row["stage"], 0) + 1
    print(json.dumps({"top5": report["top5"], "stages": summary,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
