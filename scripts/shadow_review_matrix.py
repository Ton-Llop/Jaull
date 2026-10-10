"""Step 6 review matrix: replay captured searches across hardware, task and priority.

Metadata only. Searches replay frozen Hub queries, quality comes read-only from
the local store, and nothing is downloaded or run. For every combination it
checks that stored quality leaves the active order untouched, then reports what
the shadow policy would change and why each remaining plan keeps its place.
Synthetic hardware profiles are named as such; they validate wiring, not physics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.audit_recommendation_tail import ReplaySearchClient

from jaull.advisor.service import AdvisorService
from jaull.application.recommendation.service import recommend
from jaull.artifacts.service import ArtifactService
from jaull.domain.candidates import ModelCandidate, SearchQuery
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import RecommendationPriority, UserAnswers
from jaull.workflow.orchestrator import run_workflow

NIGHT = Path(".codex-night")
CAPTURES = {"chat": "pool-live-chat-20261007.json", "code": "pool-live-code-20261007.json"}
HARDWARE = {
    "rtx2060": "pool-tail-chat-2060-20261007-v2.json",
    "rtx4060": "pool-tail-chat-4060-20261007-v2.json",
    "cpu": "pool-tail-chat-cpu-20261007-v2.json",
}
PRIORITIES = (RecommendationPriority.QUALITY, RecommendationPriority.SPEED,
              RecommendationPriority.BALANCED)
#: Fallback probes: requests no audited suite covers must claim nothing.
#: Other use cases run queries no capture holds, so they are covered by tests instead.
FALLBACK = (
    ("chat", {"languages": ["Spanish"]}, "chat in Spanish"),
    ("chat", {"languages": ["English", "Spanish"]}, "chat in English and Spanish"),
    ("code", {"languages": ["Spanish"]}, "code in Spanish"),
)


class PartialReplay(ReplaySearchClient):
    """A probe may add a query the capture never ran: it finds nothing, and says so."""

    def __init__(self, observations: list[dict[str, Any]]) -> None:
        super().__init__(observations)
        self.missing: list[str] = []

    def search(self, query: SearchQuery) -> list[ModelCandidate]:
        try:
            return super().search(query)
        except ValueError:
            self.missing.append(query.label)
            return []


def require(condition: bool, message: str) -> None:
    """An audit claim, checked even under ``python -O``."""
    if not condition:
        raise RuntimeError(message)


def run_case(advisor: AdvisorService, hardware: HardwareProfile, capture: dict[str, Any],
             answers: UserAnswers) -> dict[str, Any]:
    replay = PartialReplay(capture["queries"])
    services = dataclasses.replace(advisor.services, search_client=replay)
    context = advisor._plan_ranking_context(hardware)
    state = run_workflow(answers, hardware, services, plan_context=context)
    require(state.completed and state.requirements is not None,
            f"Workflow failed: {state.errors} {state.no_results_reason}")
    if not state.recommendations:
        return {"no_results": state.no_results_reason, "queries_not_captured": replay.missing}
    require(state.shadow is not None, "Results without a shadow report: the policy raised")
    assert state.requirements is not None and state.shadow is not None
    # The gate: only Quality (activated in Step 6) may follow stored quality, and
    # exactly as its policy proposes; every other priority keeps the base order.
    without = recommend(
        state.evaluated_candidates, state.requirements, hardware=hardware,
        capability_analyzer=services.capability_analyzer,
        plan_context=dataclasses.replace(context, quality_records=()),
    )
    shadow = state.shadow
    active = [rec.plan.plan_id for rec in state.recommendations if rec.plan is not None]
    expected = (list(shadow.shadow_top5) if answers.priority is RecommendationPriority.QUALITY
                else [rec.plan.plan_id for rec in without if rec.plan is not None])
    require(active == expected, "The active order is not what its priority's policy gives")
    names = {item.plan.plan_id: f"{item.evaluated.repo_id} / {item.plan.artifact.label}"
             for item in state.ranked_plans}
    return {
        "top5": [rec.repo_id for rec in state.recommendations],
        "pool_plans": len(state.ranked_plans),
        "applicable_quality": sum(
            item.assessment.quality.applicable is not None for item in state.ranked_plans
        ),
        "shadow_applied": shadow.applied,
        "groups": len(shadow.groups),
        "moves": [{"plan": names.get(m.plan_id, m.plan_id), "from": m.base_index + 1,
                   "to": m.shadow_index + 1, "rule": m.rule} for m in shadow.moves],
        "top5_changes": [{"plan": names.get(c.plan_id, c.plan_id), "change": c.change,
                          "moved": c.moved} for c in shadow.top5_changes],
        "first_fallback_reasons": dict(Counter(
            f.reasons[0] if f.reasons else "none" for f in shadow.fallback
        ).most_common()),
        "warnings": state.warnings,
        "queries_not_captured": replay.missing,
        "active_order_follows_policy": True,
        "active_changed_by_quality": active != [
            rec.plan.plan_id for rec in without if rec.plan is not None
        ],
    }


def matrix(advisor: AdvisorService) -> dict[str, Any]:
    captures = {task: json.loads((NIGHT / name).read_text(encoding="utf-8"))
                for task, name in CAPTURES.items()}
    profiles = {}
    for key, name in HARDWARE.items():
        source = json.loads((NIGHT / name).read_text(encoding="utf-8"))
        profiles[key] = (HardwareProfile.model_validate(source["hardware"]),
                         source["hardware_kind"])
    cases = []
    for hw_key, (hardware, kind) in profiles.items():
        for task, capture in captures.items():
            base = UserAnswers.model_validate(capture["input"]["answers"])
            probes = [(priority.value, {"priority": priority}) for priority in PRIORITIES]
            probes += [(label, {**change, "priority": RecommendationPriority.QUALITY})
                       for probe_task, change, label in FALLBACK if probe_task == task]
            for label, change in probes:
                answers = base.model_copy(update=change)
                print(f"{hw_key} / {task} / {label}", flush=True)
                cases.append({"hardware": hw_key, "hardware_kind": kind, "task": task,
                              "probe": label, "use_case": answers.use_case,
                              "languages": answers.languages, "priority": answers.priority}
                             | run_case(advisor, hardware, capture, answers))
    return {
        "at": datetime.now(tz=UTC).isoformat(), "mode": "replay_metadata_only",
        "captures": {task: c.get("capture_started_at") for task, c in captures.items()},
        "quality_records": len(advisor._stored_quality_records()),
        "cases": cases,
        "limitations": [
            "Two frozen search captures; not recall or a general quality verdict.",
            "rtx4060 and cpu are synthetic profiles; only measured hardware validates fit.",
            "Speed evidence never applies yet, so Speed and Balanced cannot reorder.",
            "Probes that add a query the capture never ran see no results for it.",
            "Quality reorders only plans whose exact GGUF bytes were measured.",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Step 6 shadow review matrix", "",
        f"Generated {report['at']} ({report['mode']}); {report['quality_records']} stored "
        "quality records. In every case the active order is what its priority's policy "
        "gives: Quality follows comparable measurements, Speed and Balanced keep the base order.",
        "", "| Hardware | Task | Probe | Applicable | Groups | Moves | Top-5 changes |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if "no_results" in case:
            lines.append(f"| {case['hardware']} | {case['task']} | {case['probe']} | no results "
                         "| | | |")
            continue
        hardware = case["hardware"] + (" (synthetic)" if case["hardware_kind"] == "synthetic"
                                       else "")
        lines.append(
            f"| {hardware} | {case['task']} | {case['probe']} | "
            f"{case['applicable_quality']}/{case['pool_plans']} | {case['groups']} | "
            f"{len(case['moves'])} | {len(case['top5_changes'])} |"
        )
    lines += ["", "## Why plans kept their place (first reason, all cases)", ""]
    totals: Counter[str] = Counter()
    for case in report["cases"]:
        totals.update(case.get("first_fallback_reasons", {}))
    lines += [f"- {count} x {reason}" for reason, count in totals.most_common()]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def forbidden(*_: Any, **__: Any) -> None:
    raise RuntimeError("The review matrix must not download weights or run an evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON; a .md is written beside")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with patch.object(ArtifactService, "download", forbidden), patch.object(
        AdvisorService, "run_quality_evaluation", forbidden,
    ):
        report = matrix(AdvisorService.default())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        handle.write("\n")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
