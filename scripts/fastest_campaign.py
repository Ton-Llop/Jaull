"""Fastest step 3: llama_bench_v2 on the exact GGUFs that already have IFEval.

Each artifact is benchmarked at the runtime and offload of the plan Jaull builds for
it on this machine, never at a hand-picked ``-ngl``: a different offload measures a
different deployment. Plans come from a search pinned to those artifacts, the same
controlled pool as the Quality pair audit. Records are stored before anything is
analysed, and ``--dry-run`` prints the commands without touching the GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from scripts.audit_quality_recompare import PinnedPairSearch

from jaull.advisor.service import AdvisorService
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import UserAnswers
from jaull.evaluation.quality_records import load_record
from jaull.workflow.orchestrator import run_workflow
from jaull.workflow.state import RecommendationWorkflowState


def pinned_search(advisor: AdvisorService, hardware: HardwareProfile, answers: UserAnswers,
                  records: list[dict[str, Any]]) -> RecommendationWorkflowState:
    """The pool Search builds when it can only find these exact artifacts."""
    analyses: dict[str, ModelAnalysis] = {}
    for record in records:
        manifest = json.loads(record["provenance"]["evidence"]["verified-artifact.json"])
        analysis = advisor.inspect_model(manifest["repo_id"])
        exact = [v for v in analysis.classification.gguf_variants
                 if v.sha256 == record["identity"]["artifact_sha256"]
                 and len(v.files) == 1 and v.files[0].path == manifest["filename"]]
        if len(exact) != 1:
            raise ValueError(f"Metadata does not identify {manifest['repo_id']} exactly")
        analyses[manifest["repo_id"]] = analysis.model_copy(update={
            "classification": analysis.classification.model_copy(update={"gguf_variants": exact}),
        })

    def inspect(repo_id: str, **_: Any) -> ModelAnalysis:
        return analyses[repo_id]

    services = dataclasses.replace(advisor.services, search_client=PinnedPairSearch(analyses),
                                   inspect_model=inspect, model_analysis_cache=None)
    state = run_workflow(answers, hardware, services,
                         plan_context=advisor._plan_ranking_context(hardware))
    if not state.completed or not state.ranked_plans:
        raise RuntimeError(f"Pinned search produced no pool: {state.errors}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True,
                        help="Stored IFEval records whose exact artifacts to benchmark")
    parser.add_argument("--capture", type=Path, required=True,
                        help="A pool capture whose answers define the request")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    advisor = AdvisorService.default()
    hardware = advisor.scan_hardware()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    answers = UserAnswers.model_validate(capture["input"]["answers"])
    records = [load_record(path) for path in args.records]
    state = pinned_search(advisor, hardware, answers, records)
    for item in state.ranked_plans:
        flags = {flag.name: flag.value for flag in item.plan.runtime.flags}
        print(f"{item.evaluated.repo_id} / {item.plan.artifact.label}: "
              f"-ngl {flags.get('--n-gpu-layers')} {flags.get('--device', '(only device)')}",
              flush=True)
        if not args.dry_run:
            # The product path: the same preparation and request the TUI uses.
            result = advisor.benchmark_plan_speed(item.plan)
            observation = result.record.observation
            print(f"  stored {result.persisted_path} success={observation.success}", flush=True)
    if args.dry_run:
        return
    # Analysis only after every record is stored: rebuild the pool and read what applies.
    after = pinned_search(advisor, hardware, answers, records)
    for item in after.ranked_plans:
        speed, quality = item.assessment.speed, item.assessment.quality.applicable
        applied = speed.applicable
        print(f"{item.evaluated.repo_id}: speed={speed.applicability} "
              f"tg128={applied.tokens_per_second if applied else None} "
              f"quality={quality.value if quality else None}")
        if applied is None and speed.references:
            print("  blockers:", list(speed.references[0].blockers))


if __name__ == "__main__":
    main()
