"""Fastest step 3: llama_bench_v2 on the exact GGUFs that already have IFEval.

Each artifact is benchmarked at the runtime and offload of the plan Jaull builds for
it on this machine, never at a hand-picked ``-ngl``: a different offload measures a
different deployment. Plans come from a search pinned to those artifacts, the same
controlled pool as the Quality pair audit. Records are stored before anything is
analysed, and ``--dry-run`` prints the commands without touching the GPU.

``--rounds N`` repeats the whole set N times, interleaved (A B C, A B C, ...), so
drift during the session shows up in every plan instead of only the last one;
``--summary`` writes each plan's tg128 per round with mean, spread and how far
apart the plans are relative to that spread.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.audit_quality_recompare import PinnedPairSearch

from jaull.advisor.service import AdvisorService
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import UserAnswers
from jaull.evaluation.quality_records import load_record
from jaull.recommendation.local_evidence import assess_speed
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
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds of idle GPU before each benchmark (cool-down)")
    parser.add_argument("--summary", type=Path,
                        help="Markdown file for the per-round tg128 table")
    args = parser.parse_args()

    advisor = AdvisorService.default()
    hardware = advisor.scan_hardware()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    answers = UserAnswers.model_validate(capture["input"]["answers"])
    records = [load_record(path) for path in args.records]
    state = pinned_search(advisor, hardware, answers, records)
    assert state.requirements is not None
    measured: dict[str, list[tuple[float, float | None, str]]] = {}
    for round_index in range(1, args.rounds + 1):
        for item in state.ranked_plans:
            flags = {flag.name: flag.value for flag in item.plan.runtime.flags}
            print(f"[round {round_index}/{args.rounds}] {item.evaluated.repo_id} / "
                  f"{item.plan.artifact.label}: -ngl {flags.get('--n-gpu-layers')} "
                  f"{flags.get('--device', '(only device)')}", flush=True)
            if args.dry_run:
                continue
            time.sleep(args.pause)
            # The product path: the same preparation and request the TUI uses.
            result = advisor.benchmark_plan_speed(item.plan)
            observation = result.record.observation
            print(f"  stored {result.persisted_path} success={observation.success}", flush=True)
            speed = assess_speed(item.plan, state.requirements, [result.record]).applicable
            if speed is not None:
                print(f"  tg128 {speed.tokens_per_second:.1f} tok/s", flush=True)
                measured.setdefault(item.evaluated.repo_id, []).append((
                    speed.tokens_per_second, speed.stddev_tokens_per_second,
                    speed.benchmark_id,
                ))
    if args.dry_run:
        return
    if args.summary is not None:
        args.summary.write_text(_summary(measured, args), encoding="utf-8")
        print(f"summary written to {args.summary}", flush=True)
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


def _summary(measured: dict[str, list[tuple[float, float | None, str]]],
             args: argparse.Namespace) -> str:
    rounds = max((len(v) for v in measured.values()), default=0)
    lines = [
        f"# tg128 noise campaign ({datetime.now(UTC):%Y-%m-%d %H:%M} UTC)", "",
        f"{args.rounds} interleaved rounds, {args.pause:.0f} s cool-down before each run; "
        "llama_bench_v2, 5 repetitions per run. Values in tok/s.", "",
        "| Model | " + " | ".join(f"R{i}" for i in range(1, rounds + 1))
        + " | Mean | Min-max | Spread % | Within-run sd |",
        "|---|" + "---|" * (rounds + 4),
    ]
    means = {}
    for repo, runs in measured.items():
        values = [v for v, _, _ in runs]
        mean = statistics.fmean(values)
        means[repo] = (mean, min(values), max(values))
        sds = [sd for _, sd, _ in runs if sd is not None]
        cells = [f"{v:.1f}" for v in values] + [""] * (rounds - len(values))
        sd = f"{statistics.fmean(sds):.2f}" if sds else "-"
        lines.append(
            f"| {repo} | " + " | ".join(cells) + f" | {mean:.1f} | "
            f"{min(values):.1f}-{max(values):.1f} | "
            f"{(max(values) - min(values)) / mean:.1%} | {sd} |"
        )
    lines += ["", "## Pairs", ""]
    repos = sorted(means, key=lambda r: -means[r][0])
    for i, a in enumerate(repos):
        for b in repos[i + 1:]:
            overlap = means[b][2] >= means[a][1]
            lines.append(
                f"- {a} vs {b}: {means[a][0] / means[b][0] - 1:+.1%} on the mean; ranges "
                + ("**overlap**: not distinguishable from this noise." if overlap
                   else "separate.")
            )
    lines += ["", "## Benchmark records", ""]
    lines += [f"- {repo}: " + ", ".join(bid for _, _, bid in runs)
              for repo, runs in measured.items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
