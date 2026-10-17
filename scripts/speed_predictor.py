"""Experimental tg128 predictor with strict offline llama_bench_v2 validation gates.

Nothing in Jaull reads this: no ranking, no screen. Three steps, run in this
order so that a validation measurement never calibrates its own prediction:

  predict  freeze predictions for target GGUFs from the calibration records
  measure  run llama_bench_v2 on the targets, at the plan Jaull builds for them
  score    compare the frozen predictions with measurements taken after freezing

Scope: dense models, full GPU offload, llama_bench_v2, one GPU and one
llama.cpp build. Three predictors are scored side by side:

  reference  bandwidth / GGUF bytes. Idealised: the file is not the bytes read
             from VRAM per token, so this is neither a ceiling nor a prediction.
  llmfit     reference x 0.55, the default efficiency llmfit uses.
  additive   seconds per token = GGUF bytes / bandwidth + c. ``c`` is an
             empirical residual fitted on this machine and protocol; it may
             absorb kernels, architecture or quantization, and is not shown to
             be a fixed per-token cost.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Spec-sheet memory bandwidth: data rate x bus width / 8. Unlisted GPU: no prediction.
BANDWIDTH_BYTES_PER_SECOND = {
    # NVIDIA's reference specifications, not measured bandwidth:
    # https://www.nvidia.com/en-gb/geforce/graphics-cards/rtx-2060
    "NVIDIA GeForce RTX 2060": 336e9,  # 14 Gbps GDDR6, 192-bit
    # https://marketplace.nvidia.com/en-us/consumer/graphics-cards/zotac-gaming-geforce-rtx-4060-twin-edge-oc-white-edition/
    "NVIDIA GeForce RTX 4060": 272e9,  # 17 Gbps GDDR6, 128-bit
}
LLMFIT_EFFICIENCY = 0.55


@dataclasses.dataclass(frozen=True)
class Point:
    """One artifact's measured tg128 on one machine and protocol."""

    sha256: str
    filename: str
    size_bytes: int
    median_tps: float
    benchmark_ids: tuple[str, ...]
    cohort: str = ""

    def __post_init__(self) -> None:
        if self.size_bytes <= 0 or not math.isfinite(self.median_tps) or self.median_tps <= 0:
            raise ValueError("A calibration point needs positive bytes and finite positive tok/s")


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def fit_residual(points: list[Point], bandwidth: float) -> float:
    """Mean of measured seconds per token minus bytes / bandwidth."""
    if not points:
        raise ValueError("No calibration points")
    if not _positive(bandwidth):
        raise ValueError("Bandwidth must be finite and positive")
    if len({p.sha256 for p in points}) != len(points):
        raise ValueError("Calibration artifacts must be distinct")
    if len({p.cohort for p in points}) != 1:
        raise ValueError("Calibration points must share one speed cohort")
    return statistics.fmean(1 / p.median_tps - p.size_bytes / bandwidth for p in points)


def predict(size_bytes: int, bandwidth: float, residual: float) -> dict[str, float]:
    if size_bytes <= 0 or not _positive(bandwidth) or not math.isfinite(residual):
        raise ValueError("Prediction inputs must be finite with positive bytes and bandwidth")
    seconds = size_bytes / bandwidth + residual
    if not _positive(seconds):
        raise ValueError("The fitted residual predicts a non-positive time; model is inapplicable")
    reference = bandwidth / size_bytes
    return {
        "reference": reference,
        "llmfit": reference * LLMFIT_EFFICIENCY,
        "additive": 1 / seconds,
    }


def relative_error(predicted: float, measured: float) -> float:
    if not _positive(predicted) or not _positive(measured):
        raise ValueError("Error requires finite positive predicted and measured tok/s")
    return predicted / measured - 1


def leave_one_out(points: list[Point], bandwidth: float) -> list[tuple[Point, float]]:
    """Additive prediction of each point from a residual fitted on the others."""
    return [(p, predict(p.size_bytes, bandwidth,
                        fit_residual([q for q in points if q is not p], bandwidth))["additive"])
            for p in points]


def comparable_points(records: list[Any], gpu: str, build: int, *,
                      after: datetime | None = None,
                      shas: set[str] | None = None,
                      cohort: str | None = None) -> list[Point]:
    """Median tg128 per artifact over successful llama_bench_v2 full-offload calls."""
    from jaull.domain.benchmarks import LLAMA_BENCH_V2, BenchmarkMeasurementKind
    from jaull.domain.hardware import ComputeBackend
    from jaull.recommendation.local_evidence import fastest_record_blockers, speed_cohort

    calls: dict[str, dict[str, tuple[Any, float]]] = {}
    # A repeated ID with different contents is not resolved by input order.
    by_id: dict[str, Any] = {}
    for record in records:
        key = record.identity.benchmark_id
        if key in by_id and by_id[key] != record:
            raise ValueError(f"Conflicting benchmark ID: {key}")
        by_id[key] = record
    cohorts = set()
    for record in by_id.values():
        observation = record.observation
        tg = [m.mean_tokens_per_second for m in observation.measurements
              if m.kind is BenchmarkMeasurementKind.GENERATION and m.tokens == 128]
        protocol = observation.protocol
        if (not observation.success or observation.methodology != LLAMA_BENCH_V2
                or not record.gpu_layers.full_offload or len(tg) != 1
                or record.requested_backend is not ComputeBackend.CUDA
                or len(record.hardware.gpus) != 1 or record.hardware.gpus[0].name != gpu
                or protocol is None or protocol.build_number != build
                or fastest_record_blockers(record)
                or record.artifact.sha256 is None or record.artifact.size_bytes is None
                or record.artifact.size_bytes <= 0 or not _positive(tg[0])
                or (after is not None and record.identity.created_at <= after)
                or (shas is not None and record.artifact.sha256 not in shas)):
            continue
        key = speed_cohort(record)
        if cohort is not None and key != cohort:
            continue
        cohorts.add(key)
        calls.setdefault(record.artifact.sha256, {})[record.identity.benchmark_id] = (
            record, tg[0])
    if len(cohorts) > 1:
        raise ValueError("Multiple speed cohorts: select one explicitly; never pool environments")
    points = []
    for sha, by_id in calls.items():
        first = next(iter(by_id.values()))[0]
        if any(r.artifact.size_bytes != first.artifact.size_bytes for r, _ in by_id.values()):
            raise ValueError(f"Conflicting artifact sizes for {sha}")
        points.append(Point(
            sha256=sha, filename=first.artifact.filename, size_bytes=first.artifact.size_bytes,
            median_tps=statistics.median(tps for _, tps in by_id.values()),
            benchmark_ids=tuple(sorted(by_id)),
            cohort=speed_cohort(first),
        ))
    return sorted(points, key=lambda p: p.size_bytes)


def _records(advisor: Any) -> list[Any]:
    return [advisor.load_benchmark_record(i) for i in advisor.list_benchmark_ids()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return None
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          check=True).stdout.strip()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    # Exclusive creation protects both predictions and reports, including input/output aliases.
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def calibration_cohort(frozen: dict[str, Any], records: list[Any]) -> str:
    """Recover the cohort from frozen calibration IDs, including legacy freezes.

    No timestamp or digest is added to an old file: this is a derived check on
    its immutable source records, not a repair of the historical prediction.
    """
    expected = {i for p in frozen["calibration"] for i in p["benchmark_ids"]}
    selected = [r for r in records if r.identity.benchmark_id in expected]
    points = comparable_points(selected, frozen["gpu"], frozen["llama_cpp_build"])
    if {i for p in points for i in p.benchmark_ids} != expected or not points:
        raise ValueError("Frozen calibration records are missing or fail protocol checks")
    before = datetime.fromisoformat(frozen["frozen_at"])
    if any(r.identity.created_at >= before for r in selected):
        raise ValueError("Calibration must predate the frozen prediction")
    actual = {p.sha256: p for p in points}
    for p in frozen["calibration"]:
        point = actual.get(p["sha256"])
        if (point is None or point.size_bytes != p["size_bytes"]
                or point.median_tps != p["median_tps"]
                or point.benchmark_ids != tuple(sorted(p["benchmark_ids"]))):
            raise ValueError("Frozen calibration differs from its source records")
    cohort = points[0].cohort
    if frozen.get("speed_cohort", cohort) != cohort:
        raise ValueError("Frozen speed cohort contradicts its calibration")
    return cohort


def cmd_predict(args: argparse.Namespace) -> None:
    from jaull.advisor.service import AdvisorService

    if args.out.exists():
        raise SystemExit(f"{args.out} exists: a frozen prediction is never overwritten.")
    advisor = AdvisorService.default()
    hardware = advisor.scan_hardware()
    if len(hardware.gpus) != 1 or hardware.gpus[0].name not in BANDWIDTH_BYTES_PER_SECOND:
        raise ValueError("Prediction requires one GPU with a documented bandwidth")
    gpu = hardware.gpus[0].name
    bandwidth = BANDWIDTH_BYTES_PER_SECOND[gpu]
    records = _records(advisor)
    calibration = comparable_points(records, gpu, args.build, cohort=args.cohort)
    if len(calibration) < 2 or any(len(p.benchmark_ids) < 3 for p in calibration):
        raise ValueError("Calibration needs at least two distinct artifacts with three calls each")
    from jaull.recommendation.local_evidence import speed_cohort

    template = next(r for r in records if r.identity.benchmark_id in calibration[0].benchmark_ids)
    if speed_cohort(template.model_copy(update={"hardware": hardware})) != calibration[0].cohort:
        raise ValueError("Calibration environment differs from this machine's current environment")
    residual = fit_residual(calibration, bandwidth)
    targets: list[dict[str, Any]] = []
    for spec in args.targets:
        repo_id, filename = spec.split(":", 1)
        path = args.models / repo_id / filename
        sha = _sha256(path)
        if sha in {p.sha256 for p in calibration}:
            raise SystemExit(f"{filename} is a calibration artifact, not a new plan.")
        if sha in {t["sha256"] for t in targets}:
            raise ValueError("Validation targets must be distinct artifacts")
        size = path.stat().st_size
        targets.append({"repo_id": repo_id, "filename": filename, "sha256": sha,
                        "size_bytes": size, "predicted": predict(size, bandwidth, residual)})
    git_status = _git("status", "--porcelain")
    frozen = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(git_status) if git_status is not None else None,
        "gpu": gpu, "bandwidth_bytes_per_second": bandwidth, "llama_cpp_build": args.build,
        "speed_cohort": calibration[0].cohort,
        "hardware": hardware.model_dump(mode="json"),
        "script_sha256": _sha256(Path(__file__)),
        "residual_seconds_per_token": residual,
        "calibration": [dataclasses.asdict(p) for p in calibration],
        "calibration_leave_one_out": [
            {"filename": p.filename, "predicted": pred, "measured": p.median_tps,
             "error": relative_error(pred, p.median_tps)}
            for p, pred in leave_one_out(calibration, bandwidth)],
        "targets": targets,
    }
    _write_new(args.out, frozen)
    print(f"c = {residual * 1000:.3f} ms/token from {len(calibration)} plans")
    for target in targets:
        print(f"{target['filename']}: " + ", ".join(
            f"{k} {v:.1f}" for k, v in target["predicted"].items()))
    print(f"frozen to {args.out}")


def cmd_measure(args: argparse.Namespace) -> None:
    from scripts.audit_quality_recompare import PinnedPairSearch

    from jaull.advisor.service import AdvisorService
    from jaull.domain.requirements import UserAnswers
    from jaull.workflow.orchestrator import run_workflow

    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    advisor = AdvisorService.default()
    hardware = advisor.scan_hardware()
    records = _records(advisor)
    cohort = calibration_cohort(frozen, records)
    from jaull.recommendation.local_evidence import speed_cohort

    template = next(r for r in records
                    if r.identity.benchmark_id in frozen["calibration"][0]["benchmark_ids"])
    if speed_cohort(template.model_copy(update={"hardware": hardware})) != cohort:
        raise ValueError("Current environment differs from frozen calibration; do not measure")
    if args.calls < 3 or not math.isfinite(args.pause) or args.pause < 0:
        raise ValueError("Measure requires at least three calls and a finite non-negative pause")
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    answers = UserAnswers.model_validate(capture["input"]["answers"])
    analyses = {}
    for target in frozen["targets"]:
        analysis = advisor.inspect_model(target["repo_id"])
        exact = [v for v in analysis.classification.gguf_variants
                 if len(v.files) == 1 and v.files[0].path == target["filename"]
                 and v.sha256 == target["sha256"]]
        if len(exact) != 1 or target["repo_id"] in analyses:
            raise SystemExit(f"Cannot pin {target['repo_id']}/{target['filename']} alone.")
        analyses[target["repo_id"]] = analysis.model_copy(update={
            "classification": analysis.classification.model_copy(
                update={"gguf_variants": exact})})
    services = dataclasses.replace(
        advisor.services, search_client=PinnedPairSearch(analyses),
        inspect_model=lambda repo_id, **_: analyses[repo_id], model_analysis_cache=None)
    state = run_workflow(answers, hardware, services,
                         plan_context=advisor._plan_ranking_context(hardware))
    plans: dict[str, Any] = {}
    for item in state.ranked_plans:
        flags = {flag.name: flag.value for flag in item.plan.runtime.flags}
        sha = item.plan.artifact.sha256
        if sha is not None and item.plan.runtime_family.value == "llama.cpp" and sha in {
                t["sha256"] for t in frozen["targets"]}:
            if flags.get("--n-gpu-layers") not in {"-1", "all"}:
                print(f"skip {item.plan.artifact.filename}: plan is not full offload")
                continue
            plans.setdefault(sha, item.plan)
    missing = {t["sha256"] for t in frozen["targets"]} - set(plans)
    print(f"plans: {sorted(plans)}; missing: {sorted(missing)}", flush=True)
    if missing or not state.completed:
        raise ValueError("Not every frozen target has a full-offload plan; no GPU work started")
    for round_index in range(1, args.calls + 1):
        for plan in plans.values():
            print(f"[call {round_index}/{args.calls}] {plan.artifact.filename}", flush=True)
            if args.dry_run:
                continue
            time.sleep(args.pause)
            result = advisor.benchmark_plan_speed(plan)
            if not comparable_points([result.record], frozen["gpu"], frozen["llama_cpp_build"],
                                     cohort=cohort, shas={plan.artifact.sha256}):
                raise ValueError("Stored benchmark differs from the frozen protocol; stopping")
            tg = [m.mean_tokens_per_second for m in result.record.observation.measurements
                  if m.source_label == "tg128"]
            print(f"  stored {result.persisted_path} success="
                  f"{result.record.observation.success} tg128={tg}", flush=True)


def score_rows(frozen: dict[str, Any], measured: list[Point]) -> list[dict[str, Any]]:
    if not frozen["targets"]:
        raise ValueError("No frozen targets to validate")
    if len({t["sha256"] for t in frozen["targets"]}) != len(frozen["targets"]):
        raise ValueError("Validation targets must be distinct artifacts")
    calibration_shas = {p["sha256"] for p in frozen.get("calibration", [])}
    by_sha = {p.sha256: p for p in measured}
    rows = []
    for target in frozen["targets"]:
        if target["sha256"] in calibration_shas:
            raise ValueError("A validation target cannot also be a calibration artifact")
        point = by_sha.get(target["sha256"])
        if point is not None and (
            point.size_bytes != target["size_bytes"]
            or point.cohort != frozen.get("speed_cohort", point.cohort)
        ):
            raise ValueError("Measured artifact size or cohort differs from the frozen prediction")
        sufficient = point is not None and len(point.benchmark_ids) >= 3
        rows.append({
            "filename": target["filename"], "size_bytes": target["size_bytes"],
            "measured": point.median_tps if point else None,
            "calls": len(point.benchmark_ids) if point else 0,
            "status": "comparable" if sufficient else "insufficient_calls" if point else "missing",
            "benchmark_ids": list(point.benchmark_ids) if point else [],
            "predicted": target["predicted"],
            "error": ({k: relative_error(v, point.median_tps)
                       for k, v in target["predicted"].items()}
                      if point is not None and sufficient else None),
        })
    return rows


def cmd_score(args: argparse.Namespace) -> None:
    from jaull.advisor.service import AdvisorService

    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    advisor = AdvisorService.default()
    records = _records(advisor)
    cohort = calibration_cohort(frozen, records)
    measured = comparable_points(
        records, frozen["gpu"], frozen["llama_cpp_build"], cohort=cohort,
        after=datetime.fromisoformat(frozen["frozen_at"]),
        shas={t["sha256"] for t in frozen["targets"]})
    rows = score_rows({**frozen, "speed_cohort": cohort}, measured)
    _write_new(args.out, {"frozen": str(args.frozen), "frozen_sha256": _sha256(args.frozen),
                         "speed_cohort": cohort,
                         "complete": all(r["status"] == "comparable" for r in rows), "rows": rows})
    for row in rows:
        errors = row["error"] or {}
        print(f"{row['filename']}: measured {row['measured']} over {row['calls']} calls; "
              + ", ".join(f"{k} {v:+.1%}" for k, v in errors.items()))
    print(f"scores written to {args.out}")
    if not all(row["status"] == "comparable" for row in rows):
        raise SystemExit("Validation incomplete: targets need three calls in the frozen cohort")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--targets", nargs="+", required=True, help="repo_id:filename")
    p.add_argument("--build", type=int, required=True, help="llama.cpp build number")
    p.add_argument("--cohort", help="Predeclared speed cohort; required if the store has several")
    p.add_argument("--models", type=Path,
                   default=Path.home() / ".local/share/jaull/models")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_predict)
    m = sub.add_parser("measure")
    m.add_argument("--frozen", type=Path, required=True)
    m.add_argument("--capture", type=Path, required=True,
                   help="A pool capture whose answers define the request")
    m.add_argument("--calls", type=int, default=3)
    m.add_argument("--pause", type=float, default=60.0)
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(func=cmd_measure)
    s = sub.add_parser("score")
    s.add_argument("--frozen", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
