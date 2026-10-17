"""Results-only speed readout with distinct applicable and historical measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jaull.domain.benchmarks import BenchmarkMeasurementKind, BenchmarkRecord
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.requirements import UserRequirements
from jaull.domain.runtime import RuntimeName
from jaull.recommendation.local_evidence import assess_speed, matching_benchmark


@dataclass(frozen=True)
class SpeedReadout:
    label: str = ""
    details: str = "No applicable speed measurement for this plan."
    measured: bool = False
    source: Literal["unknown", "measured", "historical"] = "unknown"


def speed_readout(
    plan: ExecutionPlan | None,
    requirements: UserRequirements | None,
    records: Sequence[BenchmarkRecord] = (),
) -> SpeedReadout:
    if plan is None or requirements is None:
        return SpeedReadout()
    by_id: dict[str, BenchmarkRecord] = {}
    for record in records:
        key = record.identity.benchmark_id
        if key in by_id and by_id[key] != record:
            return SpeedReadout(details="Conflicting benchmark IDs; speed evidence withheld.")
        by_id[key] = record
    speed = assess_speed(plan, requirements, tuple(by_id.values()))
    if (measured := speed.applicable) is not None:
        observed = (
            f"Observed range {measured.observed_min:.1f}-{measured.observed_max:.1f} tok/s. "
            if measured.observed_min is not None and measured.observed_max is not None else ""
        )
        return SpeedReadout(
            f"Measured {measured.tokens_per_second:.1f} tok/s",
            f"tg128; {measured.run_count} calls. {observed}Not a confidence interval. "
            f"{measured.scope}\nBenchmark IDs: "
            + ", ".join(measured.benchmark_ids or (measured.benchmark_id,)),
            measured=True,
            source="measured",
        )
    if plan.runtime_family is RuntimeName.TRANSFORMERS:
        historical = matching_benchmark(plan, [
            record for record in by_id.values()
            if record.observation.exit_code == 0
            and (
                plan.artifact.sha256 is None or record.artifact.sha256 is None
                or plan.artifact.sha256 == record.artifact.sha256.lower()
            )
            and sum(
                m.kind is BenchmarkMeasurementKind.GENERATION and m.tokens == 128
                for m in record.observation.measurements
            ) == 1
        ])
        if historical is not None:
            generation = next(
                m for m in historical.observation.measurements
                if m.kind is BenchmarkMeasurementKind.GENERATION and m.tokens == 128
            )
            flags = ", ".join(f"{flag.name}={flag.value}" for flag in historical.runtime.flags)
            return SpeedReadout(
                f"Last benchmark {generation.mean_tokens_per_second:.1f} tok/s (Transformers)",
                "Historical measurement matched by metadata, not verified exact artifact bytes "
                "or current execution. Not used for ranking or a production-workload prediction.\n"
                f"tg128; {historical.request.repetitions} repetitions. "
                f"Protocol: {historical.observation.methodology}.\n"
                f"Recorded: {historical.identity.created_at.isoformat()}\n"
                f"Artifact: {historical.artifact.repo_id}@{historical.artifact.revision}\n"
                f"Backend: {historical.requested_backend.value}; "
                f"device: {historical.requested_device or 'unknown'}; {flags}\n"
                f"Benchmark ID: {historical.identity.benchmark_id}",
                source="historical",
            )
    history = "\n".join(
        f"Historical reference only: {ref.benchmark_id}; " + "; ".join(ref.blockers)
        for ref in speed.references
    )
    return SpeedReadout(details="\n".join(filter(None, (SpeedReadout().details, history))))
