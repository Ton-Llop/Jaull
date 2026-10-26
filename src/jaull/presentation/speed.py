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
    saved_runs: str = "No saved speed runs for this configuration on this machine."

    @property
    def scope_label(self) -> str:
        if self.measured:
            return "Comparable measurement"
        if self.source == "historical":
            return "Historical benchmark"
        return "No comparable measurement"

    @property
    def scope_note(self) -> str:
        if self.measured:
            return "Eligible for speed ranking; not a prediction for this workload."
        if self.source == "historical":
            return "Saved result; excluded from the current speed ranking."
        return "No matching measurement for the selected plan."


def historical_speed_benchmarks(
    plan: ExecutionPlan, records: Sequence[BenchmarkRecord],
) -> tuple[BenchmarkRecord, ...]:
    """Metadata-matched history, never proof of exact bytes or ranking applicability."""
    by_id: dict[str, BenchmarkRecord] = {}
    for record in records:
        key = record.identity.benchmark_id
        if key in by_id and by_id[key] != record:
            return ()
        by_id[key] = record
    return tuple(sorted(
        (
            record for record in by_id.values()
            if record.observation.exit_code == 0
            and (
                plan.artifact.sha256 is None or record.artifact.sha256 is None
                or plan.artifact.sha256.lower() == record.artifact.sha256.lower()
            )
            and sum(
                m.kind is BenchmarkMeasurementKind.GENERATION and m.tokens == 128
                for m in record.observation.measurements
            ) == 1
            and matching_benchmark(plan, [record]) is not None
        ),
        key=lambda record: (record.identity.created_at, record.identity.benchmark_id),
        reverse=True,
    ))


def _saved_runs_text(records: Sequence[BenchmarkRecord]) -> str:
    if not records:
        return SpeedReadout().saved_runs
    sections = []
    for record in records:
        observation = record.observation
        device = record.requested_device or (
            "no GPU device" if record.requested_backend.value == "cpu" else "unknown device"
        )
        lines = [
            f"{record.identity.created_at.isoformat()} · {record.runtime.runtime.value} · "
            f"{record.requested_backend.value} · {device}",
            f"Artifact: {record.artifact.repo_id}@{record.artifact.revision} / "
            f"{record.artifact.filename} · "
            f"{record.artifact.quantization or record.artifact.format}",
            f"Methodology: {observation.methodology}; {record.request.repetitions} repetitions",
        ]
        for measurement in observation.measurements:
            label = (
                "Prompt" if measurement.kind is BenchmarkMeasurementKind.PREFILL else "Generation"
            )
            lines.append(
                f"{label} {measurement.tokens}: {measurement.mean_tokens_per_second:.1f} "
                f"± {measurement.stddev_tokens_per_second:.1f} tok/s (stddev)"
            )
        for label, value in (
            ("Peak RAM", observation.peak_ram_bytes), ("Peak VRAM", observation.peak_vram_bytes),
        ):
            lines.append(f"{label}: {value / 1024**3:.2f} GiB" if value is not None
                         else f"{label}: unavailable")
        for label, seconds in (
            ("Model load", observation.model_load_seconds),
            ("Time to first token", observation.time_to_first_token_seconds),
            ("Generation latency", observation.generation_latency_seconds),
        ):
            if seconds is not None:
                lines.append(f"{label}: {seconds:.2f} s")
        lines.append(f"Record: {record.identity.benchmark_id}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


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
    history_records = historical_speed_benchmarks(plan, tuple(by_id.values()))
    saved_runs = _saved_runs_text(history_records)
    speed = assess_speed(plan, requirements, tuple(by_id.values()))
    if (measured := speed.applicable) is not None:
        latest = by_id[measured.benchmark_id]
        generation = next(
            m for m in latest.observation.measurements
            if m.kind is BenchmarkMeasurementKind.GENERATION and m.tokens == 128
        )
        observed = (
            f"Observed range {measured.observed_min:.1f}-{measured.observed_max:.1f} tok/s. "
            if measured.observed_min is not None and measured.observed_max is not None else ""
        )
        return SpeedReadout(
            f"Measured {generation.mean_tokens_per_second:.1f} tok/s",
            f"tg128; {measured.run_count} calls. {observed}Not a confidence interval. "
            f"Latest saved run: {latest.identity.benchmark_id}; "
            f"recorded {latest.identity.created_at.isoformat()}.\n"
            f"Ranking aggregate (median): {measured.tokens_per_second:.1f} tok/s. "
            f"{measured.scope}\nBenchmark IDs: "
            + ", ".join(measured.benchmark_ids or (measured.benchmark_id,)),
            measured=True,
            source="measured",
            saved_runs=saved_runs,
        )
    if plan.runtime_family in {RuntimeName.TRANSFORMERS, RuntimeName.LLAMA_CPP}:
        historical = matching_benchmark(plan, history_records)
        if historical is not None:
            generation = next(
                m for m in historical.observation.measurements
                if m.kind is BenchmarkMeasurementKind.GENERATION and m.tokens == 128
            )
            flags = ", ".join(f"{flag.name}={flag.value}" for flag in historical.runtime.flags)
            runtime_label = (
                "Transformers" if plan.runtime_family is RuntimeName.TRANSFORMERS else "llama.cpp"
            )
            return SpeedReadout(
                f"Last benchmark {generation.mean_tokens_per_second:.1f} tok/s ({runtime_label})",
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
                saved_runs=saved_runs,
            )
    history = "\n".join(
        f"Historical reference only: {ref.benchmark_id}; " + "; ".join(ref.blockers)
        for ref in speed.references
    )
    return SpeedReadout(
        details="\n".join(filter(None, (SpeedReadout().details, history))), saved_runs=saved_runs,
    )
