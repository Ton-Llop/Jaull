"""Descriptive comparison of equivalent benchmark records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.benchmarks import (
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkRecord,
)
from jaull.domain.hardware import ComputeBackend


class BenchmarkConfigurationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    label: str
    backend: ComputeBackend
    device: str | None = None
    gpu_layers: str


class BenchmarkMetricComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: BenchmarkMeasurementKind
    tokens: int
    baseline_label: str
    candidate_label: str
    baseline_tokens_per_second: float
    candidate_tokens_per_second: float
    speedup: float


class BenchmarkAggregate(BaseModel):
    model_config = ConfigDict(frozen=True)

    equivalent: bool
    reason: str | None = None
    configurations: list[BenchmarkConfigurationSummary] = Field(default_factory=list)
    comparisons: list[BenchmarkMetricComparison] = Field(default_factory=list)


def aggregate_benchmark_records(records: list[BenchmarkRecord]) -> BenchmarkAggregate:
    if not records:
        return BenchmarkAggregate(equivalent=False, reason="No benchmark records.")

    configurations = [_configuration(record) for record in records]
    reason = _non_equivalence_reason(records)
    if reason is not None:
        return BenchmarkAggregate(
            equivalent=False,
            reason=reason,
            configurations=configurations,
        )
    if len(records) < 2:
        return BenchmarkAggregate(
            equivalent=True,
            configurations=configurations,
            comparisons=[],
        )

    baseline = next(
        (record for record in records if record.requested_backend is ComputeBackend.CPU),
        records[0],
    )
    baseline_label = _configuration_label(baseline)
    comparisons: list[BenchmarkMetricComparison] = []
    baseline_measurements = _measurement_map(baseline)
    for candidate in records:
        if candidate == baseline:
            continue
        candidate_measurements = _measurement_map(candidate)
        candidate_label = _configuration_label(candidate)
        for key, base_measurement in baseline_measurements.items():
            candidate_measurement = candidate_measurements.get(key)
            if candidate_measurement is None:
                continue
            speedup = (
                candidate_measurement.mean_tokens_per_second
                / base_measurement.mean_tokens_per_second
                if base_measurement.mean_tokens_per_second > 0
                else 0.0
            )
            comparisons.append(
                BenchmarkMetricComparison(
                    kind=key[0],
                    tokens=key[1],
                    baseline_label=baseline_label,
                    candidate_label=candidate_label,
                    baseline_tokens_per_second=base_measurement.mean_tokens_per_second,
                    candidate_tokens_per_second=(
                        candidate_measurement.mean_tokens_per_second
                    ),
                    speedup=speedup,
                )
            )
    return BenchmarkAggregate(
        equivalent=True,
        configurations=configurations,
        comparisons=comparisons,
    )


def _non_equivalence_reason(records: list[BenchmarkRecord]) -> str | None:
    first = records[0]
    for record in records[1:]:
        if _artifact_identity(record) != _artifact_identity(first):
            return "Benchmark records use different artifacts."
        if record.hardware != first.hardware:
            return "Benchmark records use different hardware snapshots."
        if record.request.repetitions != first.request.repetitions:
            return "Benchmark records use different repetition counts."
    return None


def _artifact_identity(record: BenchmarkRecord) -> tuple[str, str | None, str, str | None]:
    artifact = record.artifact
    return (
        artifact.repo_id,
        artifact.revision,
        artifact.filename,
        artifact.sha256,
    )


def _measurement_map(
    record: BenchmarkRecord,
) -> dict[tuple[BenchmarkMeasurementKind, int], BenchmarkMeasurement]:
    return {
        (measurement.kind, measurement.tokens): measurement
        for measurement in record.observation.measurements
    }


def _configuration(record: BenchmarkRecord) -> BenchmarkConfigurationSummary:
    return BenchmarkConfigurationSummary(
        benchmark_id=record.identity.benchmark_id,
        label=_configuration_label(record),
        backend=record.requested_backend,
        device=record.requested_device,
        gpu_layers=record.gpu_layers.label,
    )


def _configuration_label(record: BenchmarkRecord) -> str:
    if record.requested_backend is ComputeBackend.CPU:
        return "CPU"
    device = record.requested_device or record.requested_backend.value
    return f"{record.requested_backend.value.title()} {device} ngl {record.gpu_layers.label}"


__all__ = [
    "BenchmarkAggregate",
    "BenchmarkConfigurationSummary",
    "BenchmarkMetricComparison",
    "aggregate_benchmark_records",
]
