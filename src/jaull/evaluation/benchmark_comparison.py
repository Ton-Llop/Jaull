"""Compare benchmark records as complete execution plans, not isolated runtimes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.benchmarks import (
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkRecord,
)
from jaull.domain.execution_plans import ModelIdentity
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import LlamaCppRuntimeCapability, PyTorchRuntimeCapability
from jaull.evaluation.hardware_fingerprint import machine_fingerprint

_PREFERRED_METHODOLOGY_BY_RUNTIME = {
    "llama.cpp": "llama_bench_v1",
    "transformers": "transformers_isolated_inference_v2",
}


class BenchmarkPlanSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    label: str
    runtime: str
    artifact_format: str
    quantization_or_precision: str | None = None
    backend: str
    methodology: str | None = None
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    model_load_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    compatible_run_count: int = 1


class BenchmarkPlanMetricComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: BenchmarkMeasurementKind
    tokens: int
    baseline_label: str
    candidate_label: str
    baseline_tokens_per_second: float
    candidate_tokens_per_second: float
    relative_throughput: float
    warning: str | None = None


class BenchmarkComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_identity: ModelIdentity
    same_hardware: bool
    hardware: HardwareProfile | None = None
    plans: list[BenchmarkPlanSummary] = Field(default_factory=list)
    metrics: list[BenchmarkPlanMetricComparison] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    overall_score: None = None


def compare_benchmark_records(
    *,
    model_identity: ModelIdentity,
    records: list[BenchmarkRecord],
) -> BenchmarkComparison:
    selected_records, run_counts, selection_warnings = _latest_records_by_configuration(
        records
    )
    plans = [
        _plan_summary(record, compatible_run_count=run_counts[_configuration_key(record)])
        for record in selected_records
    ]
    warnings = [*selection_warnings, *_warnings(selected_records)]
    metrics = _metric_comparisons(selected_records, warnings=warnings)
    hardware = (
        selected_records[0].hardware
        if selected_records and _same_machine(selected_records)
        else None
    )
    return BenchmarkComparison(
        model_identity=model_identity,
        same_hardware=_same_machine(selected_records),
        hardware=hardware,
        plans=plans,
        metrics=metrics,
        warnings=warnings,
    )


def _latest_records_by_configuration(
    records: list[BenchmarkRecord],
) -> tuple[list[BenchmarkRecord], dict[tuple[str, ...], int], list[str]]:
    if not records:
        return [], {}, []

    filtered, warnings = _prefer_current_methodologies(records)
    grouped: dict[tuple[str, ...], list[BenchmarkRecord]] = {}
    for record in filtered:
        grouped.setdefault(_configuration_key(record), []).append(record)

    latest_records: list[BenchmarkRecord] = []
    run_counts: dict[tuple[str, ...], int] = {}
    for key, group in grouped.items():
        sorted_group = sorted(group, key=lambda item: item.identity.created_at)
        latest_records.append(sorted_group[-1])
        run_counts[key] = len(sorted_group)
    latest_records.sort(key=lambda item: item.identity.created_at)
    return latest_records, run_counts, warnings


def _prefer_current_methodologies(
    records: list[BenchmarkRecord],
) -> tuple[list[BenchmarkRecord], list[str]]:
    grouped: dict[tuple[str, ...], list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(_configuration_key(record, include_methodology=False), []).append(
            record
        )

    selected: list[BenchmarkRecord] = []
    omitted = False
    for group in grouped.values():
        preferred = _PREFERRED_METHODOLOGY_BY_RUNTIME.get(group[0].runtime.runtime.value)
        preferred_records = [
            record for record in group if record.observation.methodology == preferred
        ]
        if preferred_records:
            selected.extend(preferred_records)
            omitted = omitted or len(preferred_records) != len(group)
        else:
            selected.extend(group)
    warnings = (
        ["Older benchmark methodologies were omitted from the default comparison."]
        if omitted
        else []
    )
    return selected, warnings


def _metric_comparisons(
    records: list[BenchmarkRecord],
    *,
    warnings: list[str],
) -> list[BenchmarkPlanMetricComparison]:
    if len(records) < 2:
        return []
    baseline = records[0]
    baseline_label = _label(baseline)
    baseline_map = _measurement_map(baseline)
    comparisons: list[BenchmarkPlanMetricComparison] = []
    methodological_warning = _methodological_warning(warnings)
    for candidate in records[1:]:
        candidate_map = _measurement_map(candidate)
        candidate_label = _label(candidate)
        for key, baseline_measurement in baseline_map.items():
            candidate_measurement = candidate_map.get(key)
            if candidate_measurement is None:
                continue
            relative = (
                candidate_measurement.mean_tokens_per_second
                / baseline_measurement.mean_tokens_per_second
                if baseline_measurement.mean_tokens_per_second > 0
                else 0.0
            )
            comparisons.append(
                BenchmarkPlanMetricComparison(
                    kind=key[0],
                    tokens=key[1],
                    baseline_label=baseline_label,
                    candidate_label=candidate_label,
                    baseline_tokens_per_second=(
                        baseline_measurement.mean_tokens_per_second
                    ),
                    candidate_tokens_per_second=(
                        candidate_measurement.mean_tokens_per_second
                    ),
                    relative_throughput=relative,
                    warning=methodological_warning,
                )
            )
    return comparisons


def _warnings(records: list[BenchmarkRecord]) -> list[str]:
    if not records:
        return ["No benchmark records were provided."]
    warnings: list[str] = []
    if not _same_machine(records):
        warnings.append("Machine identity differs.")
    if len({_artifact_format(record) for record in records}) > 1:
        warnings.append("Artifact formats differ; runtime effects are not isolated.")
    if len({_quantization_or_precision(record) for record in records}) > 1:
        warnings.append(
            "Quantization or precision differs; throughput includes artifact effects."
        )
    if len({record.observation.methodology for record in records}) > 1:
        warnings.append(
            "Benchmark methodologies differ; metrics are conceptually comparable only."
        )
    if len({record.request.repetitions for record in records}) > 1:
        warnings.append("Repetition counts differ.")
    return warnings


def _methodological_warning(warnings: list[str]) -> str | None:
    relevant = [
        warning
        for warning in warnings
        if "differ" in warning or "methodolog" in warning
    ]
    if not relevant:
        return None
    return " ".join(relevant)


def _plan_summary(
    record: BenchmarkRecord,
    *,
    compatible_run_count: int = 1,
) -> BenchmarkPlanSummary:
    return BenchmarkPlanSummary(
        benchmark_id=record.identity.benchmark_id,
        label=_label(record),
        runtime=record.runtime.runtime.value,
        artifact_format=_artifact_format(record),
        quantization_or_precision=_quantization_or_precision(record),
        backend=record.requested_backend.value,
        methodology=record.observation.methodology,
        peak_ram_bytes=record.observation.peak_ram_bytes,
        peak_vram_bytes=record.observation.peak_vram_bytes,
        model_load_seconds=record.observation.model_load_seconds,
        time_to_first_token_seconds=record.observation.time_to_first_token_seconds,
        compatible_run_count=compatible_run_count,
    )


def _same_machine(records: list[BenchmarkRecord]) -> bool:
    if not records:
        return False
    first = machine_fingerprint(records[0].hardware)
    return all(machine_fingerprint(record.hardware) == first for record in records)


def _configuration_key(
    record: BenchmarkRecord,
    *,
    include_methodology: bool = True,
) -> tuple[str, ...]:
    values = [
        *machine_fingerprint(record.hardware),
        record.artifact.repo_id,
        record.artifact.revision or "",
        record.artifact.filename,
        record.artifact.sha256 or "",
        record.artifact.format.lower(),
        record.artifact.quantization or "",
        _quantization_or_precision(record) or "",
        record.runtime.runtime.value,
        _runtime_version(record) or "",
        record.requested_backend.value,
        record.requested_device or "",
        record.gpu_layers.label,
        ",".join(str(item) for item in record.request.prefill_sizes),
        ",".join(str(item) for item in record.request.generation_sizes),
        str(record.request.repetitions),
    ]
    if include_methodology:
        values.append(record.observation.methodology or "")
    return tuple(values)

def _runtime_version(record: BenchmarkRecord) -> str | None:
    if record.llama_bench_capability is not None:
        return record.llama_bench_capability.version_text
    capability = record.runtime_capability
    if isinstance(capability, LlamaCppRuntimeCapability):
        return capability.version_text
    if isinstance(capability, PyTorchRuntimeCapability):
        parts = [capability.torch_version] if capability.torch_version is not None else []
        if capability.transformers_version is not None:
            parts.append(capability.transformers_version)
        return " / ".join(parts) if parts else None
    return None


def _measurement_map(
    record: BenchmarkRecord,
) -> dict[tuple[BenchmarkMeasurementKind, int], BenchmarkMeasurement]:
    return {
        (measurement.kind, measurement.tokens): measurement
        for measurement in record.observation.measurements
    }


def _artifact_format(record: BenchmarkRecord) -> str:
    return record.artifact.format.lower()


def _quantization_or_precision(record: BenchmarkRecord) -> str | None:
    return record.artifact.quantization or next(
        (flag.value for flag in record.runtime.flags if flag.name == "torch_dtype"),
        None,
    )


def _label(record: BenchmarkRecord) -> str:
    parts = [
        record.runtime.runtime.value,
        record.artifact.format.lower(),
        _quantization_or_precision(record) or "default",
        record.requested_backend.value,
    ]
    return " · ".join(parts)


__all__ = [
    "BenchmarkComparison",
    "BenchmarkPlanMetricComparison",
    "BenchmarkPlanSummary",
    "compare_benchmark_records",
]
