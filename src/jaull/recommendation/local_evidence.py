"""Eligibility of historical observations for a recommendation plan.

Matching is offline and does not depend on runtime installation/readiness.
Benchmark evidence describes its recorded microbenchmark, not the throughput
of the user's context/concurrency workload. Artifact metadata matching is not
cryptographic verification when the plan has no digest.
"""

from collections.abc import Sequence

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import BenchmarkMeasurementKind, BenchmarkRecord
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.evaluation.benchmark_comparison import preferred_benchmark_methodology
from jaull.evaluation.hardware_fingerprint import machine_fingerprint


def matching_experiment(
    plan: ExecutionPlan, records: Sequence[ExperimentRecord],
) -> ExperimentRecord | None:
    compatible = []
    for record in records:
        if not _common_matches(plan, record.artifact, record.runtime, record.hardware):
            continue
        assert plan.backend_selection is not None
        expected_backend = plan.backend_selection.selected_backend
        if record.preflight.execution_readiness.selection.selected_backend != expected_backend:
            continue
        observed = record.backend_trace.observed_backend
        if observed is not None and observed != expected_backend:
            continue
        if plan.memory_prediction is None or not _configuration_matches(
            plan.memory_prediction.inference_configuration,
            record.prediction.inference_configuration,
        ):
            continue
        if _flags(plan.runtime) != _flags(record.runtime):
            continue
        compatible.append(record)
    return max(
        compatible,
        key=lambda record: (record.identity.created_at, record.identity.experiment_id),
        default=None,
    )


def matching_benchmark(
    plan: ExecutionPlan, records: Sequence[BenchmarkRecord],
) -> BenchmarkRecord | None:
    compatible = []
    for record in records:
        if not _common_matches(plan, record.artifact, record.runtime, record.hardware):
            continue
        assert plan.backend_selection is not None
        if record.requested_backend != plan.backend_selection.selected_backend:
            continue
        method = preferred_benchmark_methodology(plan.runtime.runtime)
        if (
            method is None or not record.observation.success
            or record.observation.methodology != method
        ):
            continue
        if not any(
            m.kind is BenchmarkMeasurementKind.GENERATION
            for m in record.observation.measurements
        ):
            continue
        if any(
            m.tokens not in record.request.generation_sizes
            for m in record.observation.measurements
            if m.kind is BenchmarkMeasurementKind.GENERATION
        ):
            continue
        if plan.runtime.runtime is RuntimeName.LLAMA_CPP:
            raw = _flags(plan.runtime).get("--n-gpu-layers")
            if raw is None:
                # CPU implies no offload; an unspecified GPU split is unknown.
                if record.requested_backend is not ComputeBackend.CPU:
                    continue
                raw = "0"
            expected = "all" if record.gpu_layers.full_offload else str(record.gpu_layers.count)
            if raw != expected:
                continue
        elif _flags(plan.runtime).get("torch_dtype") != _flags(record.runtime).get("torch_dtype"):
            continue
        compatible.append(record)
    return max(
        compatible,
        key=lambda record: (record.identity.created_at, record.identity.benchmark_id),
        default=None,
    )


def _common_matches(
    plan: ExecutionPlan, artifact: ModelArtifact,
    runtime: RuntimeRecommendation, hardware: HardwareProfile,
) -> bool:
    return (
        plan.hardware is not None
        and plan.backend_selection is not None
        and machine_fingerprint(plan.hardware) == machine_fingerprint(hardware)
        and plan.runtime.runtime is runtime.runtime
        and _artifact_matches(plan.artifact.to_model_artifact(), artifact)
    )


def _artifact_matches(left: ModelArtifact, right: ModelArtifact) -> bool:
    if (
        left.repo_id, left.revision, left.filename, left.format, left.quantization,
    ) != (
        right.repo_id, right.revision, right.filename, right.format, right.quantization,
    ):
        return False
    return all(
        a is None or b is None or a == b
        for a, b in ((left.size_bytes, right.size_bytes), (left.sha256, right.sha256))
    )


def _configuration_matches(left: InferenceConfiguration, right: InferenceConfiguration) -> bool:
    # Reserve, safety margin and target AUTO/CPU/GPU are planning policy, not
    # observed workload. Backend and actual launch flags are checked separately.
    fields = (
        "context_length", "batch_size", "concurrent_users", "precision",
        "quantization", "kv_cache_dtype",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _flags(runtime: RuntimeRecommendation) -> dict[str, str]:
    return {flag.name: flag.value for flag in runtime.flags}
