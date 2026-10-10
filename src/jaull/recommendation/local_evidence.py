"""Eligibility of historical observations for a recommendation plan.

Matching is offline and does not depend on runtime installation/readiness.
Benchmark evidence describes its recorded microbenchmark, not the throughput
of the user's context/concurrency workload. Artifact metadata matching is not
cryptographic verification when the plan has no digest.
"""

from collections.abc import Sequence
from re import fullmatch

from jaull.domain.artifacts import ModelArtifact, artifact_identity_matches
from jaull.domain.benchmarks import (
    BenchmarkMeasurementKind,
    BenchmarkRecord,
    LlamaBenchProtocol,
)
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.recommendation import LocalSpeedReference, SpeedAssessment
from jaull.domain.requirements import UserRequirements, WorkloadMode
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
            expected = (
                {"-1", "all"} if record.gpu_layers.full_offload else {str(record.gpu_layers.count)}
            )
            if raw not in expected:
                continue
        elif not _transformers_precision_matches(plan, record):
            continue
        compatible.append(record)
    return max(
        compatible,
        key=lambda record: (record.identity.created_at, record.identity.benchmark_id),
        default=None,
    )


def fastest_record_blockers(record: BenchmarkRecord) -> tuple[str, ...]:
    """Audit recorded tg128/pp512 evidence, not authorization to change ranking.

    These are prerequisites, not a complete comparability decision or cohort.
    Checking the command is necessary but never sufficient: a flag llama-bench
    accepted does not prove what it applied,
    and settings it left at a default are absent from the table entirely. Never
    infer those from today's binary or from inherited server flags.
    """
    blockers = []
    if not record.observation.success or record.observation.exit_code != 0:
        blockers.append("Benchmark did not complete with a confirmed successful exit.")
    if record.runtime.runtime is not RuntimeName.LLAMA_CPP or record.artifact.format != "gguf":
        blockers.append("Fastest v1 requires a llama-bench GGUF measurement.")
    if record.observation.methodology != "llama_bench_v1":
        blockers.append("llama-bench methodology is missing or unsupported.")
    if (
        record.artifact.sha256 is None
        or fullmatch(r"[0-9a-fA-F]{64}", record.artifact.sha256) is None
    ):
        blockers.append("Exact artifact SHA256 is missing or invalid.")
    if not record.artifact.is_verified:
        blockers.append("The record does not confirm a verified artifact.")
    build = _reported_bench_build("\n".join((
        record.observation.raw_stdout, record.observation.raw_stderr,
    )))
    if build is None:
        blockers.append("The producing llama-bench build is unknown or ambiguous in the output.")
    capability = record.llama_bench_capability
    if capability is not None and capability.version_text:
        probed = _reported_bench_build(capability.version_text)
        if probed is not None and probed != build:
            blockers.append("The probed llama-bench build disagrees with the producing build.")
    if record.request.repetitions != 5 or record.observation.repetitions != 5:
        blockers.append("Fastest v1 requires five repetitions.")
    for kind, tokens, label, sizes in (
        (BenchmarkMeasurementKind.GENERATION, 128, "tg128", record.request.generation_sizes),
        (BenchmarkMeasurementKind.PREFILL, 512, "pp512", record.request.prefill_sizes),
    ):
        measurements = [m for m in record.observation.measurements if (
            m.kind is kind and m.tokens == tokens
        )]
        if tokens not in sizes or len(measurements) != 1:
            blockers.append(f"A single requested {label} measurement is required.")
            continue
        measurement = measurements[0]
        if measurement.repetitions != 5:
            blockers.append(f"{label} repetition count is missing or incompatible.")
        expected_device = (
            "none" if record.requested_backend is ComputeBackend.CPU else record.requested_device
        )
        if expected_device is None or measurement.raw_device != expected_device:
            blockers.append(
                f"{label} runtime-reported device is unknown or differs from the request."
            )
        ngl = "-1" if record.gpu_layers.full_offload else str(record.gpu_layers.count)
        if measurement.raw_ngl != ngl:
            blockers.append(
                f"{label} runtime-reported offload is unknown or differs from the request."
            )
        # raw_backend lists loaded backends; CUDA there does not prove GPU use.

    # Compare only the six options the current producer actually passes.
    # Even an explicit, matching argv is not proof of a complete applied protocol.
    expected = {
        "-m": str(record.artifact.local_path) if record.artifact.local_path else None,
        "-dev": (
            "none" if record.requested_backend is ComputeBackend.CPU else record.requested_device
        ),
        "-ngl": "-1" if record.gpu_layers.full_offload else str(record.gpu_layers.count),
        "-p": ",".join(map(str, record.request.prefill_sizes)),
        "-n": ",".join(map(str, record.request.generation_sizes)),
        "-r": str(record.request.repetitions),
    }
    command = record.observation.command
    for flag, value in expected.items():
        if value is None or command.count(flag) != 1:
            blockers.append(f"Recorded command {flag} is missing or ambiguous.")
            continue
        index = command.index(flag) + 1
        if index >= len(command) or command[index] != value:
            blockers.append(f"Recorded command {flag} differs from the recorded request.")
    blockers.extend(_protocol_blockers(record.observation.protocol, build))
    return tuple(blockers)


# Set once the producer asks llama-bench to restate its settings. A record
# written before that cannot be audited after the fact: the markdown table it
# was parsed from never carried these columns.
_UNAUDITED_PROTOCOL = (
    "Effective threads, batching, cache types and load protocol were not recorded by "
    "the run that produced this measurement."
)


def _protocol_blockers(
    protocol: LlamaBenchProtocol | None, reported_build: str | None,
) -> list[str]:
    """Require the settings to be recorded, not reconstructed from today's defaults.

    llama-bench omits any default-valued column from its table, and its defaults
    are machine-dependent: `-t` follows the core count. So an absent value is
    unknown, and filling it in from this machine would invent the protocol of a
    run that happened somewhere else.
    """
    if protocol is None:
        return [_UNAUDITED_PROTOCOL]
    blockers = [
        f"Recorded protocol is missing {label}."
        for label, value in (
            ("the effective thread count", protocol.n_threads),
            ("the reported batch size", protocol.n_batch),
            ("the reported microbatch size", protocol.n_ubatch),
            ("the effective K cache type", protocol.cache_type_k),
            ("the effective V cache type", protocol.cache_type_v),
            ("the effective load mode", protocol.load_mode),
            ("its own build", protocol.build_commit),
            ("its own build number", protocol.build_number),
        )
        if value is None or value == ""
    ]
    # The producer passes no -fa, so llama-bench echoes `auto` back unresolved.
    # Whether flash attention actually ran is absent from every output format.
    if protocol.flash_attn is None:
        blockers.append("Recorded protocol is missing the effective flash-attention setting.")
    elif protocol.flash_attn == "auto_unresolved":
        blockers.append(
            "The effective flash-attention setting is absent from llama-bench output: it "
            "reports the requested value, and this run requested `auto`."
        )
    elif protocol.flash_attn == "on":
        blockers.append(
            "Requested flash attention `on` is not proof it ran: the runtime can disable it."
        )
    blockers.append(
        "JSONL reports requested batch/microbatch limits, not runtime-applied context limits."
    )
    if protocol.warmup is None:
        blockers.append("Whether warmup ran was not recorded.")
    if (
        reported_build is not None and protocol.build_commit is not None
        and protocol.build_number is not None
        and reported_build != f"{protocol.build_commit.lower()} ({protocol.build_number})"
    ):
        blockers.append("The recorded protocol build disagrees with the producing build.")
    return blockers


def _reported_bench_build(output: str) -> str | None:
    """Strict output attribution, unlike a probe's permissive last-build enrichment."""
    builds = set()
    for line in output.splitlines():
        if not line.startswith("build:"):
            continue
        match = fullmatch(r"build:\s*([0-9a-fA-F]{7,40})\s*\((\d+)\)\s*", line)
        if match is None:
            return None
        builds.add(f"{match[1].lower()} ({int(match[2])})")
    return next(iter(builds)) if len(builds) == 1 else None


def fastest_benchmark_blockers(
    plan: ExecutionPlan, requirements: UserRequirements, record: BenchmarkRecord,
) -> tuple[str, ...]:
    """Additional applicability checks for one plan; existing matching stays unchanged."""
    blockers = list(fastest_record_blockers(record))
    if (
        requirements.workload_mode is not WorkloadMode.INTERACTIVE
        or requirements.concurrent_users != 1
    ):
        blockers.append(
            "A single-request microbenchmark does not cover batch or multiuser throughput."
        )
    sha = record.artifact.sha256
    if plan.artifact.sha256 is None or sha is None or plan.artifact.sha256 != sha.lower():
        blockers.append("This plan has no matching exact artifact SHA256.")
    if matching_benchmark(plan, [record]) is None:
        blockers.append("Historical artifact/machine/runtime/backend/offload matching failed.")
    if record.requested_backend is not ComputeBackend.CPU:
        devices = [flag.value for flag in plan.runtime.flags if flag.name == "--device"]
        if len(devices) != 1 or devices[0] != record.requested_device:
            blockers.append("The plan's runtime device is unknown or differs from the measurement.")
    return tuple(blockers)


def assess_speed(
    plan: ExecutionPlan, requirements: UserRequirements, records: Sequence[BenchmarkRecord],
) -> SpeedAssessment:
    """Project stored benchmarks of this exact artifact; never authorization to rank.

    A record is referenced only by content digest, never by name, family or
    quantization label. Everything else that could make it inapplicable - other
    machine, runtime, offload, workload, unaudited protocol - is reported as a
    blocker on the reference instead of hiding the measurement.
    """
    # ArtifactVariant validates this as lowercase hex; record digests are free-form.
    sha = plan.artifact.sha256
    if sha is None:
        return SpeedAssessment()
    references = [
        LocalSpeedReference(
            benchmark_id=record.identity.benchmark_id,
            created_at=record.identity.created_at,
            artifact_sha256=sha,
            **_single(record, BenchmarkMeasurementKind.GENERATION, 128, "tg128"),
            **_single(record, BenchmarkMeasurementKind.PREFILL, 512, "pp512"),
            repetitions=record.observation.repetitions,
            blockers=fastest_benchmark_blockers(plan, requirements, record),
        )
        for record in records
        if _exact_sha(record.artifact.sha256) == sha
    ]
    references.sort(key=lambda item: (item.created_at, item.benchmark_id), reverse=True)
    return SpeedAssessment(
        applicability="reference_only" if references else "absent",
        references=tuple(references),
    )


def _exact_sha(value: str | None) -> str | None:
    """A record's digest in the plan's lowercase form, or nothing if it is not one."""
    if value is None or fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        return None
    return value.lower()


def _single(
    record: BenchmarkRecord, kind: BenchmarkMeasurementKind, tokens: int, label: str,
) -> dict[str, float]:
    matches = [m for m in record.observation.measurements if m.kind is kind and m.tokens == tokens]
    if len(matches) != 1:
        return {}
    return {
        f"{label}_tokens_per_second": matches[0].mean_tokens_per_second,
        f"{label}_stddev_tokens_per_second": matches[0].stddev_tokens_per_second,
    }


def _transformers_precision_matches(
    plan: ExecutionPlan, record: BenchmarkRecord,
) -> bool:
    """Both halves of the precision, not just the dtype one.

    A quantized plan carries ``quantization`` and no ``torch_dtype``, so
    comparing only the dtype made an int4 record and an int8 record look like
    the same configuration — both answer ``None``.
    """
    plan_flags = _flags(plan.runtime)
    record_flags = _flags(record.runtime)
    return all(
        plan_flags.get(name) == record_flags.get(name)
        for name in ("torch_dtype", "quantization")
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
        and artifact_identity_matches(plan.artifact.to_model_artifact(), artifact)
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
