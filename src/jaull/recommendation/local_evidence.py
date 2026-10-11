"""Eligibility of historical observations for a recommendation plan.

Matching is offline and does not depend on runtime installation/readiness.
Benchmark evidence describes its recorded microbenchmark, not the throughput
of the user's context/concurrency workload. Artifact metadata matching is not
cryptographic verification when the plan has no digest.
"""

import hashlib
import json
from collections.abc import Sequence
from re import fullmatch
from statistics import median

from jaull.domain.artifacts import ModelArtifact, artifact_identity_matches
from jaull.domain.benchmarks import (
    LLAMA_BENCH_V2,
    LLAMA_BENCH_V2_FLAGS,
    BenchmarkMeasurementKind,
    BenchmarkRecord,
    LlamaBenchProtocol,
    llama_bench_v2_mismatches,
)
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.recommendation import (
    ApplicableSpeed,
    LocalSpeedReference,
    SpeedAssessment,
)
from jaull.domain.requirements import UserRequirements, WorkloadMode
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.evaluation.benchmark_comparison import known_benchmark_methodologies
from jaull.evaluation.hardware_fingerprint import (
    EVIDENCE_MEMORY_GRANULARITY,
    machine_fingerprint,
)


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
    methods = known_benchmark_methodologies(plan.runtime.runtime)
    for record in records:
        if not _common_matches(plan, record.artifact, record.runtime, record.hardware):
            continue
        assert plan.backend_selection is not None
        if record.requested_backend != plan.backend_selection.selected_backend:
            continue
        if not record.observation.success or record.observation.methodology not in methods:
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
    # A current-protocol record wins over any older one, then the newest run.
    return max(
        compatible,
        key=lambda record: (
            bool(methods) and record.observation.methodology == methods[0],
            record.identity.created_at, record.identity.benchmark_id,
        ),
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
    if record.observation.methodology not in ("llama_bench_v1", "llama_bench_v2"):
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
    if record.observation.methodology == LLAMA_BENCH_V2:
        blockers.extend(_v2_blockers(record, build))
    else:
        # v1 left these settings at llama-bench's defaults: a reference, never more.
        blockers.extend(_protocol_blockers(record.observation.protocol, build))
    return tuple(blockers)


def _v2_blockers(record: BenchmarkRecord, reported_build: str | None) -> list[str]:
    """A v2 run proves its settings: argv carries them and llama-bench restates them.

    This replaces v1's unconditional batching and flash-attention blockers. Batch
    limits are clamped to each test's context, but deterministically: equal
    -p/-n/-b/-ub give equal applied limits on every model. ``-fa off`` cannot be
    turned on by the runtime, unlike ``auto`` or ``on``.
    """
    protocol = record.observation.protocol
    cores = record.hardware.cpu.physical_cores
    blockers = [
        f"Reported setting differs from llama_bench_v2: {item}."
        for item in llama_bench_v2_mismatches(protocol, cores)
    ]
    command = record.observation.command
    pairs = list(zip(LLAMA_BENCH_V2_FLAGS[::2], LLAMA_BENCH_V2_FLAGS[1::2], strict=True))
    for flag, value in [*pairs, ("-t", str(cores))]:
        index = command.index(flag) + 1 if command.count(flag) == 1 else None
        if index is None or index >= len(command) or command[index] != value:
            blockers.append(f"Recorded command {flag} {value} is missing or differs.")
    if (
        protocol is not None and reported_build is not None
        and reported_build != f"{(protocol.build_commit or '').lower()} ({protocol.build_number})"
    ):
        blockers.append("The recorded protocol build disagrees with the producing build.")
    return blockers


def speed_cohort(record: BenchmarkRecord) -> str:
    """Everything two tg128 figures must share to be ranked, digested.

    Machine, build, protocol, backend, device and workload. The artifact and its
    offload are excluded: those are what Fastest compares between plans.
    """
    protocol = record.observation.protocol
    return hashlib.sha256(json.dumps({
        "machine": machine_fingerprint(record.hardware, memory_granularity=_MEMORY_GRANULARITY),
        # The general fingerprint leaves drivers out; a driver change can move tok/s.
        "gpu_environment": _gpu_environment(record.hardware),
        "methodology": record.observation.methodology,
        "protocol": protocol.model_dump(mode="json") if protocol is not None else None,
        "backend": record.requested_backend.value,
        "device": record.requested_device,
        "workload": [list(record.request.prefill_sizes),
                     list(record.request.generation_sizes), record.request.repetitions],
    }, sort_keys=True).encode()).hexdigest()


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
        implied = not devices and _runtime_device_is_the_plans_accelerator(plan, record)
        if not implied and (len(devices) != 1 or devices[0] != record.requested_device):
            blockers.append("The plan's runtime device is unknown or differs from the measurement.")
        environment = _gpu_environment(record.hardware)
        if (
            plan.hardware is None
            or not environment
            or any(
                not (driver or "").strip()
                or (record.requested_backend is ComputeBackend.CUDA and not (cuda or "").strip())
                for _, driver, cuda in environment
            )
            or environment != _gpu_environment(plan.hardware)
        ):
            blockers.append(
                "The GPU driver or CUDA version differs from this machine's, or is unknown."
            )
    return tuple(blockers)


def _runtime_device_is_the_plans_accelerator(
    plan: ExecutionPlan, record: BenchmarkRecord,
) -> bool:
    """Search plans name no device: prove the measured one is the plan's accelerator.

    The run's own output carries the runtime's enumeration ("Device 0: <name>, ...").
    Only CUDA ids are read, as ``CUDA<i>`` maps to that index; anything that cannot be
    shown to be the accelerator the plan selected stays unknown, never assumed.
    """
    match = fullmatch(r"CUDA(\d+)", record.requested_device or "")
    selection = plan.backend_selection
    accelerator = selection.selected_accelerator if selection is not None else None
    if match is None or accelerator is None:
        return False
    output = "\n".join((record.observation.raw_stdout, record.observation.raw_stderr))
    enumerated = [found.groups() for line in output.splitlines()
                  if (found := fullmatch(r"\s*Device (\d+): ([^,]+),.*", line))]
    # The name must point at exactly one device: two identical GPUs prove nothing.
    holders = [index for index, name in enumerated if name == accelerator.name]
    return holders == [match[1]]


def _gpu_environment(hardware: HardwareProfile) -> tuple[tuple[str, str | None, str | None], ...]:
    """What a speed figure depends on beyond the machine fingerprint: drivers and CUDA."""
    # Preserve unknown values without comparing None with strings.
    return tuple(sorted(
        ((gpu.name, gpu.driver_version, gpu.cuda_version) for gpu in hardware.gpus),
        key=lambda environment: tuple((value is not None, value or "") for value in environment),
    ))


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
    by_id: dict[str, BenchmarkRecord] = {}
    conflicts: set[str] = set()
    for record in records:
        if _exact_sha(record.artifact.sha256) != sha:
            continue
        identity = record.identity.benchmark_id
        previous = by_id.get(identity)
        if previous is not None and previous != record:
            conflicts.add(identity)
            record = min((previous, record), key=lambda item: item.model_dump_json())
        by_id[identity] = record
    matched = list(by_id.values())
    matched.sort(
        key=lambda record: (record.identity.created_at, record.identity.benchmark_id),
        reverse=True,
    )
    references = [
        LocalSpeedReference(
            benchmark_id=record.identity.benchmark_id,
            created_at=record.identity.created_at,
            artifact_sha256=sha,
            **_single(record, BenchmarkMeasurementKind.GENERATION, 128, "tg128"),
            **_single(record, BenchmarkMeasurementKind.PREFILL, 512, "pp512"),
            repetitions=record.observation.repetitions,
            blockers=fastest_benchmark_blockers(plan, requirements, record) + (
                ("Conflicting records share this benchmark ID.",)
                if record.identity.benchmark_id in conflicts else ()
            ),
        )
        for record in matched
    ]
    eligible = [
        (reference, record)
        for reference, record in zip(references, matched, strict=True)
        if not reference.blockers and reference.tg128_tokens_per_second
    ]
    applicable = None
    if eligible:
        cohort = speed_cohort(eligible[0][1])
        runs = [ref for ref, record in eligible if speed_cohort(record) == cohort]
        figures = [ref.tg128_tokens_per_second for ref in runs
                   if ref.tg128_tokens_per_second is not None]
        prefill = [ref.pp512_tokens_per_second for ref in runs
                   if ref.pp512_tokens_per_second is not None]
        applicable = ApplicableSpeed(
            tokens_per_second=median(figures),
            # A within-call SD is not the uncertainty between calls.
            stddev_tokens_per_second=(
                runs[0].tg128_stddev_tokens_per_second if len(runs) == 1 else None
            ),
            pp512_tokens_per_second=median(prefill) if len(prefill) == len(runs) else None,
            repetitions=eligible[0][1].observation.repetitions,
            cohort=cohort, benchmark_id=runs[0].benchmark_id,
            benchmark_ids=tuple(sorted(ref.benchmark_id for ref in runs)),
            run_count=len(runs), observed_min=min(figures), observed_max=max(figures),
        )
    return SpeedAssessment(
        applicability=(
            "measured_plan" if applicable is not None
            else "reference_only" if references else "absent"
        ),
        applicable=applicable,
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


_MEMORY_GRANULARITY = EVIDENCE_MEMORY_GRANULARITY


def _same_machine(left: HardwareProfile, right: HardwareProfile) -> bool:
    return machine_fingerprint(left, memory_granularity=_MEMORY_GRANULARITY) == (
        machine_fingerprint(right, memory_granularity=_MEMORY_GRANULARITY)
    )


def _same_artifact(plan: ExecutionPlan, artifact: ModelArtifact) -> bool:
    """Identity as usual, or the same file of the same repo with the same SHA256.

    A plan resolved from Search says ``main``; a run pins the immutable commit.
    Equal digests name the same bytes, which is stronger than any revision label.
    """
    if artifact_identity_matches(plan.artifact.to_model_artifact(), artifact):
        return True
    planned, recorded = plan.artifact.sha256, artifact.sha256
    return bool(
        planned and recorded and planned.lower() == recorded.lower()
        and plan.artifact.repo_id == artifact.repo_id
        and plan.artifact.filename == artifact.filename
    )


def _common_matches(
    plan: ExecutionPlan, artifact: ModelArtifact,
    runtime: RuntimeRecommendation, hardware: HardwareProfile,
) -> bool:
    return (
        plan.hardware is not None
        and plan.backend_selection is not None
        and _same_machine(plan.hardware, hardware)
        and plan.runtime.runtime is runtime.runtime
        and _same_artifact(plan, artifact)
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
