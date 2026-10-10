"""Offline synthetic microbenchmarks; not measured performance."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRecord,
    BenchmarkRequest,
    LlamaBenchBinaryStatus,
    LlamaBenchCapability,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.requirements import WorkloadMode
from jaull.domain.runtime import RuntimeFlag, RuntimeFlagSource
from jaull.recommendation.local_evidence import (
    _UNAUDITED_PROTOCOL,
    assess_speed,
    fastest_benchmark_blockers,
    fastest_record_blockers,
    matching_benchmark,
)
from tests.test_local_recommendation_evidence import _plan
from tests.test_recommendation_engine_v2 import _requirements, _selection


def _record():
    plan = _plan()
    plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": "a" * 64,
    })})
    artifact = plan.artifact.to_model_artifact().model_copy(update={
        "local_path": Path("/synthetic/model.gguf"), "is_verified": True, "is_downloaded": True,
        "sha256": plan.artifact.sha256,
    })
    request = BenchmarkRequest(
        artifact=artifact, runtime=plan.runtime, backend=ComputeBackend.CPU,
        device="none", prefill_sizes=(512,), generation_sizes=(128,), repetitions=5,
    )
    observation = BenchmarkObservation(
        success=True, exit_code=0, repetitions=5, duration_seconds=1,
        methodology="llama_bench_v1",
        raw_stdout="build: fffffffff (1)\n",
        command=("synthetic-llama-bench", "-m", str(artifact.local_path), "-dev", "none",
                 "-ngl", "0", "-p", "512", "-n", "128", "-r", "5"),
        measurements=[BenchmarkMeasurement(
            kind=kind, tokens=tokens, mean_tokens_per_second=10,
            stddev_tokens_per_second=1, repetitions=5, source_label=label,
            raw_backend="CPU", raw_device="none", raw_ngl="0",
        ) for kind, tokens, label in (
            (BenchmarkMeasurementKind.GENERATION, 128, "tg128"),
            (BenchmarkMeasurementKind.PREFILL, 512, "pp512"),
        )],
    )
    assert plan.hardware is not None
    record = BenchmarkRecord.create(
        hardware=plan.hardware, request=request, observation=observation,
        llama_bench_capability=LlamaBenchCapability(
            binary_status=LlamaBenchBinaryStatus.AVAILABLE, version_text="build: fffffffff (1)",
        ),
    )
    return plan, record


def test_successful_command_and_warmup_duration_do_not_prove_effective_protocol() -> None:
    plan, record = _record()
    before = record.model_dump_json()
    blockers = fastest_benchmark_blockers(plan, _requirements(), record)
    # Everything else about this record checks out. What is missing is llama-bench's
    # own account of the settings it ran with, which nothing else substitutes for.
    assert blockers == (_UNAUDITED_PROTOCOL,)
    raw = record.model_dump()
    raw["observation"]["warmup_seconds"] = .5
    assert fastest_record_blockers(BenchmarkRecord.model_validate(raw)) == blockers
    assert matching_benchmark(plan, [record]) is record  # Historical API is unchanged.
    assert record.model_dump_json() == before


def test_jsonl_configuration_alone_cannot_clear_the_effective_protocol_audit() -> None:
    plan, record = _record()
    raw = record.model_dump()
    raw["observation"]["protocol"] = {
        "build_commit": "fffffffff", "build_number": 1, "n_threads": 4,
        "n_batch": 2048, "n_ubatch": 512, "cache_type_k": "f16",
        "cache_type_v": "f16", "flash_attn": "on", "load_mode": "mmap",
        "warmup": "ran",
    }
    audited = BenchmarkRecord.model_validate(raw)
    blockers = fastest_benchmark_blockers(plan, _requirements(), audited)
    assert _UNAUDITED_PROTOCOL not in blockers
    assert any("runtime-applied context limits" in reason for reason in blockers)
    assert any("runtime can disable it" in reason for reason in blockers)
    # The measurements themselves never changed; only what is known about them.
    assert audited.observation.measurements == record.observation.measurements


def test_loaded_cuda_backend_is_not_evidence_against_cpu_execution() -> None:
    _, record = _record()
    raw = record.model_dump()
    for measurement in raw["observation"]["measurements"]:
        measurement["raw_backend"] = "CUDA"
    assert fastest_record_blockers(BenchmarkRecord.model_validate(raw)) == (
        fastest_record_blockers(record)
    )


@pytest.mark.parametrize("case,reason", [
    ("sha", "SHA256"), ("build", "build is unknown"),
    ("method", "methodology"), ("exit", "successful exit"),
    ("repetitions", "five repetitions"), ("metric_repetitions", "repetition count"),
    ("device", "runtime-reported device"), ("offload", "runtime-reported offload"),
    ("command", "command -ngl differs"), ("duplicate_flag", "command -ngl is missing"),
    ("duplicate_metric", "single requested tg128"),
])
def test_fastest_checks_recorded_conditions_instead_of_success_alone(case, reason) -> None:
    _, record = _record()
    raw = record.model_dump()
    observation = raw["observation"]
    if case == "sha":
        raw["artifact"]["sha256"] = raw["request"]["artifact"]["sha256"] = None
    elif case == "build":
        raw["llama_bench_capability"]["version_text"] = None
        observation["raw_stdout"] = ""
    elif case == "method":
        observation["methodology"] = None
    elif case == "exit":
        observation["exit_code"] = None
    elif case == "repetitions":
        observation["repetitions"] = 3
    elif case == "metric_repetitions":
        observation["measurements"][0]["repetitions"] = None
    elif case == "device":
        observation["measurements"][0]["raw_device"] = "CUDA0"
    elif case == "offload":
        observation["measurements"][0]["raw_ngl"] = "12"
    elif case == "command":
        command = list(observation["command"])
        command[command.index("-ngl") + 1] = "12"
        observation["command"] = command
    elif case == "duplicate_flag":
        observation["command"] = (*observation["command"], "-ngl", "0")
    else:
        observation["measurements"].append(observation["measurements"][0].copy())
    checked = BenchmarkRecord.model_validate(raw)
    assert any(reason in blocker for blocker in fastest_record_blockers(checked))


@pytest.mark.parametrize("output,reason", [
    ("build: eeeeeeeee (2)\n", "disagrees"),
    ("build: fffffffff (1)\nbuild: eeeeeeeee (2)\n", "ambiguous"),
    ("build: unknown\n", "ambiguous"),
])
def test_ambiguous_or_conflicting_build_provenance_is_not_trusted(output, reason) -> None:
    _, record = _record()
    raw = record.model_dump()
    raw["observation"]["raw_stdout"] = output
    assert any(reason in blocker for blocker in fastest_record_blockers(
        BenchmarkRecord.model_validate(raw),
    ))


def test_recorded_footer_supplies_build_without_rewriting_missing_metadata() -> None:
    _, record = _record()
    raw = record.model_dump()
    raw["llama_bench_capability"] = None
    without_probe = BenchmarkRecord.model_validate(raw)
    before = without_probe.model_dump_json()
    assert fastest_record_blockers(without_probe) == fastest_record_blockers(record)
    assert without_probe.model_dump_json() == before


def test_tg64_is_not_tg128_even_if_it_has_higher_throughput() -> None:
    _, record = _record()
    raw = record.model_dump()
    raw["observation"]["measurements"][0].update(tokens=64, mean_tokens_per_second=1000)
    raw["request"]["generation_sizes"] = (64,)
    command = list(raw["observation"]["command"])
    command[command.index("-n") + 1] = "64"
    raw["observation"]["command"] = command
    checked = BenchmarkRecord.model_validate(raw)
    assert "A single requested tg128 measurement is required." in fastest_record_blockers(checked)


@pytest.mark.parametrize("mode,users", [(WorkloadMode.BATCH, 1), (WorkloadMode.INTERACTIVE, 2)])
def test_single_request_benchmark_does_not_validate_other_workloads(mode, users) -> None:
    plan, record = _record()
    request = _requirements().model_copy(update={"workload_mode": mode, "concurrent_users": users})
    assert any("batch or multiuser" in reason for reason in (
        fastest_benchmark_blockers(plan, request, record)
    ))


@pytest.mark.parametrize("sha", [None, "b" * 64])
def test_digest_and_machine_are_required_but_inherited_context_is_not_applied(sha) -> None:
    plan, record = _record()
    missing_sha = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": sha,
    })})
    assert matching_benchmark(missing_sha, [record]) is record
    assert "This plan has no matching exact artifact SHA256." in fastest_benchmark_blockers(
        missing_sha, _requirements(), record,
    )
    assert plan.hardware is not None
    other = plan.model_copy(update={"hardware": plan.hardware.model_copy(update={
        "memory": plan.hardware.memory.model_copy(update={
            "total_bytes": plan.hardware.memory.total_bytes * 2,
        }),
    })})
    assert any("matching failed" in reason for reason in fastest_benchmark_blockers(
        other, _requirements(), record,
    ))
    changed_context = _requirements().model_copy(update={"desired_context": 32768})
    assert fastest_benchmark_blockers(plan, changed_context, record) == (
        fastest_benchmark_blockers(plan, _requirements(), record)
    )
    changed_free_memory = plan.model_copy(update={"hardware": plan.hardware.model_copy(update={
        "memory": plan.hardware.memory.model_copy(update={"available_bytes": 1}),
    })})
    assert fastest_benchmark_blockers(changed_free_memory, _requirements(), record) == (
        fastest_benchmark_blockers(plan, _requirements(), record)
    )


def test_another_runtime_device_cannot_support_this_gpu_plan() -> None:
    plan, record = _record()
    runtime = plan.runtime.model_copy(update={"flags": [RuntimeFlag(
        name=name, value=value, source=RuntimeFlagSource.USER_INPUT,
        explanation="Synthetic setting",
    ) for name, value in (("--n-gpu-layers", "-1"), ("--device", "CUDA1"))]})
    plan = plan.model_copy(update={
        "runtime": runtime, "backend_selection": _selection(ComputeBackend.CUDA),
    })
    raw = record.model_dump()
    raw["runtime"] = raw["request"]["runtime"] = runtime.model_dump()
    raw["requested_backend"] = raw["request"]["backend"] = ComputeBackend.CUDA
    raw["requested_device"] = raw["request"]["device"] = "CUDA0"
    raw["gpu_layers"] = raw["request"]["gpu_layers"] = BenchmarkGpuLayers.full().model_dump()
    raw["observation"]["command"] = (
        "synthetic-llama-bench", "-m", str(record.artifact.local_path), "-dev", "CUDA0",
        "-ngl", "-1", "-p", "512", "-n", "128", "-r", "5",
    )
    for measurement in raw["observation"]["measurements"]:
        measurement.update(raw_device="CUDA0", raw_ngl="-1", raw_backend="CUDA")
    checked = BenchmarkRecord.model_validate(raw)
    assert matching_benchmark(plan, [checked]) is checked
    assert "The plan's runtime device is unknown or differs from the measurement." in (
        fastest_benchmark_blockers(plan, _requirements(), checked)
    )


def _with_prefill(record, tokens_per_second):
    raw = record.model_dump()
    for measurement in raw["observation"]["measurements"]:
        if measurement["kind"] == "prefill":
            measurement["mean_tokens_per_second"] = tokens_per_second
    return BenchmarkRecord.model_validate(raw)


def test_speed_references_the_exact_artifact_and_keeps_its_blockers() -> None:
    plan, record = _record()
    record = _with_prefill(record, 400)

    speed = assess_speed(plan, _requirements(), [record])

    assert speed.applicability == "reference_only"
    reference, = speed.references
    assert reference.benchmark_id == record.identity.benchmark_id
    # Separate lengths, never the larger of the two standing in for generation.
    assert reference.tg128_tokens_per_second == 10
    assert reference.pp512_tokens_per_second == 400
    assert reference.repetitions == 5
    # Every reason it cannot be compared travels with the number.
    assert reference.blockers == fastest_benchmark_blockers(plan, _requirements(), record)
    assert reference.blockers


def test_a_record_digest_in_upper_case_names_the_same_bytes() -> None:
    # Plan digests are validated lowercase; a benchmark record's ModelArtifact is not.
    plan, record = _record()
    raw = record.model_dump()
    for artifact in (raw["artifact"], raw["request"]["artifact"]):
        artifact["sha256"] = artifact["sha256"].upper()
    upper = BenchmarkRecord.model_validate(raw)

    reference, = assess_speed(plan, _requirements(), [upper]).references

    assert reference.artifact_sha256 == plan.artifact.sha256
    assert "This plan has no matching exact artifact SHA256." not in reference.blockers


@pytest.mark.parametrize("sha", [None, "b" * 64])
def test_speed_without_the_same_exact_bytes_is_absent(sha) -> None:
    plan, record = _record()
    other = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={"sha256": sha})})

    speed = assess_speed(other, _requirements(), [record])

    assert speed.applicability == "absent"
    assert speed.references == ()


def test_speed_references_read_newest_first_with_id_breaking_ties() -> None:
    plan, record = _record()
    raw = record.model_dump()

    def copy(benchmark_id, hours):
        clone = deepcopy(raw)
        clone["identity"] = {"benchmark_id": benchmark_id,
                             "created_at": record.identity.created_at + timedelta(hours=hours)}
        return BenchmarkRecord.model_validate(clone)

    records = [copy("bench-a", 0), copy("bench-z", 2), copy("bench-b", 2)]

    speed = assess_speed(plan, _requirements(), records)

    assert [item.benchmark_id for item in speed.references] == ["bench-z", "bench-b", "bench-a"]
