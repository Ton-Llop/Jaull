"""Step 2: a llama_bench_v2 run that proves its settings applies to its own plan.

Synthetic records; they test the rule, not anyone's speed.
"""

from __future__ import annotations

from typing import Any

import pytest

from jaull.domain.benchmarks import LLAMA_BENCH_V2, LLAMA_BENCH_V2_FLAGS, LlamaBenchProtocol
from jaull.domain.recommendation import PlanAssessment
from jaull.recommendation.local_evidence import (
    assess_speed,
    fastest_benchmark_blockers,
    speed_cohort,
)
from jaull.recommendation.shadow import production_evidence
from tests.test_fastest_evidence import _record
from tests.test_recommendation_engine_v2 import _requirements


def _v2(**protocol_changes: Any) -> tuple[Any, Any]:
    plan, record = _record()
    cores = record.hardware.cpu.physical_cores
    protocol = LlamaBenchProtocol(
        build_commit="fffffffff", build_number=1, n_threads=cores, n_batch=512,
        n_ubatch=512, cache_type_k="f16", cache_type_v="f16", flash_attn="off",
        load_mode="mmap", warmup="ran", thread_policy="physical_cores",
    ).model_copy(update=protocol_changes)
    observation = record.observation.model_copy(update={
        "methodology": LLAMA_BENCH_V2, "protocol": protocol,
        "command": (*record.observation.command, *LLAMA_BENCH_V2_FLAGS, "-t", str(cores)),
    })
    return plan, record.model_copy(update={"observation": observation})


def test_a_v2_run_that_proves_its_settings_applies_to_its_plan() -> None:
    plan, record = _v2()
    assert fastest_benchmark_blockers(plan, _requirements(), record) == ()
    speed = assess_speed(plan, _requirements(), [record])
    assert speed.applicability == "measured_plan" and speed.applicable is not None
    assert speed.applicable.tokens_per_second == 10
    assert speed.applicable.benchmark_id == record.identity.benchmark_id
    assert speed.applicable.cohort == speed_cohort(record)


@pytest.mark.parametrize("change,blocker", [
    ({"flash_attn": "on"}, "flash_attn='on'"),
    ({"n_batch": 2048}, "n_batch=2048"),
    ({"thread_policy": None}, "thread_policy=None"),
])
def test_a_v2_run_that_departs_from_the_protocol_stays_a_reference(change, blocker) -> None:
    plan, record = _v2(**change)
    speed = assess_speed(plan, _requirements(), [record])
    assert speed.applicability == "reference_only" and speed.applicable is None
    assert any(blocker in item for item in speed.references[0].blockers)


def test_settings_restated_but_missing_from_argv_do_not_apply() -> None:
    plan, record = _v2()
    command = list(record.observation.command)
    del command[command.index("-fa"):command.index("-fa") + 2]
    record = record.model_copy(update={
        "observation": record.observation.model_copy(update={"command": tuple(command)}),
    })
    blockers = fastest_benchmark_blockers(plan, _requirements(), record)
    assert "Recorded command -fa off is missing or differs." in blockers


def test_v1_records_are_never_reinterpreted() -> None:
    plan, record = _record()  # llama_bench_v1, settings at llama-bench's defaults.
    speed = assess_speed(plan, _requirements(), [record])
    assert speed.applicability == "reference_only" and speed.applicable is None


def test_the_cohort_ignores_the_artifact_but_not_the_protocol_or_device() -> None:
    _, record = _v2()
    other_artifact = record.model_copy(update={
        "artifact": record.artifact.model_copy(update={"sha256": "b" * 64}),
    })
    assert speed_cohort(other_artifact) == speed_cohort(record)
    _, more_threads = _v2(n_threads=12)
    assert speed_cohort(more_threads) != speed_cohort(record)
    other_device = record.model_copy(update={"requested_device": "CUDA1"})
    assert speed_cohort(other_device) != speed_cohort(record)


def test_production_evidence_hands_the_measured_speed_to_the_policies() -> None:
    plan, record = _v2()
    assessment = PlanAssessment(
        plan_id=plan.plan_id, speed=assess_speed(plan, _requirements(), [record]),
    )
    evidence = production_evidence(assessment)
    assert evidence.speed is not None and evidence.speed.value == 10
    assert evidence.speed.label == "tg128 tok/s"
    assert evidence.speed.evidence_ids == (record.identity.benchmark_id,)


def _cuda_case(
    *, gpus: int = 1, device: str = "CUDA0", banner: str | None = None,
    record_driver: str | None = None,
) -> tuple[Any, Any]:
    """A Search-style CUDA plan (no --device flag) and its own v2 record.

    Like a real run, the record keeps llama.cpp's device enumeration in its stderr
    and the plan knows the accelerator its backend selection chose.
    """
    from jaull.domain.benchmarks import (
        BenchmarkGpuLayers,
        BenchmarkRecord,
        BenchmarkRequest,
        LlamaBenchBinaryStatus,
        LlamaBenchCapability,
    )
    from jaull.domain.hardware import AcceleratorType, AcceleratorVendor, ComputeBackend
    from jaull.domain.runtime import RuntimeFlag, RuntimeFlagSource, SelectedAcceleratorRef
    from jaull.recommendation.engine_v2 import PlanRankingContext, generate_execution_plans
    from tests._workflow_fixtures import hardware
    from tests.test_recommendation_engine_v2 import _evaluated_gguf, _selection

    machine = hardware()
    machine = machine.model_copy(update={"gpus": machine.gpus * gpus})
    gpu = machine.gpus[0].name
    selection = _selection(ComputeBackend.CUDA).model_copy(update={
        "selected_accelerator": SelectedAcceleratorRef(
            name=gpu, vendor=AcceleratorVendor.NVIDIA, type=AcceleratorType.DEDICATED,
        ),
    })
    plan = generate_execution_plans(
        _evaluated_gguf(), _requirements(),
        context=PlanRankingContext(hardware=machine, backend_selection=selection),
    )[0]
    # As the real campaign plans: full offload and no --device flag.
    ngl = "-1"
    plan = plan.model_copy(update={
        "artifact": plan.artifact.model_copy(update={"sha256": "a" * 64}),
        "runtime": plan.runtime.model_copy(update={"flags": [
            RuntimeFlag(name="--n-gpu-layers", value=ngl, source=RuntimeFlagSource.HARDWARE,
                        explanation="Full offload, as the real campaign plans."),
        ]}),
    })
    _, cpu_record = _v2()
    artifact = cpu_record.artifact.model_copy(update={
        "repo_id": plan.artifact.repo_id, "filename": plan.artifact.filename,
        "revision": plan.artifact.revision, "size_bytes": plan.artifact.size_bytes,
        "quantization": plan.artifact.quantization,
    })
    request = BenchmarkRequest(
        artifact=artifact, runtime=plan.runtime, backend=ComputeBackend.CUDA, device=device,
        gpu_layers=BenchmarkGpuLayers.full(),
        prefill_sizes=(512,), generation_sizes=(128,), repetitions=5,
    )
    enumeration = "\n".join(
        f"  Device {index}: {banner or gpu}, compute capability 7.5, VMM: yes"
        for index in range(gpus)
    )
    observation = cpu_record.observation.model_copy(update={
        "command": tuple(device if part == "none" else ngl if part == "0" else part
                         for part in cpu_record.observation.command),
        "measurements": [m.model_copy(update={"raw_device": device, "raw_ngl": ngl})
                         for m in cpu_record.observation.measurements],
        "raw_stderr": f"ggml_cuda_init: found {gpus} CUDA devices:\n{enumeration}\n",
    })
    recorded = machine
    if record_driver is not None:
        recorded = machine.model_copy(update={"gpus": [
            g.model_copy(update={"driver_version": record_driver}) for g in machine.gpus
        ]})
    record = BenchmarkRecord.create(
        hardware=recorded, request=request, observation=observation,
        llama_bench_capability=LlamaBenchCapability(
            binary_status=LlamaBenchBinaryStatus.AVAILABLE, version_text="build: fffffffff (1)",
        ),
    )
    return plan, record


DEVICE_BLOCKER = "The plan's runtime device is unknown or differs from the measurement."


def test_a_search_plan_without_device_takes_the_gpu_the_run_enumerated() -> None:
    plan, record = _cuda_case()
    assert fastest_benchmark_blockers(plan, _requirements(), record) == ()
    assert assess_speed(plan, _requirements(), [record]).applicable is not None


def test_a_record_claiming_another_cuda_index_is_not_the_plans_device() -> None:
    # Codex's repro: a coherent record on CUDA1 used to apply with one GPU present.
    plan, record = _cuda_case(device="CUDA1")
    assert DEVICE_BLOCKER in fastest_benchmark_blockers(plan, _requirements(), record)
    assert assess_speed(plan, _requirements(), [record]).applicable is None


def test_a_device_the_run_names_differently_stays_unknown() -> None:
    plan, record = _cuda_case(banner="NVIDIA GeForce RTX 4090")
    assert DEVICE_BLOCKER in fastest_benchmark_blockers(plan, _requirements(), record)


def test_two_identical_gpus_cannot_prove_which_one_ran() -> None:
    plan, record = _cuda_case(gpus=2)
    assert DEVICE_BLOCKER in fastest_benchmark_blockers(plan, _requirements(), record)


def test_another_driver_blocks_and_splits_the_cohort() -> None:
    # Codex's repro: a changed driver kept the cohort and still applied.
    plan, same = _cuda_case()
    _, other = _cuda_case(record_driver="560.0")
    assert any("driver or CUDA version" in item
               for item in fastest_benchmark_blockers(plan, _requirements(), other))
    assert speed_cohort(other) != speed_cohort(same)


@pytest.mark.parametrize("field", ["driver_version", "cuda_version"])
@pytest.mark.parametrize("unknown", [None, "", " "])
@pytest.mark.parametrize("side", ["plan", "record", "both"])
def test_unknown_cuda_environment_stays_a_reference(field, unknown, side) -> None:
    plan, record = _cuda_case()
    hardware = record.hardware.model_copy(update={"gpus": [
        gpu.model_copy(update={field: unknown}) for gpu in record.hardware.gpus
    ]})
    if side in {"plan", "both"}:
        plan = plan.model_copy(update={"hardware": hardware})
    if side in {"record", "both"}:
        record = type(record).model_validate(
            record.model_copy(update={"hardware": hardware}).model_dump(),
        )
    speed = assess_speed(plan, _requirements(), [record])
    assert speed.applicability == "reference_only"
    assert speed.applicable is None
    assert any("driver or CUDA version" in blocker for blocker in speed.references[0].blockers)


@pytest.mark.parametrize("field", ["driver_version", "cuda_version"])
def test_identical_gpus_with_partial_environment_do_not_crash(field) -> None:
    plan, record = _cuda_case(gpus=2)
    gpus = list(record.hardware.gpus)
    gpus[1] = gpus[1].model_copy(update={field: None})
    hardware = record.hardware.model_copy(update={"gpus": gpus})
    plan = plan.model_copy(update={"hardware": hardware})
    record = type(record).model_validate(
        record.model_copy(update={"hardware": hardware}).model_dump(),
    )
    before = record.model_dump()
    speed = assess_speed(plan, _requirements(), [record])
    assert speed.applicability == "reference_only"
    assert speed.applicable is None
    reversed_record = record.model_copy(update={
        "hardware": hardware.model_copy(update={"gpus": list(reversed(gpus))}),
    })
    assert speed_cohort(record) == speed_cohort(reversed_record)
    assert record.model_dump() == before


# --- One authority: only speed assess_speed declares applicable reaches the ranking ---


def _assessed(records: list[Any]) -> Any:
    from jaull.recommendation.engine_v2 import PlanRankingContext, assess_plan
    from tests._workflow_fixtures import hardware
    from tests.test_recommendation_engine_v2 import _evaluated_gguf, _selection

    plan, _ = _record()
    context = PlanRankingContext(hardware=hardware(), backend_selection=_selection(),
                                 benchmark_records=tuple(records))
    return assess_plan(_evaluated_gguf(), plan, _requirements(), context=context)


def _ranking_axes(assessment: Any) -> tuple[Any, ...]:
    return (assessment.measured_generation_tokens_per_second,
            assessment.performance_evidence)


def test_only_a_v1_benchmark_ranks_exactly_like_no_benchmark() -> None:
    _, v1 = _record()
    with_v1 = _assessed([v1])
    assert with_v1.local_benchmark_id == v1.identity.benchmark_id  # Still visible.
    assert _ranking_axes(with_v1) == _ranking_axes(_assessed([]))


def test_with_v1_and_an_applicable_v2_only_v2_sets_throughput() -> None:
    from datetime import timedelta

    _, v1 = _record()
    _, v2 = _v2()
    v2 = v2.model_copy(update={
        "identity": v2.identity.model_copy(update={
            "benchmark_id": "bench-v2", "created_at": v1.identity.created_at - timedelta(days=1),
        }),
        "observation": v2.observation.model_copy(update={"measurements": [
            m.model_copy(update={"mean_tokens_per_second": 30.0})
            for m in v2.observation.measurements
        ]}),
    })
    assessment = _assessed([v1, v2])
    assert assessment.measured_generation_tokens_per_second == 30.0  # Not v1's 10.
    assert assessment.speed.applicable is not None
    assert assessment.speed.applicable.benchmark_id == "bench-v2"


def test_a_blocked_v2_benchmark_does_not_set_throughput_either() -> None:
    _, blocked = _v2(flash_attn="on")
    assessment = _assessed([blocked])
    assert assessment.speed.applicability == "reference_only"
    assert _ranking_axes(assessment) == _ranking_axes(_assessed([]))


# --- Matching survives a reboot and a pinned commit, never another machine or bytes ---


def _with_ram(record: Any, delta: int) -> Any:
    memory = record.hardware.memory
    return record.model_copy(update={"hardware": record.hardware.model_copy(update={
        "memory": memory.model_copy(update={"total_bytes": memory.total_bytes + delta}),
    })})


def test_ram_drift_between_boots_is_the_same_machine() -> None:
    from jaull.recommendation.local_evidence import matching_benchmark

    plan, record = _cuda_case()
    # WSL reported 8281112576 then 8281108480 bytes across one reboot of the RTX 2060 box.
    assert matching_benchmark(plan, [_with_ram(record, -4096)]) is not None
    assert matching_benchmark(plan, [_with_ram(record, 1024**3)]) is None


def test_a_pinned_commit_matches_a_main_plan_only_with_the_same_digest() -> None:
    from jaull.recommendation.local_evidence import matching_benchmark

    plan, record = _cuda_case()
    plan = plan.model_copy(update={
        "artifact": plan.artifact.model_copy(update={"revision": "main"}),
    })
    pinned = record.model_copy(update={"artifact": record.artifact.model_copy(update={
        "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
    })})
    assert matching_benchmark(plan, [pinned]) is not None
    other_bytes = pinned.model_copy(update={"artifact": pinned.artifact.model_copy(update={
        "sha256": "b" * 64,
    })})
    assert matching_benchmark(plan, [other_bytes]) is None
