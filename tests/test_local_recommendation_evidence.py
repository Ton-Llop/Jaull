"""Local records must describe the plan they are used to support."""

from datetime import UTC, datetime

import pytest

from jaull.domain.benchmarks import (
    BenchmarkFailureReason,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
)
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeFlag, RuntimeFlagSource
from jaull.recommendation.engine_v2 import PlanRankingContext, generate_execution_plans
from jaull.recommendation.local_evidence import matching_benchmark, matching_experiment
from tests._workflow_fixtures import hardware
from tests.test_recommendation_engine_v2 import (
    _benchmark_record,
    _evaluated_gguf,
    _evaluated_transformers,
    _experiment_record,
    _requirements,
    _selection,
)


def _plan() -> ExecutionPlan:
    return generate_execution_plans(
        _evaluated_gguf(), _requirements(),
        context=PlanRankingContext(hardware=hardware(), backend_selection=_selection()),
    )[0]


@pytest.mark.parametrize("kind", ["experiment", "benchmark"])
@pytest.mark.parametrize("field,value", [("revision", "other-revision"), ("size_bytes", 123)])
def test_conflicting_artifact_metadata_is_not_local_evidence(
    kind: str, field: str, value: object,
) -> None:
    plan = _plan()
    artifact = plan.artifact.to_model_artifact().model_copy(update={field: value})
    if kind == "experiment":
        record = _experiment_record(artifact, plan.runtime)
        assert matching_experiment(plan, [record]) is None
    else:
        benchmark = _benchmark_record(artifact, plan.runtime, machine=hardware(), tps=50)
        assert matching_benchmark(plan, [benchmark]) is None


@pytest.mark.parametrize("field", ["hardware", "backend_selection"])
def test_missing_plan_context_is_not_a_wildcard(field: str) -> None:
    plan = _plan()
    experiment = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    benchmark = _benchmark_record(
        plan.artifact.to_model_artifact(), plan.runtime, machine=hardware(), tps=50,
    )
    incomplete = plan.model_copy(update={field: None})
    assert matching_experiment(incomplete, [experiment]) is None
    assert matching_benchmark(incomplete, [benchmark]) is None


@pytest.mark.parametrize("field", ["context_length", "batch_size", "concurrent_users"])
def test_experiment_workload_must_match(field: str) -> None:
    plan = _plan()
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    cfg = record.prediction.inference_configuration
    record = record.model_copy(update={"prediction": record.prediction.model_copy(update={
        "inference_configuration": cfg.model_copy(update={field: getattr(cfg, field) * 2}),
    })})
    assert matching_experiment(plan, [record]) is None


def test_observed_backend_fallback_is_not_evidence_for_requested_gpu() -> None:
    plan = _plan().model_copy(update={"backend_selection": _selection(ComputeBackend.CUDA)})
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    readiness = record.preflight.execution_readiness.model_copy(update={
        "selection": _selection(ComputeBackend.CUDA),
    })
    record = record.model_copy(update={
        "preflight": record.preflight.model_copy(update={"execution_readiness": readiness}),
        "backend_trace": record.backend_trace.model_copy(
            update={"observed_backend": ComputeBackend.CPU},
        ),
    })
    assert matching_experiment(plan, [record]) is None


def test_runtime_offload_flags_must_match_without_using_hfa_blocks() -> None:
    plan = _plan()
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    runtime = plan.runtime.model_copy(update={"flags": [RuntimeFlag(
        name="--n-gpu-layers", value="18", source=RuntimeFlagSource.USER_INPUT,
        explanation="Explicit runtime offload count",
    )]})
    assert matching_experiment(plan.model_copy(update={"runtime": runtime}), [record]) is None
    benchmark = _benchmark_record(
        plan.artifact.to_model_artifact(), runtime, machine=hardware(), tps=50,
    )
    assert matching_benchmark(plan.model_copy(update={"runtime": runtime}), [benchmark]) is None


def test_failed_benchmark_cannot_displace_successful_evidence() -> None:
    plan = _plan()
    good = _benchmark_record(
        plan.artifact.to_model_artifact(), plan.runtime, machine=hardware(), tps=50,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    failed = good.model_copy(update={
        "identity": good.identity.model_copy(
            update={"created_at": datetime(2026, 1, 2, tzinfo=UTC)},
        ),
        "observation": BenchmarkObservation(
            success=False, repetitions=1, duration_seconds=1,
            failure_reason=BenchmarkFailureReason.TIMEOUT,
        ),
    })
    assert matching_benchmark(plan, [good, failed]) is good


def test_selection_is_deterministic_and_preserves_records() -> None:
    plan = _plan()
    record = _benchmark_record(
        plan.artifact.to_model_artifact(), plan.runtime, machine=hardware(), tps=50,
    )
    other = record.model_copy(update={
        "identity": record.identity.model_copy(update={"benchmark_id": "zzz"}),
    })
    before = [r.model_dump_json() for r in (record, other)]
    assert matching_benchmark(plan, [record, other]) is other
    assert matching_benchmark(plan, [other, record]) is other
    assert [r.model_dump_json() for r in (record, other)] == before


def test_readiness_does_not_gate_historical_evidence() -> None:
    plan = _plan()
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    missing = plan.model_copy(update={"execution_readiness": None})
    assert matching_experiment(missing, [record]) is record


def test_failed_matching_experiment_remains_negative_evidence() -> None:
    plan = _plan()
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    record = record.model_copy(update={
        "observation": record.observation.model_copy(update={"success": False}),
    })
    assert matching_experiment(plan, [record]) is record


@pytest.mark.parametrize("case", ["backend", "tokens", "prefill_only", "method"])
def test_incompatible_benchmark_does_not_supply_performance(case: str) -> None:
    plan = _plan()
    record = _benchmark_record(
        plan.artifact.to_model_artifact(), plan.runtime, machine=hardware(), tps=50,
    )
    if case == "backend":
        plan = plan.model_copy(update={"backend_selection": _selection(ComputeBackend.CUDA)})
    elif case == "tokens":
        request = record.request.model_copy(update={"generation_sizes": (256,)})
        record = record.model_copy(update={"request": request})
    elif case == "prefill_only":
        measurement = record.observation.measurements[0].model_copy(update={
            "kind": BenchmarkMeasurementKind.PREFILL,
        })
        record = record.model_copy(update={"observation": record.observation.model_copy(
            update={"measurements": [measurement]},
        )})
    else:
        record = record.model_copy(update={"observation": record.observation.model_copy(
            update={"methodology": None},
        )})
    assert matching_benchmark(plan, [record]) is None


def test_hash_absence_does_not_pretend_metadata_is_missing() -> None:
    plan = _plan()
    artifact = plan.artifact.to_model_artifact().model_copy(update={"sha256": "a" * 64})
    record = _experiment_record(artifact, plan.runtime)
    before = record.model_dump_json()
    assert matching_experiment(plan, [record]) is record
    assert record.model_dump_json() == before


def test_planning_margin_is_not_experimental_workload_identity() -> None:
    plan = _plan()
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    cfg = record.prediction.inference_configuration.model_copy(update={
        "safety_margin_percent": 40, "device_reserve_bytes": 123,
    })
    record = record.model_copy(update={"prediction": record.prediction.model_copy(
        update={"inference_configuration": cfg},
    )})
    assert matching_experiment(plan, [record]) is record


def _transformers_plan() -> ExecutionPlan:
    return generate_execution_plans(
        _evaluated_transformers(), _requirements(),
        context=PlanRankingContext(hardware=hardware(), backend_selection=_selection()),
    )[0]


def _with_precision_flag(runtime, name: str, value: str):
    """A Transformers plan carries exactly one precision-bearing flag."""
    kept = [flag for flag in runtime.flags if flag.name not in {"torch_dtype", "quantization"}]
    return runtime.model_copy(update={"flags": [*kept, RuntimeFlag(
        name=name, value=value, source=RuntimeFlagSource.ESTIMATE,
        explanation="test",
    )]})


@pytest.mark.parametrize(
    ("plan_flag", "record_flag"),
    [
        (("quantization", "4bit"), ("quantization", "8bit")),
        (("quantization", "4bit"), ("torch_dtype", "torch.float16")),
        (("torch_dtype", "torch.float16"), ("quantization", "4bit")),
    ],
)
def test_a_benchmark_of_another_precision_is_not_evidence(
    plan_flag: tuple[str, str], record_flag: tuple[str, str],
) -> None:
    """Comparing only ``torch_dtype`` made two quantized plans look identical.

    A quantized plan carries ``quantization`` and no ``torch_dtype``, so an int4
    record and an int8 record both answered ``None`` and matched each other.
    """
    plan = _transformers_plan()
    plan_runtime = _with_precision_flag(plan.runtime, *plan_flag)
    record_runtime = _with_precision_flag(plan.runtime, *record_flag)
    record = _benchmark_record(
        plan.artifact.to_model_artifact(), record_runtime,
        machine=hardware(), tps=50, methodology="transformers_isolated_inference_v2",
    )

    assert matching_benchmark(
        plan.model_copy(update={"runtime": plan_runtime}), [record]
    ) is None


def test_the_same_precision_still_matches() -> None:
    """The guard must not reject evidence that does describe the plan."""
    plan = _transformers_plan()
    runtime = _with_precision_flag(plan.runtime, "quantization", "4bit")
    record = _benchmark_record(
        plan.artifact.to_model_artifact(), runtime,
        machine=hardware(), tps=50, methodology="transformers_isolated_inference_v2",
    )

    assert matching_benchmark(
        plan.model_copy(update={"runtime": runtime}), [record]
    ) is not None
