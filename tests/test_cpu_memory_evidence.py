from __future__ import annotations

from types import SimpleNamespace

import pytest

from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution import ExecutionObservation
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import TargetDevice
from jaull.estimator.overhead import estimate_gguf_cpu_overhead
from jaull.evaluation.comparison import compare_prediction
from jaull.tui.screens.recommendation_benchmark import _bytes, _memory_delta, _memory_rows
from jaull.tui.screens.recommendation_validation import _result_rows
from tests.test_experiment_record import _estimate, _record, _runtime


def _cpu_record():
    runtime = _runtime(n_gpu_layers=0)
    record = _record(runtime=runtime, prediction=_estimate(runtime=runtime))
    prediction = record.prediction.model_copy(update={
        "assessment": record.prediction.assessment.model_copy(update={
            "effective_device": TargetDevice.CPU,
        }),
    })
    return record.model_copy(update={"prediction": prediction})


@pytest.mark.parametrize("preserved", [True, False])
def test_execution_success_does_not_imply_preserved_reserve_or_performance(preserved) -> None:
    record = _cpu_record()
    reserve = record.prediction.device_reserve.bytes
    record = record.model_copy(update={"observation": record.observation.model_copy(update={
        "success": True, "min_available_ram_bytes": reserve + (1 if preserved else -1),
        "system_swap_in_bytes": 0, "system_swap_out_bytes": 1024,
    })})
    rows = dict(_result_rows(record))
    assert rows["Outcome"] == "Success"
    assert rows["Reserve preserved"] == ("Yes" if preserved else "No")
    assert rows["Sustained performance"] == "Not measured"
    assert "System swap out" in rows


def test_old_records_do_not_claim_headroom_was_measured() -> None:
    record = _cpu_record()
    rows = dict(_result_rows(record))
    assert rows["Reserve preserved"] == "Not measured"
    assert "System swap in" not in rows
    assert "System swap out" not in rows


def test_benchmark_and_validation_compare_consumption_not_reserved_budget() -> None:
    record = _cpu_record()
    prediction = record.prediction
    measured = prediction.physical_bytes + 1000
    observation = record.observation.model_copy(update={"peak_ram_bytes": measured})
    runtime = record.runtime.model_copy(update={"flags": [
        flag.model_copy(update={"value": "0"}) if flag.name == "--n-gpu-layers" else flag
        for flag in record.runtime.flags
    ]})
    comparison = compare_prediction(estimate=prediction, observation=observation, runtime=runtime)
    run = SimpleNamespace(record=SimpleNamespace(
        requested_backend=ComputeBackend.CPU, requested_device=None,
        observation=SimpleNamespace(peak_ram_bytes=measured, peak_vram_bytes=None),
    ))
    rows = dict(_memory_rows([run], prediction))
    assert rows["Estimated"] == _bytes(comparison.ram.predicted_bytes)
    assert rows["Difference"] == _memory_delta(comparison.ram.predicted_bytes, measured)
    assert rows["Budget including reserve"] == _bytes(prediction.total_bytes)
    assert prediction.physical_bytes != prediction.total_bytes


def test_unstructured_gpu_run_does_not_compare_host_rss_with_a_gpu_budget() -> None:
    record = _cpu_record()
    run = SimpleNamespace(record=SimpleNamespace(
        requested_backend=ComputeBackend.CUDA, requested_device="CUDA0",
        observation=SimpleNamespace(peak_ram_bytes=1234, peak_vram_bytes=5678),
    ))
    rows = dict(_memory_rows([run], record.prediction))
    assert "Estimated" not in rows
    assert "Difference" not in rows


def test_cpu_overhead_reports_unknown_repacking_without_a_universal_correction() -> None:
    small = estimate_gguf_cpu_overhead(1024**3)
    large = estimate_gguf_cpu_overhead(2 * 1024**3)
    assert large.component.bytes - small.component.bytes == (
        int(2 * 1024**3 * 0.1) - int(1024**3 * 0.1)
    )
    assert small.component.confidence is EstimationConfidence.LOW
    assert small.unmodelled_components == ["CPU repacking"]
    assert small.components[-1].bytes is None
    assert sum(c.bytes for c in small.components if c.bytes is not None) == small.component.bytes


def test_pressure_never_becomes_a_confirmed_prediction_after_success() -> None:
    record = _cpu_record()
    estimate = record.prediction.model_copy(update={
        "assessment": record.prediction.assessment.model_copy(update={
            "status": CompatibilityStatus.MEMORY_PRESSURE,
        }),
    })
    comparison = compare_prediction(estimate=estimate, observation=record.observation)
    assert comparison.compatibility.predicted_runnable is None
    assert comparison.compatibility.outcome.value != "correct_success"


def test_pressure_observations_are_additive_and_legacy_fields_stay_unknown() -> None:
    record = _cpu_record()
    data = record.observation.model_dump(mode="json")
    for key in ("min_available_ram_bytes", "system_swap_in_bytes", "system_swap_out_bytes"):
        data.pop(key)
    old = ExecutionObservation.model_validate(data)
    assert old.min_available_ram_bytes is None
    assert old.system_swap_in_bytes is None
    assert old.system_swap_out_bytes is None


def test_cpu_allowance_improves_the_recorded_laptop_case_without_claiming_exactness() -> None:
    # Qwen2.5-Coder-7B Q3_K_M, CPU, FP16 KV/4096, observed on 2026-11-13.
    weights, kv, measured = 3_808_391_104, 234_881_024, 5_104_406_528
    old_physical = weights + kv + 256 * 1024**2
    overhead = estimate_gguf_cpu_overhead(weights)
    physical = weights + kv + overhead.component.bytes
    assert physical == 4_960_982_150
    assert abs(measured - physical) < abs(measured - old_physical)
    assert physical != measured
    assert overhead.component.confidence is EstimationConfidence.LOW
    assert overhead.unmodelled_components == ["CPU repacking"]
