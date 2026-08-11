from __future__ import annotations

from pathlib import Path

from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionMeasurementMetadata,
    ExecutionObservation,
    ExecutionResult,
    InferenceResult,
)


def test_execution_observation_success_serializes_measurement_metadata() -> None:
    observation = ExecutionObservation(
        success=True,
        duration_seconds=1.82,
        peak_ram_bytes=2_134_532_096,
        peak_vram_bytes=1_842_937_856,
        exit_code=0,
        failure_reason=None,
        measurement=ExecutionMeasurementMetadata(sample_interval_seconds=0.05),
    )

    assert observation.model_dump() == {
        "success": True,
        "duration_seconds": 1.82,
        "peak_ram_bytes": 2_134_532_096,
        "peak_vram_bytes": 1_842_937_856,
        "exit_code": 0,
        "failure_reason": None,
        "measurement": {
            "ram_measurement": "process_rss",
            "vram_measurement": "nvml_process_memory",
            "sample_interval_seconds": 0.05,
        },
    }


def test_execution_observation_failure_allows_missing_metrics() -> None:
    observation = ExecutionObservation(
        success=False,
        duration_seconds=0.01,
        peak_ram_bytes=None,
        peak_vram_bytes=None,
        exit_code=None,
        failure_reason=ExecutionFailureReason.EXECUTABLE_NOT_FOUND,
    )

    assert observation.success is False
    assert observation.peak_ram_bytes is None
    assert observation.peak_vram_bytes is None


def test_execution_result_compatibility_properties_read_observation() -> None:
    observation = ExecutionObservation(
        success=False,
        duration_seconds=0.42,
        peak_ram_bytes=10,
        peak_vram_bytes=None,
        exit_code=7,
        failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
    )
    result = ExecutionResult(stdout="out", stderr="err", observation=observation)

    assert result.exit_code == 7
    assert result.duration_seconds == 0.42
    dumped = result.model_dump()
    assert dumped["observation"]["exit_code"] == 7
    assert "exit_code" not in dumped
    assert "duration_seconds" not in dumped


def test_inference_result_compatibility_properties_read_observation() -> None:
    observation = ExecutionObservation(
        success=True,
        duration_seconds=0.5,
        peak_ram_bytes=None,
        peak_vram_bytes=None,
        exit_code=0,
        failure_reason=None,
    )
    result = InferenceResult(
        text="generated",
        runtime="llama.cpp",
        model_path=Path("/tmp/model.gguf"),
        observation=observation,
    )

    assert result.text == "generated"
    assert result.exit_code == 0
    assert result.duration_seconds == 0.5
    assert result.observation is observation
    dumped = result.model_dump()
    assert "exit_code" not in dumped
    assert "duration_seconds" not in dumped
