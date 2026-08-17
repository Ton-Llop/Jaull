from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkFailureReason,
    BenchmarkGpuLayers,
    BenchmarkMeasurementKind,
    BenchmarkRequest,
)
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionObservation,
    ExecutionRequest,
    ExecutionResult,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.errors import ExecutionFailedError
from jaull.runtime.transformers_benchmark_runner import (
    TransformersBenchmarkRunner,
    build_transformers_benchmark_command,
)


class _FakeExecutionBackend:
    def __init__(
        self,
        *,
        stdout: str,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[ExecutionRequest] = []
        self.stdout = stdout
        self.error = error

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ExecutionResult(
            stdout=self.stdout,
            stderr="",
            observation=_observation(peak_ram_bytes=2_000, peak_vram_bytes=None),
        )


def test_builds_transformers_benchmark_command(tmp_path: Path) -> None:
    request = _request()
    command = build_transformers_benchmark_command(str(_python(tmp_path)), request)

    assert command[:3] == (
        str(_python(tmp_path)),
        "-m",
        "jaull.runtime.transformers_benchmark_worker",
    )
    assert command[command.index("--model-ref") + 1] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert command[command.index("--device-map") + 1] == "cpu"
    assert command[command.index("--prefill-sizes") + 1] == "128,512"
    assert command[command.index("--generation-sizes") + 1] == "64"
    assert command[command.index("--torch-dtype") + 1] == "torch.float32"


def test_runner_parses_successful_transformers_benchmark(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend(stdout=_worker_json())
    runner = TransformersBenchmarkRunner(
        backend=backend,
        python_executable=_python(tmp_path),
    )

    observation = runner.run(_request())

    assert observation.success is True
    assert observation.methodology == "transformers_isolated_inference_v2"
    assert observation.model_load_seconds == 12.5
    assert observation.warmup_seconds == 0.7
    assert observation.time_to_first_token_seconds == 0.4
    assert observation.time_to_first_token_stddev_seconds == 0.03
    assert observation.generation_latency_seconds == 3.2
    assert observation.generation_latency_stddev_seconds == 0.2
    assert observation.peak_ram_bytes == 2_000
    assert [(item.kind, item.tokens) for item in observation.measurements] == [
        (BenchmarkMeasurementKind.PREFILL, 128),
        (BenchmarkMeasurementKind.PREFILL, 512),
        (BenchmarkMeasurementKind.GENERATION, 64),
    ]
    assert observation.measurements[0].mean_tokens_per_second == 50.0
    assert observation.measurements[0].mean_duration_seconds == 2.56
    assert observation.measurements[0].stddev_duration_seconds == 0.1
    assert backend.requests[0].command[1:3] == (
        "-m",
        "jaull.runtime.transformers_benchmark_worker",
    )
    assert "PYTHONPATH" in backend.requests[0].environment
    assert backend.requests[0].environment["PYTHONPATH"]


def test_runner_preserves_failed_worker_as_failed_observation(tmp_path: Path) -> None:
    failed = ExecutionResult(
        stdout=json.dumps({"success": False, "error": "model load failed"}),
        stderr="traceback",
        observation=_observation(
            success=False,
            exit_code=1,
            failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
            peak_ram_bytes=1234,
        ),
    )
    runner = TransformersBenchmarkRunner(
        backend=_FakeExecutionBackend(
            stdout="",
            error=ExecutionFailedError("failed", failed),
        ),
        python_executable=_python(tmp_path),
    )

    observation = runner.run(_request())

    assert observation.success is False
    assert observation.failure_reason is BenchmarkFailureReason.NON_ZERO_EXIT
    assert observation.message == "model load failed"
    assert observation.peak_ram_bytes == 1234


def test_runner_rejects_vulkan_backend(tmp_path: Path) -> None:
    runner = TransformersBenchmarkRunner(
        backend=_FakeExecutionBackend(stdout=_worker_json()),
        python_executable=_python(tmp_path),
    )

    with pytest.raises(Exception, match="does not support vulkan"):
        runner.run(_request(backend=ComputeBackend.VULKAN))


def _request(backend: ComputeBackend = ComputeBackend.CPU) -> BenchmarkRequest:
    return BenchmarkRequest(
        artifact=ModelArtifact(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
            filename="Qwen/Qwen2.5-0.5B-Instruct",
            format="safetensors",
        ),
        runtime=RuntimeRecommendation(
            runtime=RuntimeName.TRANSFORMERS,
            flags=[
                RuntimeFlag(
                    name="torch_dtype",
                    value="torch.float32",
                    source=RuntimeFlagSource.ESTIMATE,
                    explanation="test",
                )
            ],
            confidence=EstimationConfidence.HIGH,
        ),
        backend=backend,
        device="none" if backend is ComputeBackend.CPU else None,
        gpu_layers=BenchmarkGpuLayers.count_layers(0),
        prefill_sizes=(128, 512),
        generation_sizes=(64,),
        repetitions=3,
    )


def _worker_json() -> str:
    return json.dumps(
        {
            "success": True,
            "methodology": "transformers_isolated_inference_v2",
            "model_load_seconds": 12.5,
            "warmup_seconds": 0.7,
            "time_to_first_token_seconds": 0.4,
            "time_to_first_token_stddev_seconds": 0.03,
            "generation_latency_seconds": 3.2,
            "generation_latency_stddev_seconds": 0.2,
            "measurements": [
                {
                    "kind": "prefill",
                    "tokens": 128,
                    "mean_tokens_per_second": 50.0,
                    "stddev_tokens_per_second": 1.5,
                    "mean_duration_seconds": 2.56,
                    "stddev_duration_seconds": 0.1,
                    "source_label": "pp128",
                },
                {
                    "kind": "prefill",
                    "tokens": 512,
                    "mean_tokens_per_second": 47.0,
                    "stddev_tokens_per_second": 1.0,
                    "mean_duration_seconds": 10.89,
                    "stddev_duration_seconds": 0.2,
                    "source_label": "pp512",
                },
                {
                    "kind": "generation",
                    "tokens": 64,
                    "mean_tokens_per_second": 12.0,
                    "stddev_tokens_per_second": 0.2,
                    "mean_duration_seconds": 5.33,
                    "stddev_duration_seconds": 0.1,
                    "source_label": "tg64",
                },
            ],
        }
    )


def _python(tmp_path: Path) -> Path:
    path = tmp_path / "python"
    path.write_text("# python placeholder", encoding="utf-8")
    return path


def _observation(
    *,
    success: bool = True,
    exit_code: int = 0,
    failure_reason: ExecutionFailureReason | None = None,
    peak_ram_bytes: int | None = None,
    peak_vram_bytes: int | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=10.0,
        exit_code=exit_code,
        failure_reason=failure_reason,
        peak_ram_bytes=peak_ram_bytes,
        peak_vram_bytes=peak_vram_bytes,
    )
