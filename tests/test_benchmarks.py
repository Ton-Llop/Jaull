from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jaull.benchmarks.errors import (
    BenchmarkParseError,
    BenchmarkStoreError,
    BenchmarkUnavailableError,
    InvalidBenchmarkIdError,
)
from jaull.benchmarks.matrix import BenchmarkMatrixRequest, BenchmarkMatrixRunner
from jaull.benchmarks.storage import SCHEMA_VERSION, BenchmarkStore
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRecord,
    BenchmarkRequest,
    LlamaBenchBinaryStatus,
    LlamaBenchCapability,
)
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution import ExecutionObservation, ExecutionRequest, ExecutionResult
from jaull.domain.hardware import ComputeBackend, CpuInfo, HardwareProfile, MemoryInfo
from jaull.domain.runtime import (
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.benchmarks import aggregate_benchmark_records
from jaull.execution.errors import ExecutionFailedError
from jaull.runtime.llama_bench_capability import inspect_llama_bench
from jaull.runtime.llama_bench_parser import parse_llama_bench_output
from jaull.runtime.llama_bench_runner import (
    LlamaBenchRunner,
    build_llama_bench_command,
)

CPU_OUTPUT = """
load_backend: loaded CPU backend
| model                  | size | params | backend | ngl | dev  | test   | t/s           |
| ---------------------- | ---- | ------ | ------- | --- | ---- | ------ | ------------- |
| qwen2 3B Q4_K - Medium | 1.8G | 3.1 B  | Vulkan  | 0   | none | pp128  | 62.89 ± 5.75 |
| qwen2 3B Q4_K - Medium | 1.8G | 3.1 B  | Vulkan  | 0   | none | pp512  | 59.86 ± 2.19 |
| qwen2 3B Q4_K - Medium | 1.8G | 3.1 B  | Vulkan  | 0   | none | pp2048 | 55.61 ± 0.57 |
| qwen2 3B Q4_K - Medium | 1.8G | 3.1 B  | Vulkan  | 0   | none | tg128  | 14.08 ± 0.16 |
build: abc123
"""


VULKAN_OUTPUT = """
ggml_vulkan: found device Vulkan0
| model | size | params | backend | ngl | dev     | test   | t/s            |
| qwen  | 1.8G | 3.1 B  | Vulkan  | -1  | Vulkan0 | pp128  | 124.80 ± 10.23 |
| qwen  | 1.8G | 3.1 B  | Vulkan  | -1  | Vulkan0 | pp512  | 118.46 ± 3.80  |
| qwen  | 1.8G | 3.1 B  | Vulkan  | -1  | Vulkan0 | pp2048 | 97.45 ± 3.26   |
| qwen  | 1.8G | 3.1 B  | Vulkan  | -1  | Vulkan0 | tg128  | 10.81 ± 1.26   |
"""


CUDA_OUTPUT = """
| backend | ngl | dev   | test  | t/s          | model |
| CUDA    | 99  | CUDA0 | pp512 | 240.0 ± 4.0 | qwen  |
| CUDA    | 99  | CUDA0 | tg128 | 41.0 ± 0.5  | qwen  |
"""


def test_parser_extracts_cpu_rows_without_trusting_backend_column() -> None:
    measurements = parse_llama_bench_output(CPU_OUTPUT, repetitions=5)

    assert [(item.kind, item.tokens) for item in measurements] == [
        (BenchmarkMeasurementKind.PREFILL, 128),
        (BenchmarkMeasurementKind.PREFILL, 512),
        (BenchmarkMeasurementKind.PREFILL, 2048),
        (BenchmarkMeasurementKind.GENERATION, 128),
    ]
    assert measurements[1].mean_tokens_per_second == 59.86
    assert measurements[1].stddev_tokens_per_second == 2.19
    assert measurements[1].raw_backend == "Vulkan"
    assert measurements[1].raw_device == "none"


def test_parser_extracts_vulkan_and_cuda_rows_with_spacing_variants() -> None:
    vulkan = parse_llama_bench_output(VULKAN_OUTPUT, repetitions=5)
    cuda = parse_llama_bench_output(CUDA_OUTPUT, repetitions=3)

    assert vulkan[1].raw_device == "Vulkan0"
    assert vulkan[1].mean_tokens_per_second == 118.46
    assert cuda[0].raw_device == "CUDA0"
    assert cuda[0].kind is BenchmarkMeasurementKind.PREFILL


def test_parser_rejects_malformed_tps_and_missing_rows() -> None:
    malformed = """
    | test | t/s |
    | pp128 | very fast |
    """
    with pytest.raises(BenchmarkParseError):
        parse_llama_bench_output(malformed)

    with pytest.raises(BenchmarkParseError):
        parse_llama_bench_output("load_backend only\nno table")


def test_command_building_cpu_vulkan_and_cuda(tmp_path: Path) -> None:
    model = tmp_path / "model with spaces.gguf"
    model.write_bytes(b"gguf")

    cpu = build_llama_bench_command(
        "llama-bench",
        BenchmarkRequest(
            artifact=_artifact(model),
            runtime=_runtime(ngl=0),
            backend=ComputeBackend.CPU,
            device="none",
            gpu_layers=BenchmarkGpuLayers.count_layers(0),
        ),
    )
    assert cpu[cpu.index("-dev") + 1] == "none"
    assert cpu[cpu.index("-ngl") + 1] == "0"
    assert cpu[cpu.index("-p") + 1] == "128,512,2048"

    vulkan = build_llama_bench_command(
        "llama-bench",
        BenchmarkRequest(
            artifact=_artifact(model),
            runtime=_runtime(ngl=-1),
            backend=ComputeBackend.VULKAN,
            device="Vulkan0",
            gpu_layers=BenchmarkGpuLayers.full(),
        ),
    )
    assert vulkan[vulkan.index("-dev") + 1] == "Vulkan0"
    assert vulkan[vulkan.index("-ngl") + 1] == "-1"

    cuda = build_llama_bench_command(
        "llama-bench",
        BenchmarkRequest(
            artifact=_artifact(model),
            runtime=_runtime(ngl=-1),
            backend=ComputeBackend.CUDA,
            device="CUDA0",
            gpu_layers=BenchmarkGpuLayers.full(),
            prefill_sizes=(128, 512, 2048),
            generation_sizes=(128,),
            repetitions=5,
        ),
    )
    assert cuda == (
        "llama-bench",
        "-m",
        str(model),
        "-dev",
        "CUDA0",
        "-ngl",
        "-1",
        "-p",
        "128,512,2048",
        "-n",
        "128",
        "-r",
        "5",
    )
    forbidden_llama_cli_flags = {
        "--ctx-size",
        "--single-turn",
        "--prompt",
        "--no-display-prompt",
        "--no-show-timings",
        "--simple-io",
    }
    assert forbidden_llama_cli_flags.isdisjoint(cuda)


def test_runner_parses_successful_process(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    backend = _FakeExecutionBackend(stdout=CPU_OUTPUT)
    runner = LlamaBenchRunner(backend=backend, llama_bench_path=_executable(tmp_path))

    observation = runner.run(
        BenchmarkRequest(
            artifact=_artifact(model),
            runtime=_runtime(ngl=0),
            backend=ComputeBackend.CPU,
            device="none",
            gpu_layers=BenchmarkGpuLayers.count_layers(0),
        )
    )

    assert observation.success is True
    assert len(observation.measurements) == 4
    assert backend.requests[0].command[0] == str(runner._llama_bench)


def test_runner_failed_process_preserves_argv_exit_code_and_output(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    backend = _FailingExecutionBackend(
        stdout="loading model\n",
        stderr="unknown argument: --ctx-size\n",
        exit_code=1,
    )
    runner = LlamaBenchRunner(backend=backend, llama_bench_path=_executable(tmp_path))

    observation = runner.run(
        BenchmarkRequest(
            artifact=_artifact(model),
            runtime=_runtime(ngl=-1),
            backend=ComputeBackend.CUDA,
            device="CUDA0",
            gpu_layers=BenchmarkGpuLayers.full(),
        )
    )

    assert observation.success is False
    assert observation.exit_code == 1
    assert observation.command == backend.requests[0].command
    assert "argv:" in (observation.message or "")
    assert "-dev CUDA0 -ngl -1" in (observation.message or "")
    assert "exit_code: 1" in (observation.message or "")
    assert "stderr: unknown argument: --ctx-size" in (observation.message or "")
    assert "stdout: loading model" in (observation.message or "")


def test_runner_reports_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkUnavailableError):
        LlamaBenchRunner(
            backend=_FakeExecutionBackend(stdout=""),
            llama_bench_path=tmp_path / "missing",
        )


def test_llama_bench_version_probe_failure_does_not_block_available_binary(
    tmp_path: Path,
) -> None:
    capability = inspect_llama_bench(
        backend=_FailingExecutionBackend(
            stdout="",
            stderr="unknown argument: --version",
            exit_code=1,
        ),
        llama_bench_path=_executable(tmp_path),
    )

    assert capability.binary_status is LlamaBenchBinaryStatus.AVAILABLE
    assert capability.version_text is None
    assert "unknown argument: --version" in (capability.message or "")


def test_store_roundtrip_and_safety(tmp_path: Path) -> None:
    record = _record(tmp_path)
    store = BenchmarkStore(root=tmp_path / "benchmarks")

    path = store.save(record)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["benchmark"]["identity"]["benchmark_id"] == record.identity.benchmark_id
    assert store.load(record.identity.benchmark_id) == record
    assert store.save(record) == path
    assert store.list_ids() == [record.identity.benchmark_id]

    with pytest.raises(BenchmarkStoreError, match="already exists"):
        store.save(record.model_copy(update={"notes": ["different"]}))
    with pytest.raises(InvalidBenchmarkIdError):
        store.path_for("../bad")


def test_matrix_runs_cpu_and_selected_backend_with_partial_failure(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    runner = _SequenceBenchmarkRunner(
        [
            _observation(CPU_OUTPUT),
            BenchmarkParseError("broken output"),
        ]
    )
    matrix = BenchmarkMatrixRunner(
        execution_backend=_FakeExecutionBackend(stdout="llama-bench version"),
        llama_bench_runner=runner,  # type: ignore[arg-type]
        store=BenchmarkStore(root=tmp_path / "benchmarks"),
        llama_bench_path=_executable(tmp_path),
    )

    result = matrix.run(
        BenchmarkMatrixRequest(
            hardware=_hardware(),
            artifact=_artifact(model),
            runtime=_runtime(ngl=12),
            backend_selection=_selection(ComputeBackend.VULKAN),
            runtime_capability=_runtime_capability(),
        )
    )

    assert len(result.completed) == 1
    assert result.completed[0].record.requested_backend is ComputeBackend.CPU
    assert len(result.failed) == 1
    assert result.failed[0].request.backend is ComputeBackend.VULKAN
    assert len(result.records) == 1


def test_aggregate_reports_descriptive_speedups(tmp_path: Path) -> None:
    cpu = _record(tmp_path, output=CPU_OUTPUT, backend=ComputeBackend.CPU, device="none")
    vulkan = _record(
        tmp_path,
        output=VULKAN_OUTPUT,
        backend=ComputeBackend.VULKAN,
        device="Vulkan0",
        gpu_layers=BenchmarkGpuLayers.full(),
    )

    aggregate = aggregate_benchmark_records([cpu, vulkan])

    pp512 = next(
        item
        for item in aggregate.comparisons
        if item.kind is BenchmarkMeasurementKind.PREFILL and item.tokens == 512
    )
    tg128 = next(
        item
        for item in aggregate.comparisons
        if item.kind is BenchmarkMeasurementKind.GENERATION and item.tokens == 128
    )
    assert pp512.speedup == pytest.approx(118.46 / 59.86)
    assert tg128.speedup == pytest.approx(10.81 / 14.08)


class _FakeExecutionBackend:
    def __init__(self, *, stdout: str, exit_code: int = 0) -> None:
        self.requests: list[ExecutionRequest] = []
        self.stdout = stdout
        self.exit_code = exit_code

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return ExecutionResult(
            stdout=self.stdout,
            stderr="",
            observation=ExecutionObservation(
                success=self.exit_code == 0,
                duration_seconds=0.25,
                exit_code=self.exit_code,
                peak_ram_bytes=None,
                peak_vram_bytes=None,
            ),
        )


class _FailingExecutionBackend:
    def __init__(self, *, stdout: str, stderr: str, exit_code: int) -> None:
        self.requests: list[ExecutionRequest] = []
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        result = ExecutionResult(
            stdout=self.stdout,
            stderr=self.stderr,
            observation=ExecutionObservation(
                success=False,
                duration_seconds=0.25,
                exit_code=self.exit_code,
                peak_ram_bytes=None,
                peak_vram_bytes=None,
            ),
        )
        raise ExecutionFailedError("Command failed with exit code 1.", result)


class _SequenceBenchmarkRunner:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[BenchmarkRequest] = []

    def run(self, request: BenchmarkRequest) -> BenchmarkObservation:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "llama-bench"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _artifact(path: Path) -> ModelArtifact:
    return ModelArtifact(
        repo_id="owner/repo",
        revision="main",
        filename=path.name,
        format="gguf",
        quantization="Q4_K_M",
        size_bytes=path.stat().st_size,
        local_path=path,
        sha256="1" * 64,
        is_downloaded=True,
        is_verified=True,
    )


def _runtime(*, ngl: int) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value="4096",
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value=str(ngl),
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test CPU"),
        memory=MemoryInfo(total_bytes=16, available_bytes=8),
    )


def _selection(backend: ComputeBackend) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE
        if backend is ComputeBackend.VULKAN
        else RuntimeBackendSelectionReason.CPU_FALLBACK,
    )


def _runtime_capability() -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_path="/usr/bin/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CPU,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
            ),
            LlamaCppBackendCapability(
                backend=ComputeBackend.VULKAN,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
                devices=[
                    LlamaCppRuntimeDevice(
                        backend=ComputeBackend.VULKAN,
                        runtime_id="Vulkan0",
                    )
                ],
            ),
        ],
    )


def _observation(output: str) -> BenchmarkObservation:
    return BenchmarkObservation(
        success=True,
        measurements=parse_llama_bench_output(output, repetitions=5),
        repetitions=5,
        duration_seconds=1.0,
        exit_code=0,
        raw_stdout=output,
    )


def _record(
    tmp_path: Path,
    *,
    output: str = CPU_OUTPUT,
    backend: ComputeBackend = ComputeBackend.CPU,
    device: str | None = "none",
    gpu_layers: BenchmarkGpuLayers | None = None,
) -> BenchmarkRecord:
    model = tmp_path / "record-model.gguf"
    model.write_bytes(b"gguf")
    effective_layers = gpu_layers or BenchmarkGpuLayers.count_layers(0)
    request = BenchmarkRequest(
        artifact=_artifact(model),
        runtime=_runtime(ngl=0),
        backend=backend,
        device=device,
        gpu_layers=effective_layers,
    )
    return BenchmarkRecord.create(
        hardware=_hardware(),
        request=request,
        observation=_observation(output),
        llama_bench_capability=LlamaBenchCapability(
            binary_path="/usr/bin/llama-bench",
            binary_status=LlamaBenchBinaryStatus.AVAILABLE,
            version_text="llama-bench test",
        ),
    )
