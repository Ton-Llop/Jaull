"""Run small benchmark matrices without putting loops in the UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jaull.benchmarks.errors import (
    BenchmarkParseError,
    BenchmarkRunnerError,
    BenchmarkStoreError,
    BenchmarkUnavailableError,
)
from jaull.benchmarks.storage import BenchmarkStore
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkRecord,
    BenchmarkRequest,
    BenchmarkRunResult,
    LlamaBenchBinaryStatus,
)
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.runtime import (
    LlamaCppBackendCapabilityState,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.ports import ExecutionBackendProtocol
from jaull.runtime.llama_bench_capability import inspect_llama_bench
from jaull.runtime.llama_bench_runner import LlamaBenchRunner


class BenchmarkMatrixRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    hardware: HardwareProfile
    artifact: ModelArtifact
    runtime: RuntimeRecommendation
    backend_selection: RuntimeBackendSelection
    runtime_capability: LlamaCppRuntimeCapability
    prefill_sizes: tuple[int, ...] = (128, 512, 2048)
    generation_sizes: tuple[int, ...] = (128,)
    repetitions: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=900.0, gt=0.0)
    include_cpu: bool = True
    include_selected_backend: bool = True
    selected_gpu_layers: tuple[BenchmarkGpuLayers, ...] = Field(default_factory=tuple)
    persist: bool = True
    notes: list[str] = Field(default_factory=list)


class BenchmarkMatrixSkip(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    device: str | None = None
    gpu_layers: BenchmarkGpuLayers | None = None
    reason: str


class BenchmarkMatrixFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: BenchmarkRequest
    message: str
    record: BenchmarkRecord | None = None
    persisted_path: Path | None = None


class BenchmarkMatrixResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    completed: list[BenchmarkRunResult] = Field(default_factory=list)
    skipped: list[BenchmarkMatrixSkip] = Field(default_factory=list)
    failed: list[BenchmarkMatrixFailure] = Field(default_factory=list)
    records: list[BenchmarkRecord] = Field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkMatrixRunner:
    execution_backend: ExecutionBackendProtocol
    llama_bench_runner: LlamaBenchRunner
    store: BenchmarkStore | None = None
    llama_bench_path: str | Path | None = None
    capability_timeout_seconds: float = 10.0

    def run(self, request: BenchmarkMatrixRequest) -> BenchmarkMatrixResult:
        if request.runtime.runtime is not RuntimeName.LLAMA_CPP:
            raise BenchmarkRunnerError("Benchmarking currently requires llama.cpp runtime.")

        capability = inspect_llama_bench(
            backend=self.execution_backend,
            llama_bench_path=self.llama_bench_path,
            timeout_seconds=self.capability_timeout_seconds,
        )
        if capability.binary_status is not LlamaBenchBinaryStatus.AVAILABLE:
            raise BenchmarkUnavailableError(
                capability.message or "llama-bench is not available."
            )

        completed: list[BenchmarkRunResult] = []
        skipped: list[BenchmarkMatrixSkip] = []
        failed: list[BenchmarkMatrixFailure] = []
        records: list[BenchmarkRecord] = []

        configs, config_skips = _build_requests(request)
        skipped.extend(config_skips)
        for benchmark_request in configs:
            try:
                observation = self.llama_bench_runner.run(benchmark_request)
            except (BenchmarkParseError, BenchmarkRunnerError) as exc:
                failed.append(
                    BenchmarkMatrixFailure(
                        request=benchmark_request,
                        message=str(exc),
                    )
                )
                continue

            record = BenchmarkRecord.create(
                hardware=request.hardware,
                request=benchmark_request,
                observation=observation,
                llama_bench_capability=capability,
            )
            records.append(record)
            persisted_path = None
            if request.persist:
                if self.store is None:
                    failed.append(
                        BenchmarkMatrixFailure(
                            request=benchmark_request,
                            message=(
                                "Benchmark persistence was requested but no "
                                "BenchmarkStore is configured."
                            ),
                            record=record,
                        )
                    )
                    continue
                try:
                    persisted_path = self.store.save(record)
                except BenchmarkStoreError as exc:
                    failed.append(
                        BenchmarkMatrixFailure(
                            request=benchmark_request,
                            message=(
                                "Benchmark completed, but its record could not "
                                f"be persisted: {exc}"
                            ),
                            record=record,
                        )
                    )
                    continue

            run_result = BenchmarkRunResult(
                record=record,
                persisted_path=persisted_path,
            )
            if observation.success:
                completed.append(run_result)
            else:
                failed.append(
                    BenchmarkMatrixFailure(
                        request=benchmark_request,
                        message=observation.message or "Benchmark process failed.",
                        record=record,
                        persisted_path=persisted_path,
                    )
                )

        return BenchmarkMatrixResult(
            completed=completed,
            skipped=skipped,
            failed=failed,
            records=records,
        )


def _build_requests(
    matrix: BenchmarkMatrixRequest,
) -> tuple[list[BenchmarkRequest], list[BenchmarkMatrixSkip]]:
    requests: list[BenchmarkRequest] = []
    skipped: list[BenchmarkMatrixSkip] = []

    if matrix.include_cpu:
        requests.append(
            _request(
                matrix,
                backend=ComputeBackend.CPU,
                device="none",
                gpu_layers=BenchmarkGpuLayers.count_layers(0),
            )
        )

    selected = matrix.backend_selection.selected_backend
    if matrix.include_selected_backend and selected is not ComputeBackend.CPU:
        capability = next(
            (
                item
                for item in matrix.runtime_capability.backend_capabilities
                if item.backend is selected
            ),
            None,
        )
        if capability is None or (
            capability.state is not LlamaCppBackendCapabilityState.CONFIRMED
        ):
            skipped.append(
                BenchmarkMatrixSkip(
                    backend=selected,
                    reason=f"{selected.value} is not exposed by llama.cpp runtime.",
                )
            )
        elif not capability.devices:
            skipped.append(
                BenchmarkMatrixSkip(
                    backend=selected,
                    reason=f"{selected.value} has no runtime device id.",
                )
            )
        else:
            device = capability.devices[0].runtime_id
            layers = matrix.selected_gpu_layers or (_runtime_gpu_layers(matrix.runtime),)
            for gpu_layers in layers:
                requests.append(
                    _request(
                        matrix,
                        backend=selected,
                        device=device,
                        gpu_layers=gpu_layers,
                    )
                )

    return requests, skipped


def _request(
    matrix: BenchmarkMatrixRequest,
    *,
    backend: ComputeBackend,
    device: str | None,
    gpu_layers: BenchmarkGpuLayers,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        artifact=matrix.artifact,
        runtime=_runtime_with_gpu_layers(matrix.runtime, gpu_layers),
        backend=backend,
        device=device,
        gpu_layers=gpu_layers,
        prefill_sizes=matrix.prefill_sizes,
        generation_sizes=matrix.generation_sizes,
        repetitions=matrix.repetitions,
        timeout_seconds=matrix.timeout_seconds,
        notes=matrix.notes,
    )


def _runtime_gpu_layers(runtime: RuntimeRecommendation) -> BenchmarkGpuLayers:
    raw = next(
        (flag.value for flag in runtime.flags if flag.name == "--n-gpu-layers"),
        None,
    )
    if raw is None:
        return BenchmarkGpuLayers.count_layers(0)
    try:
        value = int(raw)
    except ValueError:
        if raw.casefold() in {"all", "full", "full_offload"}:
            return BenchmarkGpuLayers.full()
        return BenchmarkGpuLayers.count_layers(0)
    if value < 0:
        return BenchmarkGpuLayers.full()
    return BenchmarkGpuLayers.count_layers(value)


def _runtime_with_gpu_layers(
    runtime: RuntimeRecommendation,
    gpu_layers: BenchmarkGpuLayers,
) -> RuntimeRecommendation:
    value = "all" if gpu_layers.full_offload else str(gpu_layers.count)
    flags = [
        flag
        for flag in runtime.flags
        if flag.name != "--n-gpu-layers"
    ]
    flags.append(
        RuntimeFlag(
            name="--n-gpu-layers",
            value=value,
            source=RuntimeFlagSource.USER_INPUT,
            explanation="benchmark configuration",
        )
    )
    return runtime.model_copy(update={"flags": flags})


__all__ = [
    "BenchmarkMatrixFailure",
    "BenchmarkMatrixRequest",
    "BenchmarkMatrixResult",
    "BenchmarkMatrixRunner",
    "BenchmarkMatrixSkip",
]
