from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, DataTable

from jaull.benchmarks.errors import BenchmarkUnavailableError
from jaull.benchmarks.matrix import (
    BenchmarkMatrixFailure,
    BenchmarkMatrixRequest,
    BenchmarkMatrixResult,
)
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRecord,
    BenchmarkRequest,
    BenchmarkRunResult,
    LlamaBenchBinaryStatus,
    LlamaBenchCapability,
)
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen
from jaull.tui.screens.recommendation_results import RecommendationResultsScreen
from jaull.workflow.state import RecommendationWorkflowState
from tests.test_tui_recommendation_execution import (
    _backend_selection,
    _FakeAdvisor,
    _recommendation,
    _run,
    _runtime_capability,
    _visible_text,
    _wait_until,
)


def test_results_screen_can_open_benchmark_screen(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = RecommendationWorkflowState(recommendations=[_recommendation()])
        app = JaullApp(advisor=_BenchmarkAdvisor(tmp_path))  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)
            assert screen.query_one("#res-benchmark-0", Button).disabled is False

            screen.query_one("#res-benchmark-0", Button).press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, RecommendationBenchmarkScreen)

    _run(scenario())


def test_benchmark_screen_shows_successful_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        advisor = _BenchmarkAdvisor(tmp_path)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationBenchmarkScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _run_workers_inline(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(
                pilot,
                lambda: "Benchmark complete" in _visible_text(screen),
            )

            text = _visible_text(screen)
            assert screen.query(DataTable)
            assert "Prompt processing 512" in text
            assert "1.98x" in text
            assert advisor.matrix_calls == 1
            assert advisor.last_matrix_request is not None
            assert advisor.last_matrix_request.include_cpu is False
            assert advisor.last_matrix_request.include_selected_backend is True
            assert advisor.last_matrix_request.selected_gpu_layers == (
                BenchmarkGpuLayers.full(),
            )
            assert "Running llama-bench: vulkan · device Vulkan0 · ngl all" in text

    _run(scenario())


def test_benchmark_screen_runs_transformers_single_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runtime = RuntimeRecommendation(
            runtime=RuntimeName.TRANSFORMERS,
            flags=[
                RuntimeFlag(
                    name="device_map",
                    value="cpu",
                    source=RuntimeFlagSource.HARDWARE,
                    explanation="test",
                )
            ],
            confidence=EstimationConfidence.HIGH,
        )
        advisor = _BenchmarkAdvisor(tmp_path)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(
                RecommendationBenchmarkScreen(
                    _recommendation(
                        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                        quantization="float32",
                        runtime=runtime,
                    )
                )
            )
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _run_workers_inline(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(
                pilot,
                lambda: "Benchmark complete" in _visible_text(screen),
            )

            assert advisor.benchmark_calls == 1
            assert advisor.matrix_calls == 0
            assert advisor.last_benchmark_request is not None
            assert advisor.last_benchmark_request.runtime is runtime
            assert advisor.last_benchmark_request.artifact.format == "transformers"
            assert advisor.operations == ["hardware", "select_backend"]
            text = _visible_text(screen)
            assert screen.query(DataTable)
            assert "Running Transformers benchmark: cpu" in text
            assert "Startup" in text
            assert "Memory" in text

    _run(scenario())


def test_benchmark_screen_shows_missing_llama_bench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        advisor = _BenchmarkAdvisor(
            tmp_path,
            failure=BenchmarkUnavailableError("llama-bench executable not found"),
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationBenchmarkScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _run_workers_inline(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(
                pilot,
                lambda: "llama-bench executable not found" in _visible_text(screen),
            )

    _run(scenario())


def test_benchmark_screen_keeps_partial_failure_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        advisor = _BenchmarkAdvisor(tmp_path, partial_failure=True)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationBenchmarkScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _run_workers_inline(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(
                pilot,
                lambda: "Benchmark complete" in _visible_text(screen),
            )

            text = _visible_text(screen)
            assert screen.query(DataTable)
            assert "Vulkan run failed" in text
            assert screen.query_one("#benchmark-details", Button).disabled is False

    _run(scenario())


class _BenchmarkAdvisor(_FakeAdvisor):
    def __init__(
        self,
        tmp_path: Path,
        *,
        failure: Exception | None = None,
        partial_failure: bool = False,
    ) -> None:
        super().__init__(
            backend_selection=_backend_selection(ComputeBackend.VULKAN),
        )
        self.tmp_path = tmp_path
        self.failure = failure
        self.partial_failure = partial_failure
        self.benchmark_calls = 0
        self.matrix_calls = 0
        self.last_benchmark_request: BenchmarkRequest | None = None
        self.last_matrix_request: BenchmarkMatrixRequest | None = None

    def inspect_llama_cpp_runtime(self) -> object:
        return _runtime_capability(ComputeBackend.VULKAN)

    def run_benchmark_matrix(
        self,
        request: BenchmarkMatrixRequest,
    ) -> BenchmarkMatrixResult:
        self.matrix_calls += 1
        self.last_matrix_request = request
        if self.failure is not None:
            raise self.failure
        cpu = _record(request, self.tmp_path, backend=ComputeBackend.CPU, device="none")
        completed = [
            BenchmarkRunResult(
                record=cpu,
                persisted_path=self.tmp_path / f"{cpu.identity.benchmark_id}.json",
            )
        ]
        failed: list[BenchmarkMatrixFailure] = []
        if self.partial_failure:
            failed.append(
                BenchmarkMatrixFailure(
                    request=BenchmarkRequest(
                        artifact=request.artifact,
                        runtime=request.runtime,
                        backend=ComputeBackend.VULKAN,
                        device="Vulkan0",
                        gpu_layers=BenchmarkGpuLayers.count_layers(12),
                    ),
                    message="Vulkan run failed",
                )
            )
            return BenchmarkMatrixResult(
                completed=completed,
                failed=failed,
                records=[cpu],
            )

        vulkan = _record(
            request,
            self.tmp_path,
            backend=ComputeBackend.VULKAN,
            device="Vulkan0",
            gpu_layers=BenchmarkGpuLayers.count_layers(12),
        )
        completed.append(
            BenchmarkRunResult(
                record=vulkan,
                persisted_path=self.tmp_path / f"{vulkan.identity.benchmark_id}.json",
            )
        )
        return BenchmarkMatrixResult(
            completed=completed,
            failed=failed,
            records=[cpu, vulkan],
        )

    def run_benchmark(
        self,
        request: BenchmarkRequest,
        *,
        hardware: object,
        persist: bool = True,
    ) -> BenchmarkRunResult:
        del hardware, persist
        self.benchmark_calls += 1
        self.last_benchmark_request = request
        record = BenchmarkRecord.create(
            hardware=self.hardware,
            request=request,
            observation=_transformers_observation(),
            runtime_capability=None,
        )
        return BenchmarkRunResult(
            record=record,
            persisted_path=self.tmp_path / f"{record.identity.benchmark_id}.json",
        )


def _transformers_observation() -> BenchmarkObservation:
    return BenchmarkObservation(
        success=True,
        repetitions=3,
        duration_seconds=20.0,
        methodology="transformers_isolated_inference_v2",
        model_load_seconds=12.0,
        warmup_seconds=1.0,
        time_to_first_token_seconds=0.31,
        time_to_first_token_stddev_seconds=0.02,
        generation_latency_seconds=4.0,
        generation_latency_stddev_seconds=0.3,
        peak_ram_bytes=2_600_000_000,
        exit_code=0,
        measurements=[
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.PREFILL,
                tokens=128,
                mean_tokens_per_second=40.0,
                stddev_tokens_per_second=2.0,
                mean_duration_seconds=3.2,
                stddev_duration_seconds=0.2,
                repetitions=3,
                source_label="pp128",
            ),
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.GENERATION,
                tokens=64,
                mean_tokens_per_second=9.0,
                stddev_tokens_per_second=0.5,
                mean_duration_seconds=7.1,
                stddev_duration_seconds=0.3,
                repetitions=3,
                source_label="tg64",
            ),
        ],
    )


class _InlineExecutor:
    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future[None]:
        fn(*args, **kwargs)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        return None


def _run_workers_inline(
    screen: RecommendationBenchmarkScreen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "_executor", _InlineExecutor())


def _record(
    matrix: BenchmarkMatrixRequest,
    tmp_path: Path,
    *,
    backend: ComputeBackend,
    device: str | None,
    gpu_layers: BenchmarkGpuLayers | None = None,
) -> BenchmarkRecord:
    effective_layers = gpu_layers or BenchmarkGpuLayers.count_layers(0)
    request = BenchmarkRequest(
        artifact=matrix.artifact,
        runtime=matrix.runtime,
        backend=backend,
        device=device,
        gpu_layers=effective_layers,
    )
    return BenchmarkRecord.create(
        hardware=matrix.hardware,
        request=request,
        observation=_observation(backend),
        llama_bench_capability=LlamaBenchCapability(
            binary_path=str(tmp_path / "llama-bench"),
            binary_status=LlamaBenchBinaryStatus.AVAILABLE,
            version_text="llama-bench test",
        ),
    )


def _observation(backend: ComputeBackend) -> BenchmarkObservation:
    pp512 = 59.86 if backend is ComputeBackend.CPU else 118.46
    tg128 = 14.08 if backend is ComputeBackend.CPU else 10.81
    return BenchmarkObservation(
        success=True,
        repetitions=5,
        duration_seconds=1.0,
        exit_code=0,
        measurements=[
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.PREFILL,
                tokens=512,
                mean_tokens_per_second=pp512,
                stddev_tokens_per_second=1.0,
                repetitions=5,
                source_label="pp512",
            ),
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.GENERATION,
                tokens=128,
                mean_tokens_per_second=tg128,
                stddev_tokens_per_second=1.0,
                repetitions=5,
                source_label="tg128",
            ),
        ],
    )
