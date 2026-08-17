from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from jaull.artifacts.errors import ArtifactDownloadError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRecord,
    BenchmarkRequest,
)
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionObservation,
    ExecutionResult,
    InferenceResult,
)
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
    PreparedExecutionPlan,
)
from jaull.domain.experiments import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentRunResult,
    ExperimentWorkload,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    PyTorchRuntimeCapability,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.benchmark_comparison import (
    BenchmarkComparison,
    compare_benchmark_records,
)
from jaull.evaluation.experiments import build_experiment_record
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution_plans import execution_plan_for_recommendation
from jaull.experiments.errors import (
    ExperimentNotReadyError,
    ExperimentPersistenceError,
)
from jaull.recommendation.models import ModelRecommendation, ScoreBreakdown
from jaull.recommendation.policies import LicenseCategory
from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness
from jaull.runtime.pytorch_capability import evaluate_pytorch_execution_readiness
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_execution import (
    InferencePrompt,
    InferenceResponse,
    RecommendationExecutionScreen,
)
from jaull.tui.screens.recommendation_results import (
    ExecutionPathBenchmarkCompareScreen,
    ExecutionPathsScreen,
    ExportReportModal,
    RecommendationCompareScreen,
    RecommendationDetailsScreen,
    RecommendationResultsScreen,
)
from jaull.tui.screens.recommendation_validation import (
    RecommendationValidationScreen,
)
from jaull.workflow.container import ServiceContainer
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import (
    GIB,
    candidate,
    gguf_analysis,
    hardware,
    memory_estimate,
)


def _run(coro: Any) -> None:
    asyncio.run(coro)


async def _settle(pilot: Any, attempts: int = 40) -> None:
    for _ in range(attempts):
        await pilot.pause()
        await asyncio.sleep(0.02)


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )


def _artifact(**updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "org/Tiny-GGUF",
        "revision": "main",
        "filename": "tiny-q4_k_m.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": 4,
        "local_path": Path("/tmp/tiny-q4_k_m.gguf"),
        "sha256": "deadbeef",
        "is_downloaded": False,
        "is_verified": False,
    }
    data.update(updates)
    return ModelArtifact(**data)


def _observation(
    *,
    success: bool = True,
    duration_seconds: float = 0.1,
    peak_ram_bytes: int | None = 512 * 1024**2,
    peak_vram_bytes: int | None = 256 * 1024**2,
    exit_code: int | None = 0,
    failure_reason: ExecutionFailureReason | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=duration_seconds,
        peak_ram_bytes=peak_ram_bytes,
        peak_vram_bytes=peak_vram_bytes,
        exit_code=exit_code,
        failure_reason=failure_reason,
    )


def _runtime(*, ctx_size: int = 4096, n_gpu_layers: int = 12) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(ctx_size),
                source=RuntimeFlagSource.ESTIMATE,
                explanation="from estimate",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value=str(n_gpu_layers),
                source=RuntimeFlagSource.HARDWARE,
                explanation="from hardware",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _transformers_runtime() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        flags=[
            RuntimeFlag(
                name="device_map",
                value="auto",
                source=RuntimeFlagSource.HARDWARE,
                explanation="from hardware",
            )
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _recommendation(
    *,
    rank: int = 1,
    repo_id: str = "org/Tiny-GGUF",
    quantization: str = "Q4_K_M",
    runtime: RuntimeRecommendation | None = None,
) -> ModelRecommendation:
    config = InferenceConfiguration(
        context_length=4096,
        quantization=quantization,
    )
    analysis = gguf_analysis(repo_id=repo_id, quantizations=(quantization,))
    estimate = memory_estimate(
        analysis,
        config,
        total_bytes=4 * GIB,
        status=CompatibilityStatus.COMFORTABLE,
    ).model_copy(update={"runtime_recommendation": runtime or _runtime()})
    base_candidate = candidate(repo_id=repo_id).model_copy(
        update={"repository_type": RepositoryType.GGUF}
    )
    evaluated = EvaluatedCandidate(
        candidate=base_candidate,
        analysis=analysis,
        selected_configuration=config,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )
    return ModelRecommendation(
        rank=rank,
        evaluated=evaluated,
        score=ScoreBreakdown(total=0.92),
        status=CompatibilityStatus.COMFORTABLE,
        confidence=EstimationConfidence.HIGH,
        license_category=LicenseCategory.COMMERCIAL_ALLOWED,
        reasons=["fixture"],
    )


def _recommendation_with_plan(
    *,
    repo_id: str,
    identity: ModelIdentity,
    format_: ArtifactVariantFormat,
    quantization: str | None = None,
    precision: str | None = None,
    runtime: RuntimeName = RuntimeName.LLAMA_CPP,
) -> ModelRecommendation:
    recommendation = _recommendation(
        repo_id=repo_id,
        quantization=quantization or "Q4_K_M",
        runtime=_runtime() if runtime is RuntimeName.LLAMA_CPP else _transformers_runtime(),
    )
    artifact = ArtifactVariant(
        model_identity=identity,
        repo_id=repo_id,
        revision="main",
        format=format_,
        filename=repo_id,
        quantization=quantization,
        precision=precision,
        compatible_runtimes=[runtime],
    )
    plan = ExecutionPlan(
        plan_id=f"{repo_id}:{artifact.label}",
        model_identity=identity,
        artifact=artifact,
        runtime=_runtime() if runtime is RuntimeName.LLAMA_CPP else _transformers_runtime(),
        runtime_family=runtime,
    )
    return recommendation.model_copy(update={"plan": plan})


class _FakeAdvisor:
    def __init__(
        self,
        *,
        artifact: ModelArtifact | None = None,
        fail_download: Exception | None = None,
        fail_run: Exception | None = None,
        prepare_delay: float = 0.0,
        response_text: str = "generated from tui",
        observation: ExecutionObservation | None = None,
        experiment_observation: ExecutionObservation | None = None,
        experiment_error: Exception | None = None,
        experiment_persisted_path: Path | None = Path("/tmp/jaull-experiment.json"),
        backend_selection: RuntimeBackendSelection | None = None,
        execution_plans: list[ExecutionPlan] | None = None,
        benchmark_comparison: BenchmarkComparison | None = None,
    ) -> None:
        self.services = _services()
        self.artifact = artifact or _artifact()
        self.fail_download = fail_download
        self.fail_run = fail_run
        self.experiment_error = experiment_error
        self.prepare_delay = prepare_delay
        self.response_text = response_text
        self.observation = observation or _observation()
        self.experiment_observation = experiment_observation or _observation()
        self.experiment_persisted_path = experiment_persisted_path
        self.hardware = hardware()
        self.backend_selection = backend_selection or _backend_selection(ComputeBackend.CPU)
        self.execution_plans = execution_plans
        self.benchmark_comparison = benchmark_comparison
        self.operations: list[str] = []
        self.resolved: list[tuple[str, str | None, str | None]] = []
        self.downloaded: list[ModelArtifact] = []
        self.verified: list[ModelArtifact] = []
        self.runs: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []
        self.experiment_requests: list[object] = []
        self.experiment_records: list[ExperimentRecord] = []
        self.prepared_plans: list[object] = []

    def resolve_artifact(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        if self.prepare_delay:
            time.sleep(self.prepare_delay)
        self.operations.append("resolve")
        self.resolved.append((repo_id, quantization, revision))
        return self.artifact

    def download_artifact(
        self,
        artifact: ModelArtifact,
        on_progress: object | None = None,
    ) -> ModelArtifact:
        del on_progress
        self.operations.append("download")
        self.downloaded.append(artifact)
        if self.fail_download is not None:
            raise self.fail_download
        self.artifact = artifact.model_copy(update={"is_downloaded": True})
        return self.artifact

    def verify_artifact(self, artifact: ModelArtifact, *, full: bool = False) -> ModelArtifact:
        del full
        self.operations.append("verify")
        self.verified.append(artifact)
        self.artifact = artifact.model_copy(update={"is_verified": True})
        return self.artifact

    def run_artifact(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        self.operations.append("run")
        self.runs.append((artifact, prompt, runtime))
        if self.fail_run is not None:
            raise self.fail_run
        return InferenceResult(
            text=self.response_text,
            runtime=runtime.runtime.value if runtime is not None else "llama.cpp",
            model_path=artifact.local_path or Path("/tmp/tiny-q4_k_m.gguf"),
            observation=self.observation,
        )

    def scan_hardware(self) -> object:
        self.operations.append("hardware")
        return self.hardware

    def select_runtime_backend(self, hardware_profile: object) -> RuntimeBackendSelection:
        del hardware_profile
        self.operations.append("select_backend")
        return self.backend_selection

    def execution_plans_for_recommendation(
        self,
        recommendation: ModelRecommendation,
    ) -> list[ExecutionPlan]:
        self.operations.append("plans")
        if self.execution_plans is not None:
            return self.execution_plans
        return [execution_plan_for_recommendation(recommendation)]

    def prepare_execution_plan(
        self,
        plan: object,
        *,
        hardware: object | None = None,
        on_progress: object | None = None,
    ) -> PreparedExecutionPlan:
        del hardware
        self.operations.append("prepare_plan")
        self.prepared_plans.append(plan)
        if callable(on_progress):
            on_progress("Preparing selected execution path")
        prepared_artifact = self.artifact.model_copy(
            update={"is_downloaded": True, "is_verified": True}
        )
        return PreparedExecutionPlan(plan=plan, artifact=prepared_artifact)  # type: ignore[arg-type]

    def run_experiment(self, request: object) -> ExperimentRunResult:
        from jaull.domain.experiments import ExperimentRequest

        assert isinstance(request, ExperimentRequest)
        self.operations.append("experiment")
        self.experiment_requests.append(request)
        if self.experiment_error is not None:
            raise self.experiment_error
        record = _experiment_record(
            request,
            observation=self.experiment_observation,
        )
        self.experiment_records.append(record)
        return ExperimentRunResult(
            record=record,
            persisted_path=self.experiment_persisted_path,
        )

    def compare_saved_benchmarks_for_recommendation(
        self,
        recommendation: ModelRecommendation,
    ) -> BenchmarkComparison:
        self.operations.append("compare_saved_benchmarks")
        if self.benchmark_comparison is not None:
            return self.benchmark_comparison
        return _benchmark_comparison(recommendation)


def _backend_selection(backend: ComputeBackend) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK
        if backend is ComputeBackend.CPU
        else RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE,
    )


def _runtime_capability(backend: ComputeBackend) -> LlamaCppRuntimeCapability:
    devices: list[LlamaCppRuntimeDevice] = []
    if backend is not ComputeBackend.CPU:
        devices.append(
            LlamaCppRuntimeDevice(
                backend=backend,
                runtime_id=f"{backend.value.title()}0",
                name="test accelerator",
            )
        )
    return LlamaCppRuntimeCapability(
        binary_path="/tmp/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text="llama-cli test",
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=backend,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                devices=devices,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE
                if backend is ComputeBackend.CPU
                else LlamaCppCapabilityReason.BACKEND_EXPOSED,
                source="test",
            )
        ],
        probe_source="test",
    )


def _pytorch_runtime_capability() -> PyTorchRuntimeCapability:
    return PyTorchRuntimeCapability(
        python_executable="/tmp/python",
        runtime_status=PyTorchRuntimeStatus.AVAILABLE,
        torch_version="2.8.0",
        transformers_version="4.55.0",
        probe_source="test",
    )


def _experiment_record(
    request: object,
    *,
    observation: ExecutionObservation,
) -> ExperimentRecord:
    from jaull.domain.experiments import ExperimentRequest

    assert isinstance(request, ExperimentRequest)
    if request.runtime.runtime is RuntimeName.TRANSFORMERS:
        capability = _pytorch_runtime_capability()
        readiness = evaluate_pytorch_execution_readiness(
            selection=request.backend_selection,
            runtime_capability=capability,
        )
    else:
        capability = _runtime_capability(request.backend_selection.selected_backend)
        readiness = evaluate_execution_readiness(
            selection=request.backend_selection,
            runtime_capability=capability,
        )
    return build_experiment_record(
        hardware=request.hardware,
        artifact=request.artifact,
        workload=request.workload,
        backend_trace=None,
        runtime=request.runtime,
        prediction=request.prediction,
        runtime_capability=capability,
        execution_readiness=readiness,
        observation=observation,
    )


def _benchmark_comparison(recommendation: ModelRecommendation) -> BenchmarkComparison:
    plan = execution_plan_for_recommendation(recommendation)
    llama_record = _benchmark_record(
        artifact=_artifact(
            repo_id="org/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-q4_k_m.gguf",
            format="gguf",
            quantization="Q4_K_M",
        ),
        runtime=_runtime(ctx_size=4096, n_gpu_layers=0),
        methodology="llama_bench_v1",
        prefill_128=250.4,
        prefill_512=237.2,
        generation_128=65.6,
        peak_ram_bytes=505 * 1024**2,
    )
    transformers_runtime = RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        flags=[
            RuntimeFlag(
                name="device_map",
                value="cpu",
                source=RuntimeFlagSource.HARDWARE,
                explanation="CPU test runtime",
            )
        ],
        confidence=EstimationConfidence.HIGH,
    )
    transformers_record = _benchmark_record(
        artifact=_artifact(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            filename="model.safetensors",
            format="safetensors",
            quantization=None,
            sha256=None,
        ),
        runtime=transformers_runtime,
        methodology="transformers_isolated_inference_v2",
        prefill_128=44.1,
        prefill_512=45.4,
        generation_128=14.6,
        peak_ram_bytes=int(2.61 * 1024**3),
    )
    return compare_benchmark_records(
        model_identity=plan.model_identity,
        records=[llama_record, transformers_record],
    )


def _benchmark_record(
    *,
    artifact: ModelArtifact,
    runtime: RuntimeRecommendation,
    methodology: str,
    prefill_128: float,
    prefill_512: float,
    generation_128: float,
    peak_ram_bytes: int,
) -> BenchmarkRecord:
    request = BenchmarkRequest(
        artifact=artifact,
        runtime=runtime,
        backend=ComputeBackend.CPU,
        device=None,
        gpu_layers=BenchmarkGpuLayers.count_layers(0),
        repetitions=3,
    )
    observation = BenchmarkObservation(
        success=True,
        repetitions=3,
        duration_seconds=10.0,
        methodology=methodology,
        peak_ram_bytes=peak_ram_bytes,
        measurements=[
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.PREFILL,
                tokens=128,
                mean_tokens_per_second=prefill_128,
                stddev_tokens_per_second=1.0,
                source_label="test",
            ),
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.PREFILL,
                tokens=512,
                mean_tokens_per_second=prefill_512,
                stddev_tokens_per_second=1.0,
                source_label="test",
            ),
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.GENERATION,
                tokens=128,
                mean_tokens_per_second=generation_128,
                stddev_tokens_per_second=1.0,
                source_label="test",
            ),
        ],
    )
    return BenchmarkRecord.create(
        hardware=hardware(),
        request=request,
        observation=observation,
    )


def _not_ready_error(status: ExecutionReadinessStatus) -> ExperimentNotReadyError:
    selection = _backend_selection(ComputeBackend.VULKAN)
    capability = LlamaCppRuntimeCapability(
        binary_path="/tmp/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.VULKAN,
                state=LlamaCppBackendCapabilityState.UNKNOWN,
                reason=LlamaCppCapabilityReason.CAPABILITY_UNKNOWN,
                source="test",
            )
        ],
        probe_source="test",
        message="test readiness unavailable",
    )
    readiness = evaluate_execution_readiness(
        selection=selection,
        runtime_capability=capability,
    )
    if status is ExecutionReadinessStatus.NOT_READY:
        missing_capability = LlamaCppRuntimeCapability(
            binary_path=None,
            binary_status=LlamaCppBinaryStatus.MISSING,
            backend_capabilities=[],
            probe_source="test",
            message="runtime missing",
        )
        readiness = evaluate_execution_readiness(
            selection=selection,
            runtime_capability=missing_capability,
        )
        capability = missing_capability
    assert readiness.status is status
    return ExperimentNotReadyError(
        "not ready",
        readiness=readiness,
        runtime_capability=capability,
    )


def _persistence_error() -> ExperimentPersistenceError:
    recommendation = _recommendation()
    estimate = recommendation.evaluated.memory_estimate
    assert estimate is not None
    runtime = estimate.runtime_recommendation
    assert runtime is not None
    request = ExperimentRequest(
        hardware=hardware(),
        artifact=_artifact(is_downloaded=True, is_verified=True),
        runtime=runtime,
        prediction=estimate,
        backend_selection=_backend_selection(ComputeBackend.CPU),
        workload=ExperimentWorkload(prompt="test"),
    )
    return ExperimentPersistenceError(
        "save failed",
        record=_experiment_record(request, observation=_observation()),
    )


def _visible_text(screen: object) -> str:
    parts: list[str] = []
    for widget in screen.query(Static):  # type: ignore[attr-defined]
        renderable = getattr(widget, "renderable", None)
        if renderable is not None:
            parts.append(str(renderable))
        else:
            parts.append(str(widget.render()))
    return "\n".join(parts)


def _widget_text(widget: Static) -> str:
    renderable = getattr(widget, "renderable", None)
    if renderable is not None:
        return str(renderable)
    return str(widget.render())


def _run_workers_inline(
    app: JaullApp,
    screen: object,
    monkeypatch: Any,
) -> None:
    del app

    class _InlineExecutor:
        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future[None]:
            fn(*args, **kwargs)
            future: Future[None] = Future()
            future.set_result(None)
            return future

        def shutdown(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        screen,
        "_executor",
        _InlineExecutor(),
    )


def _set_prompt(screen: RecommendationExecutionScreen, prompt: str) -> None:
    screen.query_one("#run-prompt-input", TextArea).load_text(prompt)


def test_results_screen_runs_an_alternative_recommendation() -> None:
    async def scenario() -> None:
        first = _recommendation(rank=1, repo_id="org/First-GGUF", quantization="Q4_K_M")
        second = _recommendation(rank=2, repo_id="org/Second-GGUF", quantization="Q5_K_M")
        state = RecommendationWorkflowState(recommendations=[first, second])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)

            screen.query_one("#res-run-1", Button).press()
            await pilot.pause()

            run_screen = pilot.app.screen
            assert isinstance(run_screen, RecommendationExecutionScreen)
            assert run_screen.recommendation.repo_id == "org/Second-GGUF"

    _run(scenario())


def test_results_screen_can_open_validation_for_selected_recommendation() -> None:
    async def scenario() -> None:
        recommendation = _recommendation()
        state = RecommendationWorkflowState(recommendations=[recommendation])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)
            assert screen.query_one("#res-validate-0", Button).disabled is False

            screen.query_one("#res-validate-0", Button).press()
            await pilot.pause()

            validation = pilot.app.screen
            assert isinstance(validation, RecommendationValidationScreen)
            assert validation.recommendation is recommendation

    _run(scenario())


def test_results_validation_and_benchmark_support_transformers_runtime() -> None:
    async def scenario() -> None:
        recommendation = _recommendation(
            runtime=RuntimeRecommendation(
                runtime=RuntimeName.TRANSFORMERS,
                confidence=EstimationConfidence.LOW,
            )
        )
        state = RecommendationWorkflowState(recommendations=[recommendation])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)
            assert screen.query_one("#res-validate-0", Button).disabled is False
            assert screen.query_one("#res-benchmark-0", Button).disabled is False

    _run(scenario())


def test_validation_screen_runs_successful_experiment(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()

            assert advisor.operations == [
                "resolve",
                "download",
                "verify",
                "hardware",
                "select_backend",
                "experiment",
            ]
            assert len(advisor.experiment_requests) == 1
            assert len(advisor.experiment_records) == 1
            text = _visible_text(screen)
            assert "Configuration validated" in text
            # Result before mechanism: what happened, then estimated against
            # measured, then how it ran. Raw backend trace fields moved into
            # the technical details screen.
            assert "Result" in text
            assert "Prediction" in text
            assert "Estimated RAM" in text
            assert "Measured RAM" in text
            assert "Execution" in text
            assert advisor.experiment_persisted_path is not None
            assert screen.query_one("#validation-details", Button).disabled is False
            screen.query_one("#validation-details", Button).press()
            await pilot.pause()
            assert str(advisor.experiment_persisted_path) in _visible_text(
                pilot.app.screen
            )

    _run(scenario())


def test_validation_screen_runs_transformers_experiment_without_gguf_resolution(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        runtime = RuntimeRecommendation(
            runtime=RuntimeName.TRANSFORMERS,
            flags=[
                RuntimeFlag(
                    name="device_map",
                    value="cpu",
                    source=RuntimeFlagSource.HARDWARE,
                    explanation="CPU validation",
                )
            ],
            confidence=EstimationConfidence.HIGH,
        )
        recommendation = _recommendation(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            quantization="float32",
            runtime=runtime,
        )
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(recommendation))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()

            assert advisor.operations == [
                "hardware",
                "select_backend",
                "experiment",
            ]
            request = advisor.experiment_requests[0]
            assert isinstance(request, ExperimentRequest)
            assert request.artifact.repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
            assert request.artifact.format == RuntimeName.TRANSFORMERS.value
            assert request.artifact.local_path is None
            assert request.runtime is runtime
            assert len(advisor.experiment_records) == 1
            assert advisor.experiment_records[0].runtime.runtime is RuntimeName.TRANSFORMERS
            text = _visible_text(screen)
            assert "Configuration validated" in text
            assert "Runtime status" not in text

    _run(scenario())


def test_validation_screen_preserves_failed_execution_record(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(
            experiment_observation=_observation(
                success=False,
                duration_seconds=0.4,
                exit_code=2,
                failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
            )
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()

            assert len(advisor.experiment_records) == 1
            assert advisor.experiment_records[0].observation.success is False
            text = _visible_text(screen)
            assert "Validation failed during execution" in text
            assert "Failed" in text

    _run(scenario())


def test_validation_screen_does_not_execute_when_not_ready(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(
            experiment_error=_not_ready_error(ExecutionReadinessStatus.NOT_READY)
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()

            assert advisor.operations == [
                "resolve",
                "download",
                "verify",
                "hardware",
                "select_backend",
                "experiment",
            ]
            assert advisor.experiment_records == []
            assert "Validation unavailable" in _visible_text(screen)

    _run(scenario())


def test_validation_screen_does_not_execute_when_readiness_unknown(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(
            experiment_error=_not_ready_error(ExecutionReadinessStatus.UNKNOWN)
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()

            assert advisor.experiment_records == []
            assert "Could not verify runtime readiness" in _visible_text(screen)

    _run(scenario())


def test_validation_screen_keeps_result_visible_when_persistence_fails(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(experiment_error=_persistence_error())
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()

            text = _visible_text(screen)
            assert "could not be saved" in text
            assert "Configuration validated (not saved)" in text
            assert "Prediction" in text

    _run(scenario())


def test_validation_screen_can_repeat_without_duplicate_ids(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationValidationScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationValidationScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#validation-start", Button).press()
            await pilot.pause()
            first_id = advisor.experiment_records[-1].identity.experiment_id
            screen.query_one("#validation-start", Button).press()
            await pilot.pause()
            second_id = advisor.experiment_records[-1].identity.experiment_id

            assert first_id != second_id
            assert len(advisor.experiment_records) == 2
            assert "Configuration validated" in _visible_text(screen)

    _run(scenario())


def test_results_export_modal_can_cancel_reopen_export_and_reopen(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state = RecommendationWorkflowState(recommendations=[_recommendation()])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            results = pilot.app.screen
            assert isinstance(results, RecommendationResultsScreen)

            results.query_one("#res-export", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, ExportReportModal)
            pilot.app.screen.query_one("#res-export-cancel", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, RecommendationResultsScreen)

            pilot.app.screen.query_one("#res-export", Button).press()
            await pilot.pause()
            modal = pilot.app.screen
            assert isinstance(modal, ExportReportModal)
            modal.query_one("#res-export-path", Input).value = str(tmp_path / "report.json")
            modal.query_one("#res-export-confirm", Button).press()
            await pilot.pause()

            assert (tmp_path / "report.json").exists()
            assert (tmp_path / "report.md").exists()
            assert "Report written" in _visible_text(modal)

            modal.query_one("#res-export-close", Button).press()
            await pilot.pause()
            pilot.app.screen.query_one("#res-export", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, ExportReportModal)

    _run(scenario())


def test_results_export_modal_shows_filesystem_errors_without_rebuilding_form(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        state = RecommendationWorkflowState(recommendations=[_recommendation()])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        def fail_write(*args: object, **kwargs: object) -> list[Path]:
            del args, kwargs
            raise OSError("permission denied")

        monkeypatch.setattr(
            "jaull.tui.screens.recommendation_results.write_recommendation_report",
            fail_write,
        )

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            pilot.app.screen.query_one("#res-export", Button).press()
            await pilot.pause()
            modal = pilot.app.screen
            assert isinstance(modal, ExportReportModal)
            path_input = modal.query_one("#res-export-path", Input)

            modal.query_one("#res-export-confirm", Button).press()
            await pilot.pause()
            modal.query_one("#res-export-confirm", Button).press()
            await pilot.pause()

            assert modal.query_one("#res-export-path", Input) is path_input
            assert "permission denied" in _visible_text(modal)

    _run(scenario())


def test_results_compare_and_details_can_reopen_without_duplicate_ids() -> None:
    async def scenario() -> None:
        state = RecommendationWorkflowState(
            recommendations=[
                _recommendation(rank=1, repo_id="org/First-GGUF"),
                _recommendation(rank=2, repo_id="org/Second-GGUF"),
            ]
        )
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            # The list names the model and how it would execute; the execution
            # paths themselves live on their own screen behind "Paths".
            assert "Recommendations" in _visible_text(pilot.app.screen)
            assert "llama.cpp" in _visible_text(pilot.app.screen)

            for _ in range(2):
                pilot.app.screen.query_one("#res-compare", Button).press()
                await pilot.pause()
                assert isinstance(pilot.app.screen, RecommendationCompareScreen)
                assert pilot.app.screen.query(DataTable)
                pilot.app.screen.query_one("#compare-back", Button).press()
                await pilot.pause()
                assert isinstance(pilot.app.screen, RecommendationResultsScreen)

                pilot.app.screen.query_one("#res-details", Button).press()
                await pilot.pause()
                assert isinstance(pilot.app.screen, RecommendationDetailsScreen)
                assert "Technical details" in _visible_text(pilot.app.screen)
                assert "preflight not checked" in _visible_text(pilot.app.screen)
                pilot.app.screen.query_one("#details-back", Button).press()
                await pilot.pause()
                assert isinstance(pilot.app.screen, RecommendationResultsScreen)

    _run(scenario())


def test_results_compare_collapses_representations_of_same_model_identity() -> None:
    async def scenario() -> None:
        qwen_identity = ModelIdentity(
            canonical_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            family="Qwen",
            model_name="Qwen2.5 1.5B Instruct",
            parameter_count=1_500_000_000,
        )
        other_identity = ModelIdentity(
            canonical_repo_id="org/Other-Model",
            family="Other",
            model_name="Other Model",
        )
        state = RecommendationWorkflowState(
            recommendations=[
                _recommendation_with_plan(
                    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    identity=qwen_identity,
                    format_=ArtifactVariantFormat.GGUF,
                    quantization="Q5_K_M",
                    runtime=RuntimeName.LLAMA_CPP,
                ),
                _recommendation_with_plan(
                    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
                    identity=qwen_identity,
                    format_=ArtifactVariantFormat.SAFETENSORS,
                    precision="int8",
                    runtime=RuntimeName.TRANSFORMERS,
                ),
                _recommendation_with_plan(
                    repo_id="org/Other-Model",
                    identity=other_identity,
                    format_=ArtifactVariantFormat.GGUF,
                    quantization="Q4_K_M",
                    runtime=RuntimeName.LLAMA_CPP,
                ),
            ]
        )
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            pilot.app.screen.query_one("#res-compare", Button).press()
            await pilot.pause()

            screen = pilot.app.screen
            assert isinstance(screen, RecommendationCompareScreen)
            table = screen.query_one(DataTable)
            assert table.row_count == 2
            model_cells = [str(table.get_row_at(index)[0]) for index in range(table.row_count)]
            assert "Qwen2.5 1.5B Instruct" in model_cells
            assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF" not in model_cells

    _run(scenario())


def test_execution_path_compare_uses_saved_benchmark_records() -> None:
    async def scenario() -> None:
        recommendation = _recommendation(repo_id="Qwen/Qwen2.5-0.5B-Instruct")
        advisor = _FakeAdvisor(benchmark_comparison=_benchmark_comparison(recommendation))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(130, 50)) as pilot:
            app.show_recommendations(
                RecommendationWorkflowState(recommendations=[recommendation])
            )
            await pilot.pause()

            results = pilot.app.screen
            assert isinstance(results, RecommendationResultsScreen)
            results.query_one("#res-compare-paths", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(
                    pilot.app.screen,
                    ExecutionPathBenchmarkCompareScreen,
                )
                and "Measured differences" in _visible_text(pilot.app.screen),
            )

            text = _visible_text(pilot.app.screen)
            # The title is the model's name, not its repository path.
            assert "Qwen2.5 0.5B Instruct" in text
            assert "Comparison notes" in text
            # Ratios come from BenchmarkComparison.relative_throughput and name
            # both sides. The screen must never declare a winner, score the
            # plans, or call one of them better.
            assert "Measured differences" in text
            assert re.search(r"\d\.\dx (llama\.cpp|PyTorch)", text)
            assert "Winner" not in text
            assert "better" not in text.lower()
            assert advisor.operations == ["compare_saved_benchmarks"]

    _run(scenario())


def test_execution_paths_selects_plan_and_run_uses_prepared_plan(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        recommendation = _recommendation()
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(
                RecommendationWorkflowState(recommendations=[recommendation])
            )
            await pilot.pause()

            results = pilot.app.screen
            assert isinstance(results, RecommendationResultsScreen)
            results.query_one("#res-paths-0", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, ExecutionPathsScreen)
                and bool(pilot.app.screen.query("#paths-gguf")),
            )
            paths = pilot.app.screen
            assert isinstance(paths, ExecutionPathsScreen)
            assert advisor.operations == ["plans"]

            paths.query_one("#paths-run", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, RecommendationExecutionScreen),
            )
            run_screen = pilot.app.screen
            assert isinstance(run_screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, run_screen, monkeypatch)

            _set_prompt(run_screen, "Say hi")
            run_screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: "generated from tui" in _visible_text(run_screen),
            )

            assert "prepare_plan" in advisor.operations
            assert advisor.runs[0][0].is_verified is True
            expected_runtime = (
                recommendation.evaluated.memory_estimate.runtime_recommendation
            )
            assert advisor.runs[0][2] is expected_runtime

    _run(scenario())


def test_execution_paths_quantization_selection_does_not_prepare() -> None:
    def plan_for(quantization: str) -> ExecutionPlan:
        recommendation = _recommendation(quantization=quantization)
        plan = execution_plan_for_recommendation(recommendation)
        artifact = plan.artifact.model_copy(
            update={
                "repo_id": "org/Tiny-GGUF",
                "format": ArtifactVariantFormat.GGUF,
                "quantization": quantization,
                "filename": f"tiny-{quantization.lower()}.gguf",
            }
        )
        return plan.model_copy(
            update={
                "plan_id": f"org/Tiny-GGUF:{quantization}",
                "artifact": artifact,
            }
        )

    async def scenario() -> None:
        recommendation = _recommendation()
        advisor = _FakeAdvisor(
            execution_plans=[
                plan_for("Q4_K_M"),
                plan_for("Q5_K_M"),
                plan_for("Q8_0"),
            ]
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(
                RecommendationWorkflowState(recommendations=[recommendation])
            )
            await pilot.pause()

            results = pilot.app.screen
            assert isinstance(results, RecommendationResultsScreen)
            results.query_one("#res-paths-0", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, ExecutionPathsScreen)
                and bool(pilot.app.screen.query("#paths-quant")),
            )
            paths = pilot.app.screen
            assert isinstance(paths, ExecutionPathsScreen)
            assert advisor.operations == ["plans"]
            assert "GGUF · Q4_K_M" in _visible_text(paths)

            # Quantization is a setting on the llama.cpp option, not a peer
            # path: one control, opening on the suggested quantizations with
            # the rest behind the advanced toggle.
            picker = paths.query_one("#paths-quant", Select)
            offered = [value for _, value in picker._options]
            assert offered == ["org/Tiny-GGUF:Q4_K_M", "org/Tiny-GGUF:Q5_K_M"]

            picker.value = "org/Tiny-GGUF:Q5_K_M"
            await _wait_until(
                pilot,
                lambda: "GGUF · Q5_K_M" in _visible_text(paths),
            )

            assert advisor.operations == ["plans"]
            assert "prepare_plan" not in advisor.operations

    _run(scenario())


def test_execution_screen_prepares_artifact_automatically_and_runs_selected_runtime(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        recommendation = _recommendation()
        runtime = recommendation.evaluated.memory_estimate.runtime_recommendation
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(recommendation))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            _set_prompt(screen, "Say hi")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: len(screen.query(InferenceResponse)) == 1,
            )

            assert advisor.operations == ["resolve", "download", "verify", "run"]
            assert advisor.resolved == [("org/Tiny-GGUF", "Q4_K_M", None)]
            assert advisor.runs[0][1] == "Say hi"
            assert advisor.runs[0][2] is runtime
            assert screen.query(InferencePrompt)
            assert screen.query(InferenceResponse)
            text = _visible_text(screen)
            assert "generated from tui" in text
            assert "0.10 s" in text
            assert "RAM 512.0 MiB" in text
            assert "VRAM 256.0 MiB" in text
            assert "Resolve artifact for org/Tiny-GGUF (Q4_K_M)" in screen.log_messages
            assert "Verifying artifact" in screen.log_messages
            assert "Generating" in screen.log_messages

    _run(scenario())


def test_execution_screen_runs_transformers_runtime_without_gguf_artifact_resolution(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        runtime = RuntimeRecommendation(
            runtime=RuntimeName.TRANSFORMERS,
            flags=[
                RuntimeFlag(
                    name="device_map",
                    value="cpu",
                    source=RuntimeFlagSource.HARDWARE,
                    explanation="CPU test",
                ),
                RuntimeFlag(
                    name="torch_dtype",
                    value="torch.float32",
                    source=RuntimeFlagSource.ESTIMATE,
                    explanation="dtype test",
                ),
            ],
            confidence=EstimationConfidence.HIGH,
        )
        recommendation = _recommendation(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            quantization="float32",
            runtime=runtime,
        )
        advisor = _FakeAdvisor(response_text="generated with transformers")
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(recommendation))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            _set_prompt(screen, "Say hi")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: len(screen.query(InferenceResponse)) == 1,
            )

            assert advisor.operations == ["run"]
            artifact, prompt, used_runtime = advisor.runs[0]
            assert artifact.repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
            assert artifact.format == RuntimeName.TRANSFORMERS.value
            assert artifact.local_path is None
            assert artifact.is_downloaded is False
            assert artifact.is_verified is False
            assert prompt == "Say hi"
            assert used_runtime is runtime
            text = _visible_text(screen)
            assert "generated with transformers" in text
            assert "transformers" in text
            assert "Preparing Transformers runtime" in screen.log_messages
            assert "Resolve artifact for Qwen/Qwen2.5-0.5B-Instruct" not in screen.log_messages

    _run(scenario())


def test_execution_screen_keeps_visual_history_across_prompts(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(
            response_text="first response",
            observation=_observation(
                duration_seconds=0.11,
                peak_ram_bytes=512 * 1024**2,
                peak_vram_bytes=256 * 1024**2,
            ),
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            _set_prompt(screen, "Say hi")
            screen.query_one("#run-generate", Button).press()
            await _settle(pilot)

            prompt_input = screen.query_one("#run-prompt-input", TextArea)
            assert prompt_input.text == ""
            assert prompt_input.has_focus

            advisor.response_text = "second response"
            advisor.observation = _observation(
                duration_seconds=0.22,
                peak_ram_bytes=768 * 1024**2,
                peak_vram_bytes=None,
            )
            _set_prompt(screen, "Say bye")
            screen.query_one("#run-generate", Button).press()
            await _settle(pilot)

            assert advisor.runs[-1][1] == "Say bye"
            assert len(screen.query(InferencePrompt)) == 2
            assert len(screen.query(InferenceResponse)) == 2
            text = _visible_text(screen)
            assert "Say hi" in text
            assert "first response" in text
            assert "0.11 s" in text
            assert "RAM 512.0 MiB" in text
            assert "VRAM 256.0 MiB" in text
            assert "Say bye" in text
            assert "second response" in text
            assert "0.22 s" in text
            assert "RAM 768.0 MiB" in text
            # The second run measured no VRAM, so the history line omits it
            # rather than printing "VRAM unavailable" under the answer.
            assert "unavailable" not in text
            second = screen.query(InferenceResponse).last()
            assert "VRAM" not in _widget_text(second.query_one(".message-meta"))

    _run(scenario())


def test_execution_screen_survives_three_prompts_without_duplicate_ids(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            for index in range(3):
                advisor.response_text = f"response {index}"
                _set_prompt(screen, f"prompt {index}")

                screen.query_one("#run-generate", Button).press()

                await _wait_until(
                    pilot,
                    lambda expected=index + 1: len(screen.query(InferenceResponse)) == expected,
                )

            assert len(screen.query(InferencePrompt)) == 3
            assert len(screen.query(InferenceResponse)) == 3
            assert screen.query_one("#run-prompt-input", TextArea).text == ""
            assert advisor.operations == [
                "resolve",
                "download",
                "verify",
                "run",
                "run",
                "run",
            ]

    _run(scenario())

async def _wait_until(
    pilot: Any,
    predicate: Any,
    *,
    timeout_seconds: float = 2.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(
                "Timed out waiting for TUI state"
            )

        await pilot.pause()
        await asyncio.sleep(0.01)

def test_execution_screen_reuses_local_artifact_without_downloading(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(artifact=_artifact(is_downloaded=True))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)
            _set_prompt(screen, "Hello")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: len(screen.query(InferenceResponse)) == 1,
            )

            assert advisor.operations == ["resolve", "verify", "run"]
            assert advisor.downloaded == []
            status = screen.query_one("#run-status", Static)
            assert "Model ready" in _widget_text(status)

    _run(scenario())


def test_execution_screen_shows_download_failure(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(fail_download=ArtifactDownloadError("download failed"))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)
            _set_prompt(screen, "Hello")
            screen.query_one("#run-generate", Button).press()
            message = screen.query_one("#run-error-message", Static)
            await _wait_until(
                pilot,
                lambda: "download failed" in _widget_text(message),
            )
            assert "download failed" in _widget_text(message)

    _run(scenario())


def test_execution_screen_shows_llama_cli_missing(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(fail_run=ExecutableNotFoundError("llama-cli missing"))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)
            _set_prompt(screen, "Hello")
            screen.query_one("#run-generate", Button).press()
            message = screen.query_one("#run-error-message", Static)
            await _wait_until(
                pilot,
                lambda: "llama-cli missing" in _widget_text(message),
            )
            assert "llama-cli missing" in _widget_text(message)

    _run(scenario())


def test_execution_screen_retry_after_run_error_reuses_prepared_artifact(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(fail_run=ExecutionTimeoutError("first run failed"))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            _set_prompt(screen, "First try")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: "first run failed" in _visible_text(screen),
            )
            assert "first run failed" in _visible_text(screen)

            advisor.fail_run = None
            advisor.response_text = "retry worked"
            _set_prompt(screen, "Retry")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: len(screen.query(InferenceResponse)) == 1,
            )

            assert advisor.operations == ["resolve", "download", "verify", "run", "run"]
            assert len(screen.query(InferenceResponse)) == 1
            assert "retry worked" in _visible_text(screen)

    _run(scenario())


def test_execution_screen_shows_timeout_and_failed_process_errors(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        failures = [
            ExecutionTimeoutError("timed out"),
            ExecutionFailedError(
                "non-zero exit",
                ExecutionResult(
                    stdout="",
                    stderr="bad",
                    observation=_observation(
                        success=False,
                        exit_code=2,
                        failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
                    ),
                ),
            ),
        ]
        for failure in failures:
            advisor = _FakeAdvisor(fail_run=failure)
            app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
            async with app.run_test(size=(120, 50)) as pilot:
                app.push_screen(RecommendationExecutionScreen(_recommendation()))
                await pilot.pause()
                screen = pilot.app.screen
                assert isinstance(screen, RecommendationExecutionScreen)
                _run_workers_inline(pilot.app, screen, monkeypatch)
                _set_prompt(screen, "Hello")
                screen.query_one("#run-generate", Button).press()
                message = screen.query_one("#run-error-message", Static)
                await _wait_until(
                    pilot,
                    lambda expected=str(failure), widget=message: expected
                    in _widget_text(widget),
                )
                assert str(failure) in _widget_text(message)

    _run(scenario())


def test_execution_screen_rejects_empty_prompt(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)
            screen.query_one("#run-generate", Button).press()
            await pilot.pause()

            message = screen.query_one("#run-error-message", Static)
            assert "Prompt must not be empty" in _widget_text(message)
            assert advisor.operations == []

    _run(scenario())


def test_generate_runs_long_operations_without_blocking_ui() -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(prepare_delay=0.6)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _set_prompt(screen, "Hello")
            started = time.perf_counter()
            screen.query_one("#run-generate", Button).press()
            await pilot.pause()

            assert time.perf_counter() - started < advisor.prepare_delay
            assert screen.query_one("#run-generate", Button).disabled is True
            assert screen.query_one("#run-prompt-input", TextArea).disabled is True

    _run(scenario())


def test_escape_returns_to_results_screen() -> None:
    """The composer has no Back button; Escape is the way out."""

    async def scenario() -> None:
        state = RecommendationWorkflowState(recommendations=[_recommendation()])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            pilot.app.screen.query_one("#res-run-0", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, RecommendationExecutionScreen)
            assert not pilot.app.screen.query("#run-back")

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(pilot.app.screen, RecommendationResultsScreen)

    _run(scenario())


def test_tui_results_execution_export_details_stress_flow(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        state = RecommendationWorkflowState(
            recommendations=[
                _recommendation(rank=1, repo_id="org/First-GGUF", quantization="Q4_K_M"),
                _recommendation(rank=2, repo_id="org/Second-GGUF", quantization="Q5_K_M"),
            ]
        )
        advisor = _FakeAdvisor(response_text="initial")
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()

            for _ in range(2):
                pilot.app.screen.query_one("#res-export", Button).press()
                await pilot.pause()
                assert isinstance(pilot.app.screen, ExportReportModal)
                pilot.app.screen.query_one("#res-export-cancel", Button).press()
                await pilot.pause()

            pilot.app.screen.query_one("#res-run-0", Button).press()
            await pilot.pause()
            run_screen = pilot.app.screen
            assert isinstance(run_screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, run_screen, monkeypatch)

            for index in range(3):
                advisor.response_text = f"response {index}"
                _set_prompt(run_screen, f"prompt {index}")
                run_screen.query_one("#run-generate", Button).press()
                await _wait_until(
                    pilot,
                    lambda expected=index + 1: len(run_screen.query(InferenceResponse))
                    == expected,
                )

            advisor.fail_run = ExecutionTimeoutError("transient failure")
            _set_prompt(run_screen, "prompt that fails")
            run_screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: "transient failure" in _visible_text(run_screen),
            )
            assert "transient failure" in _visible_text(run_screen)
            assert len(run_screen.query(InferenceResponse)) == 3

            advisor.fail_run = None
            advisor.response_text = "recovered"
            _set_prompt(run_screen, "retry prompt")
            run_screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot,
                lambda: len(run_screen.query(InferenceResponse)) == 4,
            )
            assert len(run_screen.query(InferenceResponse)) == 4
            assert "response 0" in _visible_text(run_screen)
            assert "recovered" in _visible_text(run_screen)

            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(pilot.app.screen, RecommendationResultsScreen)

            pilot.app.screen.query_one("#res-export", Button).press()
            await pilot.pause()
            modal = pilot.app.screen
            assert isinstance(modal, ExportReportModal)
            modal.query_one("#res-export-path", Input).value = str(tmp_path / "stress.json")
            modal.query_one("#res-export-confirm", Button).press()
            await pilot.pause()
            assert (tmp_path / "stress.json").exists()
            modal.query_one("#res-export-close", Button).press()
            await pilot.pause()

            pilot.app.screen.query_one("#res-details", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, RecommendationDetailsScreen)
            pilot.app.screen.query_one("#details-back", Button).press()
            await pilot.pause()

            pilot.app.screen.query_one("#res-run-1", Button).press()
            await pilot.pause()
            second_run = pilot.app.screen
            assert isinstance(second_run, RecommendationExecutionScreen)
            assert second_run.recommendation.repo_id == "org/Second-GGUF"

    _run(scenario())


def test_selecting_a_recommendation_expands_it_and_collapses_the_others() -> None:
    """Selection is a state on the row, not a separate card for the winner."""

    async def scenario() -> None:
        state = RecommendationWorkflowState(
            recommendations=[
                _recommendation(rank=1, repo_id="org/First-GGUF"),
                _recommendation(rank=2, repo_id="org/Second-GGUF"),
            ]
        )
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)

            first = screen.query_one("#rec-row-0")
            second = screen.query_one("#rec-row-1")

            def actions_shown(row: Any) -> bool:
                return row.query_one(".rec-actions").display

            assert first.has_class("-selected")
            assert not second.has_class("-selected")
            assert actions_shown(first) is True
            assert actions_shown(second) is False

            await pilot.press("down")
            await pilot.pause()

            assert not first.has_class("-selected")
            assert second.has_class("-selected")
            assert actions_shown(first) is False
            assert actions_shown(second) is True

            # The buttons themselves are never unmounted, so their ids stay
            # stable across selections — this is what keeps repeated navigation
            # free of duplicate-id collisions.
            assert screen.query_one("#res-run-0", Button)
            assert screen.query_one("#res-run-1", Button)

    _run(scenario())


def test_compare_paths_uses_the_selected_recommendation() -> None:
    """It used to always compare recommendations[0], whatever was highlighted."""

    async def scenario() -> None:
        state = RecommendationWorkflowState(
            recommendations=[
                _recommendation(rank=1, repo_id="org/First-GGUF"),
                _recommendation(rank=2, repo_id="org/Second-GGUF"),
            ]
        )
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationResultsScreen)

            await pilot.press("down")
            await pilot.pause()
            screen.query_one("#res-compare-paths", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(
                    pilot.app.screen,
                    ExecutionPathBenchmarkCompareScreen,
                ),
            )

            compare = pilot.app.screen
            assert isinstance(compare, ExecutionPathBenchmarkCompareScreen)
            assert compare.recommendation.repo_id == "org/Second-GGUF"

    _run(scenario())
