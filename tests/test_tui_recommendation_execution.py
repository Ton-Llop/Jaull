from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from textual.widgets import Button, DataTable, Input, Static, TextArea

from jaull.artifacts.errors import ArtifactDownloadError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionObservation,
    ExecutionResult,
    InferenceResult,
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
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.experiments import build_experiment_record
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.experiments.errors import (
    ExperimentNotReadyError,
    ExperimentPersistenceError,
)
from jaull.recommendation.models import ModelRecommendation, ScoreBreakdown
from jaull.recommendation.policies import LicenseCategory
from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_execution import (
    InferencePrompt,
    InferenceResponse,
    RecommendationExecutionScreen,
)
from jaull.tui.screens.recommendation_results import (
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
        self.operations: list[str] = []
        self.resolved: list[tuple[str, str | None, str | None]] = []
        self.downloaded: list[ModelArtifact] = []
        self.verified: list[ModelArtifact] = []
        self.runs: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []
        self.experiment_requests: list[object] = []
        self.experiment_records: list[ExperimentRecord] = []

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
            runtime="llama.cpp",
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


def _experiment_record(
    request: object,
    *,
    observation: ExecutionObservation,
) -> ExperimentRecord:
    from jaull.domain.experiments import ExperimentRequest

    assert isinstance(request, ExperimentRequest)
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


def test_results_validation_button_is_disabled_without_llama_cpp_runtime() -> None:
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
            assert screen.query_one("#res-validate-0", Button).disabled is True

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
            assert "Prediction vs Observation" in text
            assert "Not verified" in text
            assert "/tmp/jaull-experiment.json" in text
            assert screen.query_one("#validation-details", Button).disabled is False

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
            assert "failure" in text

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
            assert "Prediction vs Observation" in text

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
                pilot.app.screen.query_one("#details-back", Button).press()
                await pilot.pause()
                assert isinstance(pilot.app.screen, RecommendationResultsScreen)

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
            await pilot.pause()

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
            assert "VRAM unavailable" in text

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
                await pilot.pause()

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
            await pilot.pause()

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
            await pilot.pause()

            message = screen.query_one("#run-error-message", Static)
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
            await pilot.pause()

            message = screen.query_one("#run-error-message", Static)
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
            await pilot.pause()
            assert "first run failed" in _visible_text(screen)

            advisor.fail_run = None
            advisor.response_text = "retry worked"
            _set_prompt(screen, "Retry")
            screen.query_one("#run-generate", Button).press()
            await pilot.pause()

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
                await pilot.pause()
                message = screen.query_one("#run-error-message", Static)
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
                await pilot.pause()

            advisor.fail_run = ExecutionTimeoutError("transient failure")
            _set_prompt(run_screen, "prompt that fails")
            run_screen.query_one("#run-generate", Button).press()
            await pilot.pause()
            assert "transient failure" in _visible_text(run_screen)
            assert len(run_screen.query(InferenceResponse)) == 3

            advisor.fail_run = None
            advisor.response_text = "recovered"
            _set_prompt(run_screen, "retry prompt")
            run_screen.query_one("#run-generate", Button).press()
            await pilot.pause()
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
