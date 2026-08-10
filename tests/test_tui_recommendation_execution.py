from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from textual.widgets import Button, Input, Select, Static

from jaull.artifacts.errors import ArtifactDownloadError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution import ExecutionResult, InferenceResult
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.recommendation.models import ModelRecommendation, ScoreBreakdown
from jaull.recommendation.policies import LicenseCategory
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_execution import RecommendationExecutionScreen
from jaull.tui.screens.recommendation_results import RecommendationResultsScreen
from jaull.workflow.container import ServiceContainer
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import (
    GIB,
    candidate,
    gguf_analysis,
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
    ) -> None:
        self.services = _services()
        self.artifact = artifact or _artifact()
        self.fail_download = fail_download
        self.fail_run = fail_run
        self.prepare_delay = prepare_delay
        self.response_text = response_text
        self.operations: list[str] = []
        self.resolved: list[tuple[str, str | None, str | None]] = []
        self.downloaded: list[ModelArtifact] = []
        self.verified: list[ModelArtifact] = []
        self.runs: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []

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
            duration_seconds=0.1,
            exit_code=0,
            runtime="llama.cpp",
            model_path=artifact.local_path or Path("/tmp/tiny-q4_k_m.gguf"),
        )


def _visible_text(screen: object) -> str:
    parts: list[str] = []
    for widget in screen.query(Static):  # type: ignore[attr-defined]
        renderable = getattr(widget, "renderable", None)
        if renderable is not None:
            parts.append(str(renderable))
    return "\n".join(parts)


def _widget_text(widget: Static) -> str:
    renderable = getattr(widget, "renderable", None)
    if renderable is not None:
        return str(renderable)
    return str(widget.render())


def _run_workers_inline(
    app: JaullApp,
    screen: RecommendationExecutionScreen,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        app,
        "call_from_thread",
        lambda callback, *args, **kwargs: callback(*args, **kwargs),
    )
    monkeypatch.setattr(screen, "run_worker", lambda work, *args, **kwargs: work())


def test_results_screen_selects_recommendation_for_execution() -> None:
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

            screen.query_one("#res-run-select", Select).value = "1"
            screen.query_one("#res-run", Button).press()
            await pilot.pause()

            run_screen = pilot.app.screen
            assert isinstance(run_screen, RecommendationExecutionScreen)
            assert run_screen.recommendation.repo_id == "org/Second-GGUF"

    _run(scenario())


def test_execution_screen_prepares_artifact_and_runs_selected_runtime(
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

            screen.query_one("#run-prepare", Button).press()
            await pilot.pause()
            screen.query_one("#run-prompt-input", Input).value = "Say hi"
            screen.query_one("#run-execute", Button).press()
            await pilot.pause()

            assert advisor.operations == ["resolve", "download", "verify", "run"]
            assert advisor.resolved == [("org/Tiny-GGUF", "Q4_K_M", None)]
            assert advisor.runs[0][1] == "Say hi"
            assert advisor.runs[0][2] is runtime
            response = screen.query_one("#run-response", Static)
            assert _widget_text(response) == "generated from tui"
            assert "Resolve artifact for org/Tiny-GGUF (Q4_K_M)" in screen.log_messages
            assert "Verify size and SHA-256 sidecar" in screen.log_messages
            assert "Start llama.cpp single-turn generation" in screen.log_messages

    _run(scenario())


def test_execution_screen_sanitizes_control_sequences_in_response(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(response_text="\x1b[2J\x1b[HGenerated [literal]\x07")
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            screen.query_one("#run-prepare", Button).press()
            await pilot.pause()
            screen.query_one("#run-prompt-input", Input).value = "Say hi"
            screen.query_one("#run-execute", Button).press()
            await pilot.pause()

            response = screen.query_one("#run-response", Static)
            assert _widget_text(response) == "Generated [literal]"

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
            screen.query_one("#run-prepare", Button).press()
            await pilot.pause()

            assert advisor.operations == ["resolve", "verify"]
            assert advisor.downloaded == []
            status = screen.query_one("#run-status", Static)
            assert "Artifact verified" in _widget_text(status)

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
            screen.query_one("#run-prepare", Button).press()
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
            screen.query_one("#run-prepare", Button).press()
            await pilot.pause()
            screen.query_one("#run-prompt-input", Input).value = "Hello"
            screen.query_one("#run-execute", Button).press()
            await pilot.pause()

            message = screen.query_one("#run-error-message", Static)
            assert "llama-cli missing" in _widget_text(message)

    _run(scenario())


def test_execution_screen_shows_timeout_and_failed_process_errors(
    monkeypatch: Any,
) -> None:
    async def scenario() -> None:
        failures = [
            ExecutionTimeoutError("timed out"),
            ExecutionFailedError(
                "non-zero exit",
                ExecutionResult(exit_code=2, stdout="", stderr="bad", duration_seconds=0.1),
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
                screen.query_one("#run-prepare", Button).press()
                await pilot.pause()
                screen.query_one("#run-prompt-input", Input).value = "Hello"
                screen.query_one("#run-execute", Button).press()
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
            screen.query_one("#run-prepare", Button).press()
            await pilot.pause()
            screen.query_one("#run-execute", Button).press()
            await pilot.pause()

            message = screen.query_one("#run-error-message", Static)
            assert "Prompt must not be empty" in _widget_text(message)
            assert advisor.operations == ["resolve", "download", "verify"]

    _run(scenario())


def test_prepare_uses_thread_worker_for_long_operations(monkeypatch: Any) -> None:
    async def scenario() -> None:
        advisor = _FakeAdvisor(prepare_delay=0.6)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationExecutionScreen(_recommendation()))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            calls: list[bool] = []
            monkeypatch.setattr(
                screen,
                "run_worker",
                lambda work, *args, **kwargs: calls.append(bool(kwargs.get("thread"))),
            )
            screen.query_one("#run-prepare", Button).press()
            await pilot.pause()

            assert calls == [True]

    _run(scenario())


def test_back_returns_to_results_screen() -> None:
    async def scenario() -> None:
        state = RecommendationWorkflowState(recommendations=[_recommendation()])
        app = JaullApp(advisor=_FakeAdvisor())  # type: ignore[arg-type]

        async with app.run_test(size=(120, 50)) as pilot:
            app.show_recommendations(state)
            await pilot.pause()
            pilot.app.screen.query_one("#res-run", Button).press()
            await pilot.pause()
            assert isinstance(pilot.app.screen, RecommendationExecutionScreen)

            pilot.app.screen.query_one("#run-back", Button).press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, RecommendationResultsScreen)

    _run(scenario())
