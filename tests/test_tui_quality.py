"""Synthetic UI pilot tests: no Docker, network, model or GPU execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Input,
    Select,
    Static,
    TabbedContent,
)

from jaull.advisor.quality import (
    QualityReadiness,
    check_selected_artifact,
    quality_evaluation_block_reason,
)
from jaull.advisor.quality_candidates import (
    QualityCandidate,
    QualityCandidateSelection,
    run_candidates,
)
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import ArtifactVariantFormat, ExecutionPlan
from jaull.domain.runtime import RuntimeName
from jaull.recommendation.capability_catalog import CatalogReadResult
from jaull.runtime.quality_eval_runner import (
    PROFILE_GRADE,
    QualityEvaluationCancelled,
    QualityEvaluationError,
    QualityProfile,
    QualityRunRequest,
)
from jaull.tui.app import JaullApp
from jaull.tui.evidence import EvidenceIndex
from jaull.tui.screens.execution_paths import ExecutionPathsScreen, _ExecutionPathsLoaded
from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen
from jaull.tui.screens.recommendation_results import (
    RecommendationResultsScreen,
    _alternative_quality,
)
from jaull.tui.widgets.selection_workspace import SelectionWorkspace
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import hardware
from tests.test_cli_quality import _record, _request
from tests.test_execution_plans import _gguf_recommendation
from tests.test_published_evaluation_presentation import _catalog, _identity
from tests.test_tui_evidence import _FakeStoreAdvisor
from tests.test_tui_evidence import _plan as _evidence_plan
from tests.test_tui_recommendation_execution import _wait_until


def _plan(**kwargs: Any) -> ExecutionPlan:
    # Evidence-store fixtures use CPU; this UI pilot requires one CUDA GPU.
    return _evidence_plan(**kwargs).model_copy(update={"hardware": hardware()})


class FakeAdvisor:
    services = None

    def __init__(self) -> None:
        self.calls: list[QualityRunRequest] = []
        self.wait = False
        self.failed = False
        self.cleaned = Event()
        self.setup: dict[str, str] = {}
        self.plan_calls: list[tuple[Any, bool]] = []
        self.remembered: list[QualityRunRequest] = []
        self.preparations: list[QualityRunRequest] = []
        self.readiness_profiles: list[Any] = []

    def quality_evaluation_setup(self) -> dict[str, str]:
        return self.setup

    def quality_readiness(self, profile: Any, values: Any, **kwargs: Any) -> QualityReadiness:
        # A ready synthetic environment; no Docker or file inspection in UI tests.
        self.readiness_profiles.append(profile)
        return QualityReadiness((("ok", "Evaluator image", "synthetic"),))

    def remember_quality_evaluation_setup(self, request: QualityRunRequest) -> None:
        self.remembered.append(request)

    def prepare_quality_evaluation_setup(self, request: QualityRunRequest, **kwargs: Any) -> None:
        self.preparations.append(request)
        if self.failed:
            raise QualityEvaluationError("Synthetic setup failure")

    def run_quality_evaluation_for_plan(
        self, plan: Any, request: QualityRunRequest, *, allow_download: bool,
        is_cancelled: Callable[[], bool], on_progress: Callable[[str], None],
    ) -> Path:
        self.plan_calls.append((plan, allow_download))
        on_progress("Fully verifying selected GGUF")
        return self.run_quality_evaluation(request, is_cancelled=is_cancelled)

    def run_quality_evaluation(
        self, request: QualityRunRequest, *, is_cancelled: Callable[[], bool],
    ) -> Path:
        self.calls.append(request)
        if self.wait:
            # A bounded fake, so a broken UI test cannot leave a thread alive.
            import time

            for _ in range(500):
                if is_cancelled():
                    self.cleaned.set()
                    raise QualityEvaluationCancelled("Synthetic cancellation; cleanup finished.")
                time.sleep(0.01)
            raise TimeoutError("UI did not cancel the worker")
        if self.failed:
            raise ValueError("Synthetic missing Docker image")
        return Path("d" * 64 + ".json")

    def load_quality_record(self, record_id: str) -> dict[str, Any]:
        record = _record()
        # The grade each profile produces, as the real bridge enforces it.
        record["classification"] = PROFILE_GRADE[self.calls[-1].profile]
        return record


def _fill(screen: QualityEvaluationScreen, request: QualityRunRequest) -> None:
    for key, value in (
        ("artifact", request.artifact_json), ("dataset", request.dataset_file),
        ("server", request.llama_server), ("root", request.pilot_root),
        ("image", request.image), ("output", request.output),
    ):
        if key == "artifact" and not screen.query("#quality-artifact"):
            continue
        screen.query_one(f"#quality-{key}", Input).value = str(value)


async def _settled(pilot: Any, screen: QualityEvaluationScreen) -> None:
    for _ in range(100):
        await pilot.pause(0.02)
        if not screen._busy:
            return
    pytest.fail("Quality screen did not finish")


class CandidateAdvisor(FakeAdvisor):
    def __init__(self) -> None:
        super().__init__()
        self.selection_calls = 0
        self.candidates = tuple(
            QualityCandidate(_plan(repo_id=f"synthetic/Model-{index}"), "Synthetic fit", False)
            for index in range(3)
        )

    def prepare_quality_candidates(self, state: Any, **kwargs: Any) -> QualityCandidateSelection:
        self.selection_calls += 1
        return QualityCandidateSelection(self.candidates, ("Other models: outside budget",), 8)

    def run_quality_candidates(self, candidates: Any, request: Any, **kwargs: Any) -> Any:
        return run_candidates(self, candidates, request, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
@pytest.mark.parametrize("preference_failure", [False, True])
def test_search_candidate_queue_is_reviewable_opt_in_and_sequential(
    size: tuple[int, int], tmp_path: Path, preference_failure: bool,
) -> None:
    async def scenario() -> None:
        advisor = CandidateAdvisor()
        if preference_failure:
            def cannot_save(request: Any) -> None:
                raise OSError("Synthetic preferences unavailable")
            advisor.remember_quality_evaluation_setup = cannot_save  # type: ignore[method-assign]
        rec = _gguf_recommendation().model_copy(update={"plan": advisor.candidates[0].plan})
        state = RecommendationWorkflowState(recommendations=[rec])
        before = state.model_dump_json()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            results = RecommendationResultsScreen(state)
            await app.push_screen(results)
            await pilot.pause()
            assert not advisor.calls and not advisor.selection_calls
            action = results.query_one("#res-candidate-evaluation", Button)
            assert action.region.right <= size[0]
            assert action.region.bottom < size[1]
            await pilot.click("#res-candidate-evaluation")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QualityEvaluationScreen)
            await _settled(pilot, screen)
            assert advisor.selection_calls == 1 and advisor.calls == []
            assert len(screen.candidate_plans) == 3
            assert not screen.query("#quality-artifact")
            assert not screen.query_one("#quality-download", Checkbox).value
            assert not screen.query_one("#quality-dataset-download", Checkbox).value
            _fill(screen, _request(tmp_path))
            # Review changes the queue, not the recommendations.
            screen.query_one("#quality-candidate-1", Checkbox).value = False
            await pilot.pause(0.6)
            await pilot.click("#quality-start")
            await _settled(pilot, screen)
            assert len(advisor.plan_calls) == len(advisor.calls) == 2
            assert [p.artifact.repo_id for p, _ in advisor.plan_calls] == [
                "synthetic/Model-0", "synthetic/Model-2",
            ]
            assert all(not consent for _, consent in advisor.plan_calls)
            assert [request.output.name for request in advisor.calls] == ["model-01", "model-02"]
            assert "order unchanged" in str(screen.query_one("#quality-result", Static).content)
            text = str(screen.query_one("#quality-result", Static).content)
            assert "Completed 2/2; failed 0; not started 0" in text
            assert screen._stored is not None and "Saved:" in text
            if preference_failure:
                assert "setup was not saved" in text
            for name in ("start", "prepare", "cancel", "back"):
                button = screen.query_one(f"#quality-{name}", Button)
                assert button.region.right <= size[0]
                assert button.region.bottom < size[1]
            app.save_screenshot(filename=f"candidate-queue-{size[0]}.svg", path=str(tmp_path))
            await pilot.click("#quality-back")
            await pilot.pause()
            assert app.screen is results
            assert state.model_dump_json() == before
            assert results._known_paths[0][0] == advisor.candidates[0].plan
    asyncio.run(scenario())


def test_empty_candidate_selection_has_explanation_and_no_run(tmp_path: Path) -> None:
    async def scenario() -> None:
        advisor = CandidateAdvisor()
        advisor.candidates = ()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=RecommendationWorkflowState())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            assert screen.query_one("#quality-start", Button).disabled
            assert "No evaluation candidates" in str(
                screen.query_one("#quality-start", Button).tooltip,
            )
            assert "No models available" in str(screen.query_one("#quality-status", Static).content)
            assert not screen.query_one("#quality-candidates Collapsible", Collapsible).collapsed
            assert not advisor.calls
            text = "\n".join(str(widget.content)
                             for widget in screen.query("#quality-candidates Static"))
            assert "0 candidates" in text and "outside budget" in text
            await pilot.click("#quality-back")
    asyncio.run(scenario())


def test_candidate_queue_cancel_waits_for_cleanup_and_never_starts_next_model(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        advisor = CandidateAdvisor()
        advisor.wait = True
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=RecommendationWorkflowState())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            _fill(screen, _request(tmp_path))
            await pilot.click("#quality-start")
            for _ in range(100):
                await pilot.pause(0.01)
                if advisor.calls:
                    break
            assert screen._busy and len(advisor.calls) == 1
            await pilot.press("escape")
            for _ in range(100):
                await pilot.pause(0.01)
                if app.screen is not screen:
                    break
            assert app.screen is not screen
            assert advisor.cleaned.is_set() and len(advisor.calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
@pytest.mark.parametrize("profile", list(QualityProfile))
def test_quality_screen_is_opt_in_and_keeps_actions_visible(
    size: tuple[int, int], profile: QualityProfile, tmp_path: Path,
) -> None:
    asyncio.run(_screen_opt_in(size, profile, tmp_path))


async def _screen_opt_in(
    size: tuple[int, int], profile: QualityProfile, tmp_path: Path,
) -> None:
    advisor = FakeAdvisor()
    app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
    async with app.run_test(size=size) as pilot:
        screen = QualityEvaluationScreen()
        await app.push_screen(screen)
        await pilot.pause()
        assert advisor.calls == []
        assert screen.query_one("#quality-profile", Select).value == "smoke"
        start = screen.query_one("#quality-start", Button)
        assert start.region.bottom <= size[1] - 1
        assert screen.query_one("#quality-cancel", Button).disabled
        assert screen.query_one("#quality-cancel", Button).tooltip
        await pilot.click("#quality-start")
        await _wait_until(
            pilot, lambda: app.focused is screen.query_one("#quality-artifact", Input),
        )
        assert advisor.calls == []  # Empty inputs must never launch Docker.
        text = str(screen.query_one("#quality-status", Static).render())
        assert "Missing:" in text
        for label in ("artifact manifest", "Evaluation dataset", "llama-server executable"):
            assert label in text
        assert app.focused is screen.query_one("#quality-artifact", Input)
        status = screen.query_one("#quality-status", Static)
        assert status.region.y >= 0
        assert status.region.bottom <= screen.query_one("#quality-actions").region.y
        screen.query_one("#quality-prepare", Button).press()
        await _wait_until(
            pilot, lambda: app.focused is screen.query_one("#quality-dataset", Input),
        )
        assert screen.query_one("#quality-tabs", TabbedContent).active == "quality-setup-tab"
        assert not advisor.calls
        _fill(screen, _request(tmp_path))
        screen.query_one("#quality-profile", Select).value = profile.value
        await pilot.pause(0.6)  # Textual debounces two successive button clicks.
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert len(advisor.calls) == 1
        assert advisor.remembered == advisor.calls
        assert advisor.calls[0].profile is profile
        assert "Saved:" in str(screen.query_one("#quality-status", Static).render())
        result = str(screen.query_one("#quality-result", Static).render())
        grade = {QualityProfile.SMOKE: "plumbing", QualityProfile.HELLASWAG100: "limited",
                 QualityProfile.IFEVAL: "full"}[profile]
        assert f"Evaluation: {grade} evaluation" in result
        assert "not a general capability assessment" in result
        assert screen.query_one("#quality-tabs", TabbedContent).active == "quality-results-tab"
        assert screen.query_one("#quality-report", Collapsible).collapsed
        metrics = screen.query_one("#quality-measurements")
        # Models sit side by side in one table; the grade travels with every column.
        table = metrics.query_one("#quality-comparison", DataTable)
        cells = [str(cell) for row in range(table.row_count) for cell in table.get_row_at(row)]
        metric_text = "\n".join(
            [*(str(widget.content) for widget in metrics.query(Static)), *cells],
        )
        assert grade.capitalize() in metric_text
        assert "Samples" in metric_text and "Accuracy" in metric_text
        assert "Not a general capability assessment." in metric_text
        await pilot.pause()
        pane = screen.query_one("#quality-results-tab")
        assert pane.region.y <= table.region.y < pane.region.bottom
        app.save_screenshot(f"jaull-quality-{size[0]}-{profile.value}.svg", path="/tmp")


@pytest.mark.parametrize("exit_mode", ["escape", "q", "unmount"])
def test_leaving_quality_requests_cleanup_before_exit(
    tmp_path: Path, exit_mode: str,
) -> None:
    asyncio.run(_leaving_quality(tmp_path, exit_mode))


async def _leaving_quality(
    tmp_path: Path, exit_mode: str,
) -> None:
    advisor = FakeAdvisor()
    advisor.wait = True
    app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
    async with app.run_test() as pilot:
        screen = QualityEvaluationScreen()
        await app.push_screen(screen)
        _fill(screen, _request(tmp_path))
        await pilot.click("#quality-start")
        await pilot.pause(0.05)
        assert screen._busy
        assert screen.query_one("#quality-start", Button).disabled
        if exit_mode == "unmount":
            await app.pop_screen()
        else:
            await pilot.press(exit_mode)
        for _ in range(100):
            if advisor.cleaned.is_set():
                break
            await asyncio.sleep(0.01)
        assert advisor.cleaned.is_set()
        assert screen._stored is None
        assert advisor.remembered == []


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_quality_failure_shows_error_without_storing(
    tmp_path: Path, size: tuple[int, int],
) -> None:
    asyncio.run(_quality_failure(tmp_path, size))


async def _quality_failure(tmp_path: Path, size: tuple[int, int]) -> None:
    advisor = FakeAdvisor()
    advisor.failed = True
    app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
    async with app.run_test(size=size) as pilot:
        screen = QualityEvaluationScreen()
        await app.push_screen(screen)
        _fill(screen, _request(tmp_path))
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert screen._stored is None
        assert "missing Docker image" in str(screen.query_one("#quality-result", Static).render())
        status = screen.query_one("#quality-status", Static)
        assert "missing Docker image" in str(status.render())
        assert status.region.y >= 0
        assert status.region.bottom <= screen.query_one("#quality-actions").region.y
        assert not screen.query_one("#quality-start", Button).disabled


@pytest.mark.parametrize("field", ["repo_id", "filename", "quantization", "sha256", "revision",
                                   "size_bytes", "format"])
def test_quality_manifest_cannot_substitute_selected_variant(field: str) -> None:
    selected = _plan(sha256="d" * 64).artifact.model_copy(
        update={"revision": "f" * 40, "size_bytes": 5},
    )
    artifact = ModelArtifact(
        repo_id=selected.repo_id, revision=selected.revision or "", filename="model.gguf",
        format="gguf", quantization="Q4_K_M", sha256="d" * 64, size_bytes=5,
    )
    check_selected_artifact(artifact, selected)
    mutation = 6 if field == "size_bytes" else "different"
    with pytest.raises(ValueError, match="does not match"):
        check_selected_artifact(artifact.model_copy(update={field: mutation}), selected)


def test_quality_does_not_match_an_unresolved_or_transformers_variant() -> None:
    artifact = ModelArtifact(repo_id="owner/repo", revision="f" * 40,
                             filename="model.gguf", format="gguf", quantization="Q4_K_M")
    with pytest.raises(ValueError, match="single GGUF"):
        check_selected_artifact(artifact, _plan().artifact.model_copy(update={"filename": None}))
    with pytest.raises(ValueError, match="requires a GGUF"):
        check_selected_artifact(artifact, _plan().artifact.model_copy(
            update={"format": ArtifactVariantFormat.SAFETENSORS},
        ))
    with pytest.raises(ValueError, match="Multipart"):
        check_selected_artifact(artifact, _plan().artifact.model_copy(update={"file_count": 3}))


@pytest.mark.parametrize("allow_download", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_selected_quality_prepares_without_json_and_requires_explicit_start(
    tmp_path: Path, allow_download: bool, size: tuple[int, int],
) -> None:
    async def exercise() -> None:
        advisor = FakeAdvisor()
        request = _request(tmp_path)
        advisor.setup = {
            "dataset": str(request.dataset_file), "server": str(request.llama_server),
            "root": str(request.pilot_root), "image": request.image,
        }
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = QualityEvaluationScreen(_plan())
            await app.push_screen(screen)
            await pilot.pause()
            assert not screen.query("#quality-artifact")
            assert advisor.plan_calls == []
            checkbox = screen.query_one("#quality-download", Checkbox)
            assert checkbox.value is False
            checkbox.value = allow_download
            for key, value in advisor.setup.items():
                assert screen.query_one(f"#quality-{key}", Input).value == value
            assert screen.query_one("#quality-start", Button).region.bottom <= size[1] - 1
            await pilot.click("#quality-start")
            await _settled(pilot, screen)
            assert advisor.plan_calls == [(screen._plan, allow_download)]
            assert advisor.calls[0].artifact_json is None
            assert screen._stored is not None
            assert advisor.remembered == advisor.calls
            app.save_screenshot(f"jaull-quality-auto-{size[0]}.svg", path="/tmp")

    asyncio.run(exercise())


def test_paths_quality_action_follows_selected_variant_not_ranking() -> None:
    asyncio.run(_paths_quality_action())


def test_invalid_saved_setup_does_not_crash_or_launch_evaluation() -> None:
    class InvalidSetup(FakeAdvisor):
        def quality_evaluation_setup(self) -> dict[str, str]:
            raise QualityEvaluationError("Invalid quality setup")

    async def exercise() -> None:
        advisor = InvalidSetup()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(_plan())
            await app.push_screen(screen)
            await pilot.pause()
            assert advisor.calls == []
            assert "Invalid quality setup" in str(
                screen.query_one("#quality-status", Static).render()
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_guided_setup_is_explicit_separate_and_never_creates_quality_evidence(
    tmp_path: Path, failed: bool, size: tuple[int, int],
) -> None:
    async def exercise() -> None:
        advisor = FakeAdvisor()
        advisor.failed = failed
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = QualityEvaluationScreen()  # Setup needs no artifact JSON.
            await app.push_screen(screen)
            assert advisor.preparations == []
            _fill(screen, _request(tmp_path))
            screen.query_one("#quality-artifact", Input).value = ""
            consent = screen.query_one("#quality-dataset-download", Checkbox)
            assert consent.value is False
            consent.value = True
            prepare = screen.query_one("#quality-prepare", Button)
            prepare.scroll_visible(animate=False)
            await pilot.pause()
            assert prepare.region.right <= size[0]
            for name in ("prepare", "start", "cancel", "back"):
                action = screen.query_one(f"#quality-{name}", Button)
                assert action.region.right <= size[0]
                assert action.region.bottom <= size[1] - 1
            await pilot.click("#quality-prepare")
            for _ in range(100):
                await pilot.pause(0.02)
                if advisor.preparations and not screen._busy:
                    break
            assert len(advisor.preparations) == 1
            request = advisor.preparations[0]
            assert request.allow_dataset_download is True and request.artifact_json is None
            assert advisor.calls == [] and screen._stored is None
            assert advisor.remembered == ([] if failed else [request])
            text = str(screen.query_one("#quality-status", Static).content)
            assert ("Synthetic setup failure" if failed else "infrastructure ready") in text
            assert not screen.query_one("#quality-start", Button).disabled
            app.save_screenshot(f"jaull-quality-setup-{size[0]}-{failed}.svg", path="/tmp")

    asyncio.run(exercise())


async def _paths_quality_action() -> None:
    class PathsAdvisor(FakeAdvisor):
        def execution_plans_for_recommendation(self, recommendation: Any) -> list[Any]:
            return []

    app = JaullApp(advisor=PathsAdvisor())  # type: ignore[arg-type]
    async with app.run_test(size=(160, 50)) as pilot:
        screen = ExecutionPathsScreen(_gguf_recommendation())
        await app.push_screen(screen)
        await _wait_until(
            pilot, lambda: not screen.query_one("#paths-status", Static).display,
        )
        gguf = _plan()
        transformers = _plan(runtime=RuntimeName.TRANSFORMERS).model_copy(
            update={"plan_id": "transformers", "artifact": gguf.artifact.model_copy(
                update={"format": ArtifactVariantFormat.SAFETENSORS},
            )},
        )
        await screen._paths_loaded(_ExecutionPathsLoaded([gguf, transformers], screen._evidence))
        assert not screen.query_one("#paths-evaluate", Button).disabled
        screen._selected_plan_id = "transformers"
        screen._apply_selection()
        assert screen.query_one("#paths-evaluate", Button).disabled
        screen._selected_plan_id = gguf.plan_id
        screen._apply_selection()
        screen.query_one("#paths-evaluate", Button).press()
        await _wait_until(pilot, lambda: isinstance(app.screen, QualityEvaluationScreen))
        assert isinstance(app.screen, QualityEvaluationScreen)
        assert app.screen._plan == gguf


class ResultsAdvisor(_FakeStoreAdvisor):
    services = None

    def __init__(self, plans: list[Any]) -> None:
        super().__init__()
        self.plans = plans
        self.path_lookups = 0

    def execution_plans_for_recommendation(self, recommendation: Any) -> list[Any]:
        self.path_lookups += 1
        return self.plans

    def quality_evaluation_setup(self) -> dict[str, str]:
        return {}

    def capability_catalog(self) -> CatalogReadResult:
        return _catalog()


@pytest.mark.parametrize("gguf_primary", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_results_evaluation_action_opens_exact_gguf_or_chooser(
    gguf_primary: bool, size: tuple[int, int],
) -> None:
    async def exercise() -> None:
        gguf = _plan(sha256="d" * 64)
        primary = gguf if gguf_primary else _plan(runtime=RuntimeName.TRANSFORMERS).model_copy(
            update={"plan_id": "transformers", "artifact": gguf.artifact.model_copy(update={
                "format": ArtifactVariantFormat.SAFETENSORS, "sha256": None,
            })},
        )
        rec = _gguf_recommendation().model_copy(update={"plan": primary})
        advisor = ResultsAdvisor([primary, gguf] if not gguf_primary else [gguf])
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            results = RecommendationResultsScreen(
                RecommendationWorkflowState(recommendations=[rec]),
            )
            await app.push_screen(results)
            results.query_one(SelectionWorkspace).show_detail()
            results._rows[0].detail.query_one(TabbedContent).active = "tab-3"
            await pilot.pause()
            assert advisor.path_lookups == 0  # No eager network discovery on Results entry.
            button = results.query_one("#res-evaluate-0", Button)
            assert button.region.right <= size[0]
            button.press()
            if gguf_primary:
                await _wait_until(
                    pilot, lambda: isinstance(app.screen, QualityEvaluationScreen),
                )
                assert isinstance(app.screen, QualityEvaluationScreen)
                assert app.screen._plan == gguf
            else:
                await _wait_until(
                    pilot, lambda: isinstance(app.screen, ExecutionPathsScreen)
                    and bool(app.screen.query(TabbedContent))
                    and app.screen.query_one(TabbedContent).active == "paths-evaluation",
                )
                assert isinstance(app.screen, ExecutionPathsScreen)
                assert app.screen.query_one(TabbedContent).active == "paths-evaluation"

    asyncio.run(exercise())


@pytest.mark.parametrize("gguf_primary", [False, True])
def test_results_refreshes_exact_and_alternative_evidence_after_returning(
    gguf_primary: bool,
) -> None:
    async def exercise() -> None:
        record = _record()
        gguf = _plan(sha256=record["identity"]["artifact_sha256"])
        primary = gguf if gguf_primary else _plan(runtime=RuntimeName.TRANSFORMERS).model_copy(
            update={
                "plan_id": "transformers",
                "artifact": gguf.artifact.model_copy(update={
                    "format": ArtifactVariantFormat.SAFETENSORS, "sha256": None,
                }),
            },
        )
        rec = _gguf_recommendation().model_copy(update={"plan": primary})
        before = rec.model_dump_json()
        advisor = ResultsAdvisor([primary, gguf])
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(160, 50)) as pilot:
            results = RecommendationResultsScreen(
                RecommendationWorkflowState(recommendations=[rec]),
            )
            await app.push_screen(results)
            results.query_one("#res-evaluate-0", Button).press()
            target_screen = QualityEvaluationScreen if gguf_primary else ExecutionPathsScreen
            await _wait_until(pilot, lambda: isinstance(app.screen, target_screen))
            if not gguf_primary:
                paths = app.screen
                assert isinstance(paths, ExecutionPathsScreen)
                await _wait_until(pilot, lambda: bool(paths.execution_plans))
                assert paths.execution_plans
            advisor._quality[record["identity_sha256"]] = record
            await app.pop_screen()
            target = "#rec-quality-0" if gguf_primary else "#rec-quality-alternatives-0"
            alternative = results.query_one(target, Static)
            await _wait_until(pilot, lambda: alternative.display)
            assert alternative.display
            if not gguf_primary:
                assert "not the selected artifact" in str(alternative.content)
                assert gguf.artifact.filename in str(alternative.content)
            assert record["identity_sha256"] in str(alternative.tooltip)
            assert results.query_one("#rec-quality-0").display is gguf_primary
            assert results.query_one("#rec-quality-empty-0").display is not gguf_primary
            assert rec.model_dump_json() == before
            assert advisor.path_lookups == (0 if gguf_primary else 1)

    asyncio.run(exercise())


def test_results_alternatives_require_exact_digest_and_model_identity() -> None:
    record = _record()
    matching = _plan(sha256=record["identity"]["artifact_sha256"])
    primary = _plan(runtime=RuntimeName.TRANSFORMERS)
    rec = _gguf_recommendation().model_copy(update={"plan": primary})
    index = EvidenceIndex.load(_FakeStoreAdvisor(quality={record["identity_sha256"]: record}))
    assert len(_alternative_quality(index, rec, (matching, matching))) == 1
    with_alternative = rec.model_copy(update={"alternative_plans": [matching]})
    assert len(_alternative_quality(index, with_alternative, ())) == 1
    gguf_primary = with_alternative.model_copy(update={"plan": matching})
    assert not _alternative_quality(index, gguf_primary, (matching,))
    unrelated = _plan(repo_id="other/different", sha256=matching.artifact.sha256)
    assert not _alternative_quality(index, rec, (
        unrelated, _plan(sha256="7" * 64), _plan(),
        _plan(file_count=3, sha256=matching.artifact.sha256),
    ))


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
@pytest.mark.parametrize("measured", [False, True])
def test_results_separates_published_references_from_exact_artifact_measurements(
    size: tuple[int, int], measured: bool,
) -> None:
    async def exercise() -> None:
        record = _record()
        gguf = _plan(sha256=record["identity"]["artifact_sha256"]).model_copy(
            update={"model_identity": _identity()},
        )
        rec = _gguf_recommendation().model_copy(update={"plan": gguf})
        state = RecommendationWorkflowState(recommendations=[rec])
        before = state.model_dump_json()
        advisor = ResultsAdvisor([gguf])
        if measured:
            advisor._quality[record["identity_sha256"]] = record
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            results = RecommendationResultsScreen(state)
            await app.push_screen(results)
            results.query_one(SelectionWorkspace).show_detail()
            results._rows[0].detail.query_one(TabbedContent).active = "tab-3"
            for _ in range(100):
                await pilot.pause(0.02)
                if "75%" in str(results.query_one("#rec-published-0", Static).content):
                    break
            published = str(results.query_one("#rec-published-0", Static).content)
            assert "publisher-reported" in published and "independent" in published
            assert "non-thinking" in published and "not measured on this artifact" in published
            assert "75%" in published and "65%" in published
            assert results.query_one("#rec-quality-0").display is measured
            assert results.query_one("#rec-quality-empty-0").display is not measured
            assert results.query_one("#rec-published-provenance-0").display
            assert '"reasoning_mode": "non-thinking"' in str(
                results.query_one("#rec-published-details-0", Static).content
            )
            if measured:
                assert "plumbing" in str(results.query_one("#rec-quality-0", Static).content)
            assert state.model_dump_json() == before and advisor.path_lookups == 0
            button = results.query_one("#res-evaluate-0", Button)
            assert not button.disabled
            assert button.parent is not None
            # `scroll_visible` is deferred: it does not move the offset in the
            # call, only on a later refresh. The evidence worker is also still
            # growing the pane when the loop above exits — measured here, the
            # virtual height went 20 to 22 rows after it broke. Counting
            # refreshes encodes one machine's scheduling; waiting for the
            # button to actually be inside the viewport is the condition this
            # test is about, and it holds wherever it runs.
            for _ in range(100):
                button.scroll_visible(animate=False)
                await pilot.pause()
                if button.parent.content_region.contains_region(button.region):
                    break
            assert button.parent.content_region.contains_region(button.region)
            assert app.get_widget_at(*button.region.center)[0] is button
            app.save_screenshot(f"jaull-published-{size[0]}-{measured}.svg", path="/tmp")
            await pilot.click("#res-evaluate-0")
            await pilot.pause()
            assert isinstance(app.screen, QualityEvaluationScreen)
            assert app.screen._plan == gguf

    asyncio.run(exercise())


def test_paths_published_references_follow_selected_model_and_never_fill_local_metrics() -> None:
    async def exercise() -> None:
        gguf = _plan().model_copy(update={"model_identity": _identity()})
        unrelated = _plan(repo_id="other/Unrelated-GGUF").model_copy(update={
            "model_identity": _identity("other/NoCatalogEntry"),
        })
        advisor = ResultsAdvisor([gguf, unrelated])
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(160, 50)) as pilot:
            screen = ExecutionPathsScreen(_gguf_recommendation())
            await app.push_screen(screen)
            await pilot.pause()
            index = EvidenceIndex.load(advisor)  # type: ignore[arg-type]
            await screen._paths_loaded(_ExecutionPathsLoaded([gguf, unrelated], index))
            screen._selected_plan_id = gguf.plan_id
            screen._apply_selection()
            assert "75%" in str(screen.query_one("#paths-published", Static).content)
            assert not screen.query_one("#paths-quality").display
            assert screen.query_one("#paths-quality-empty").display
            screen._selected_plan_id = unrelated.plan_id
            screen._apply_selection()
            text = str(screen.query_one("#paths-published", Static).content)
            assert "Absence is not poor quality" in text and "75%" not in text
            assert not screen.query_one("#paths-published-provenance").display
            assert str(screen.query_one("#paths-published-details", Static).content) == ""

    asyncio.run(exercise())


@pytest.mark.parametrize("problem,reason", [
    ("multipart", "Multipart GGUF"), ("missing-file", "single GGUF"),
    ("no-gpu", "one CUDA GPU"),
])
def test_results_disables_known_incompatible_quality_paths_with_visible_reason(
    problem: str, reason: str,
) -> None:
    async def exercise() -> None:
        plan = _plan()
        if problem == "no-gpu":
            plan = plan.model_copy(update={"hardware": hardware(vram_gib=None)})
        else:
            change = {"file_count": 3} if problem == "multipart" else {"filename": None}
            plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update=change)})
        assert reason in (quality_evaluation_block_reason(plan) or "")
        rec = _gguf_recommendation().model_copy(update={"plan": plan})
        app = JaullApp(advisor=ResultsAdvisor([plan]))  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = RecommendationResultsScreen(RecommendationWorkflowState(recommendations=[rec]))
            await app.push_screen(screen)
            await pilot.pause()
            assert screen.query_one("#res-evaluate-0", Button).disabled
            assert reason in str(screen.query_one("#rec-evaluate-reason-0", Static).content)
            assert screen.query_one("#rec-evaluate-reason-0").display
            assert not screen.query_one("#res-paths-0", Button).disabled

    asyncio.run(exercise())
