"""Synthetic UI pilot tests: no Docker, network, model or GPU execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from textual.widgets import Button, Input, Select, Static

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import ArtifactVariantFormat
from jaull.domain.runtime import RuntimeName
from jaull.runtime.quality_eval_runner import (
    QualityEvaluationCancelled,
    QualityProfile,
    QualityRunRequest,
)
from jaull.tui.app import JaullApp
from jaull.tui.screens.execution_paths import ExecutionPathsScreen, _ExecutionPathsLoaded
from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen, check_selected_artifact
from tests.test_cli_quality import _record, _request
from tests.test_execution_plans import _gguf_recommendation
from tests.test_tui_evidence import _plan


class FakeAdvisor:
    services = None

    def __init__(self) -> None:
        self.calls: list[QualityRunRequest] = []
        self.wait = False
        self.failed = False
        self.cleaned = Event()

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
        if self.calls[-1].profile is QualityProfile.HELLASWAG100:
            record["classification"] = "limited"
        return record


def _fill(screen: QualityEvaluationScreen, request: QualityRunRequest) -> None:
    for key, value in (
        ("artifact", request.artifact_json), ("dataset", request.dataset_file),
        ("server", request.llama_server), ("root", request.pilot_root),
        ("image", request.image), ("output", request.output),
    ):
        screen.query_one(f"#quality-{key}", Input).value = str(value)


async def _settled(pilot: Any, screen: QualityEvaluationScreen) -> None:
    for _ in range(100):
        await pilot.pause(0.02)
        if not screen._busy:
            return
    pytest.fail("Quality screen did not finish")


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
        assert advisor.calls == []  # Empty inputs must never launch Docker.
        text = str(screen.query_one("#quality-status", Static).render())
        assert "Missing:" in text
        for label in ("artifact manifest", "validation dataset", "llama-server executable"):
            assert label in text
        assert app.focused is screen.query_one("#quality-artifact", Input)
        status = screen.query_one("#quality-status", Static)
        assert status.region.y >= 0
        assert status.region.bottom <= screen.query_one("#quality-actions").region.y
        _fill(screen, _request(tmp_path))
        screen.query_one("#quality-profile", Select).value = profile.value
        await pilot.pause(0.6)  # Textual debounces two successive button clicks.
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert len(advisor.calls) == 1
        assert advisor.calls[0].profile is profile
        assert "Saved:" in str(screen.query_one("#quality-status", Static).render())
        result = str(screen.query_one("#quality-result", Static).render())
        grade = "plumbing" if profile is QualityProfile.SMOKE else "limited"
        assert f"Evaluation: {grade} evaluation" in result
        assert "not a general capability assessment" in result
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


def test_manifest_mismatch_reaches_no_evaluator(tmp_path: Path) -> None:
    async def exercise() -> None:
        advisor = FakeAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test() as pilot:
            screen = QualityEvaluationScreen(_plan())
            await app.push_screen(screen)
            _fill(screen, _request(tmp_path))  # synthetic/model != owner/repo
            await pilot.click("#quality-start")
            await _settled(pilot, screen)
            assert advisor.calls == []
            assert screen._stored is None
            assert "does not match" in str(screen.query_one("#quality-result", Static).render())

    asyncio.run(exercise())


def test_paths_quality_action_follows_selected_variant_not_ranking() -> None:
    asyncio.run(_paths_quality_action())


async def _paths_quality_action() -> None:
    class PathsAdvisor(FakeAdvisor):
        def execution_plans_for_recommendation(self, recommendation: Any) -> list[Any]:
            return []

    app = JaullApp(advisor=PathsAdvisor())  # type: ignore[arg-type]
    async with app.run_test(size=(160, 50)) as pilot:
        screen = ExecutionPathsScreen(_gguf_recommendation())
        await app.push_screen(screen)
        await pilot.pause()
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
        await pilot.pause()
        assert isinstance(app.screen, QualityEvaluationScreen)
        assert app.screen._plan == gguf
