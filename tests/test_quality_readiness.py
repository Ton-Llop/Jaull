"""Automatic evaluator setup and its readiness panel. Docker is simulated; nothing runs."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, DataTable, Input, Static

from jaull.advisor.quality import QualityReadiness, check_quality_readiness
from jaull.recommendation.quality import StoredQuality
from jaull.runtime import quality_eval_runner as runner
from jaull.runtime.quality_eval_runner import (
    BUILD_COMMAND,
    QualityEvaluationError,
    QualityProfile,
    find_evaluator_image,
)
from jaull.tui.app import JaullApp
from jaull.tui.evidence import PlanEvidence
from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen
from tests.test_cli_quality import _request
from tests.test_tui_ifeval import _state
from tests.test_tui_quality import CandidateAdvisor, _settled

CONTRACT = {"io.jaull.quality.artifact-contract": "exact-local-gguf-v1"}
IMAGES = {
    "jaull-quality-eval:gguf-v1": CONTRACT,
    "jaull-quality-eval:ifeval-chat-v1": CONTRACT | {
        "io.jaull.quality.suites": "hellaswag-v1,ifeval-chat-v1",
    },
    "jaull-quality-eval:other": {"io.jaull.quality.suites": "ifeval-chat-v1"},  # No contract.
}


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def check_output(command: list[str], **kwargs: Any) -> str:
        calls.append(command)
        if command[2] == "ls":
            return "\n".join(["jaull-quality-eval:other", *IMAGES, "jaull-quality-eval:<none>"])
        return json.dumps(IMAGES.get(command[-1]))

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)
    return calls


def test_ifeval_finds_a_capable_image_even_when_the_remembered_one_is_older(docker) -> None:
    assert find_evaluator_image(
        QualityProfile.IFEVAL, "jaull-quality-eval:gguf-v1",
    ) == "jaull-quality-eval:ifeval-chat-v1"
    # Discovery only lists and inspects: never pull, build or run.
    assert {call[2] for call in docker} <= {"ls", "inspect"}


def test_hellaswag_keeps_a_remembered_image_from_before_the_suite_label(docker) -> None:
    assert find_evaluator_image(
        QualityProfile.SMOKE, "jaull-quality-eval:gguf-v1",
    ) == "jaull-quality-eval:gguf-v1"


def test_an_image_without_the_contract_is_never_chosen(docker) -> None:
    del IMAGES["jaull-quality-eval:ifeval-chat-v1"]
    try:
        assert find_evaluator_image(QualityProfile.IFEVAL) is None
    finally:
        IMAGES["jaull-quality-eval:ifeval-chat-v1"] = CONTRACT | {
            "io.jaull.quality.suites": "hellaswag-v1,ifeval-chat-v1",
        }


def test_no_docker_means_no_image_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> str:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(runner.subprocess, "check_output", missing)
    assert find_evaluator_image(QualityProfile.IFEVAL) is None
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("d", 15)))
    assert find_evaluator_image(QualityProfile.IFEVAL) is None


def _values(tmp_path: Path, *, dataset: bool = True) -> dict[str, str]:
    request = _request(tmp_path)  # A synthetic trusted checkout, server and dataset.
    if not dataset:
        request.dataset_file.unlink()
    return {"dataset": str(request.dataset_file), "server": str(request.llama_server),
            "root": str(request.pilot_root), "image": "", "output": str(tmp_path / "out")}


def _check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, image: str | None,
           dataset: bool = True, failure: str | None = None) -> tuple[QualityReadiness, list[Any]]:
    monkeypatch.setattr(runner, "find_evaluator_image", lambda profile, preferred=None: image)
    prepared: list[Any] = []

    def prepare(request: Any, **kwargs: Any) -> None:
        prepared.append(request)
        if failure:
            raise QualityEvaluationError(failure)

    readiness = check_quality_readiness(
        prepare, QualityProfile.IFEVAL, _values(tmp_path, dataset=dataset),
    )
    return readiness, prepared


def test_a_missing_image_is_red_and_says_how_to_build_it(tmp_path, monkeypatch) -> None:
    readiness, prepared = _check(tmp_path, monkeypatch, image=None)
    assert readiness.blocked and prepared == []
    state, name, detail = readiness.rows[0]
    assert (state, name) == ("bad", "Evaluator image") and BUILD_COMMAND in detail


def test_a_missing_dataset_is_amber_and_nothing_is_fetched(tmp_path, monkeypatch) -> None:
    readiness, prepared = _check(tmp_path, monkeypatch, image="img:ok", dataset=False)
    assert not readiness.blocked and readiness.dataset_missing
    assert prepared == []  # The preflight would need the file; it runs at start instead.
    assert ("warn", "Verification", "Runs at start, after the dataset download.") in readiness.rows


def test_with_everything_local_the_preflight_runs_on_the_found_image(tmp_path, monkeypatch) -> None:
    readiness, prepared = _check(tmp_path, monkeypatch, image="img:ok")
    assert not readiness.blocked and readiness.image == "img:ok"
    request, = prepared
    assert request.image == "img:ok" and not request.allow_dataset_download
    assert readiness.rows[-1][:2] == ("ok", "Verification")


def test_a_failing_preflight_is_red_with_its_own_reason(tmp_path, monkeypatch) -> None:
    readiness, _ = _check(tmp_path, monkeypatch, image="img:ok",
                          failure="Runtime unavailable: llama-server differs from the audited pin.")
    assert readiness.blocked
    assert readiness.rows[-1] == (
        "bad", "Verification", "Runtime unavailable: llama-server differs from the audited pin.",
    )


class ReadinessAdvisor(CandidateAdvisor):
    def __init__(self, readiness: QualityReadiness) -> None:
        super().__init__()
        self.readiness = readiness

    def quality_readiness(self, profile: Any, values: Any, **kwargs: Any) -> QualityReadiness:
        return self.readiness


def _screen_with(readiness: QualityReadiness):
    async def scenario(check) -> None:
        advisor = ReadinessAdvisor(readiness)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            for _ in range(50):
                await pilot.pause(0.02)
                if not screen._checking:
                    break
            await check(screen, pilot)
    return scenario


def test_the_panel_shows_each_check_and_fills_the_found_image() -> None:
    readiness = QualityReadiness((
        ("ok", "Evaluator image", "jaull-quality-eval:ifeval-chat-v1"),
        ("warn", "Dataset", "Missing; 0.2 MB download at start, with permission."),
    ), image="jaull-quality-eval:ifeval-chat-v1")

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        panel = str(screen.query_one("#quality-readiness", Static).render())
        assert "✓ Evaluator image" in panel and "! Dataset" in panel
        assert screen.query_one("#quality-image", Input).value == readiness.image
        assert not screen.query_one("#quality-start", Button).disabled
        downloads = str(screen.query_one("#quality-downloads", Static).render())
        assert "To download: dataset 0.2 MB" in downloads and "needs permission" in downloads
        # Each permission covers only what it downloads: models still wait for theirs.
        screen.query_one("#quality-dataset-download", Checkbox).value = True
        await pilot.pause()
        assert "needs permission" in str(screen.query_one("#quality-downloads", Static).render())
        screen.query_one("#quality-download", Checkbox).value = True
        await pilot.pause()
        assert "permitted" in str(screen.query_one("#quality-downloads", Static).render())

    asyncio.run(_screen_with(readiness)(check))


def test_a_red_row_disables_start_and_says_why() -> None:
    readiness = QualityReadiness((
        ("bad", "Evaluator image", "No local image supports ifeval. Build it once: x"),
    ))

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        start = screen.query_one("#quality-start", Button)
        assert start.disabled
        assert "readiness check failed" in str(start.tooltip)
        assert "Not ready: No local image supports ifeval" in str(
            screen.query_one("#quality-status", Static).render(),
        )

    asyncio.run(_screen_with(readiness)(check))


def test_a_candidate_already_measured_starts_unchecked_and_says_why(monkeypatch) -> None:
    ready = QualityReadiness((("ok", "Evaluator image", "img"),), image="img")

    async def scenario() -> None:
        advisor = ReadinessAdvisor(ready)
        first, *rest = advisor.candidates
        stored = SimpleNamespace(
            summary="strict 140/541", suite="jaull-ifeval-chat-v1", classification="full",
            samples_used=541, samples_available=541, limitations=(), context_length=4096,
            metrics=(SimpleNamespace(name="prompt_level_strict_acc", value=0.26,
                                     correct=140, samples=541),),
        )
        applicable = StoredQuality(stored, "r" * 64, "cohort", complete=1, distinct=1)
        advisor.candidates = (replace(first, stored=applicable), *rest)
        stored_details = "Stored details"
        monkeypatch.setattr(PlanEvidence, "quality_details", lambda self: stored_details)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            boxes = [screen.query_one(f"#quality-candidate-{i}", Checkbox).value for i in range(3)]
            assert boxes == [False, True, True]
            badge = str(screen.query_one("#quality-row-0 .quality-measured", Static).render())
            assert "Already measured (strict 140/541); unchecked" in badge
            # Its stored result is already in Results: nothing had to run to see it.
            table = screen.query_one("#quality-comparison", DataTable)
            assert [str(c.label) for c in table.columns.values()][1].endswith("(stored)")
            for index in (1, 2):
                screen.query_one(f"#quality-candidate-{index}", Checkbox).value = False
            await pilot.pause()
            status = screen.query_one("#quality-status", Static)
            await pilot.click("#quality-start")
            await _settled(pilot, screen)
            assert "Select at least one candidate." in str(status.render())
            # Everything measured: say so before any path check could get in the way.
            screen._selection = replace(screen._selection, candidates=tuple(
                replace(candidate, stored=applicable)
                for candidate in screen._selection.candidates
            ))
            await pilot.pause(0.6)  # Textual debounces two successive button clicks.
            await pilot.click("#quality-start")
            await _settled(pilot, screen)
            assert "Every candidate is already measured" in str(status.render())

    asyncio.run(scenario())


SAME_PROTOCOL = (("suite", "s"), ("evaluator", "e"), ("classification", "full"))


def _run(comparison: tuple[tuple[str, str], ...] = SAME_PROTOCOL, **values: float) -> Any:
    return SimpleNamespace(
        suite="jaull-ifeval-chat-v1", classification="full", samples_used=541,
        samples_available=541, comparison=comparison, limitations=(), context_length=4096,
        summary="synthetic", metrics=tuple(
            SimpleNamespace(name=name, value=value, correct=1, samples=1)
            for name, value in values.items()
        ),
    )


def _cells(table: Any, label: str) -> list[Any]:
    for row in range(table.row_count):
        cells = table.get_row_at(row)
        if str(cells[0]) == label:
            return list(cells[1:])
    raise AssertionError(f"no row {label!r}")


def test_the_comparison_puts_models_side_by_side_and_marks_the_best() -> None:
    from jaull.tui import palette
    from jaull.tui.screens.quality_evaluation import comparison_table

    async def scenario() -> None:
        async with App().run_test():
            table = comparison_table((
                ("Qwen", _run(prompt_level_strict_acc=0.26, inst_level_strict_acc=0.36)),
                ("LFM", _run(prompt_level_strict_acc=0.80)),
            ))
            columns = [str(column.label) for column in table.columns.values()]
            assert columns == ["", "Qwen", "LFM"]
            assert [str(cell) for cell in _cells(table, "Suite")] == ["jaull-ifeval-chat-v1"] * 2
            qwen, lfm = _cells(table, "Prompt strict")
            assert str(lfm).endswith("80.0%")
            assert palette.OK in str(lfm.spans[-1].style) and palette.OK not in str(qwen.style)
            assert str(_cells(table, "Instr. strict")[1]) == "—"  # Not measured: never borrowed.

    asyncio.run(scenario())


def test_runs_the_comparator_would_refuse_show_no_winner_and_say_why() -> None:
    from jaull.tui import palette
    from jaull.tui.screens.quality_evaluation import comparison_table, not_comparable

    other = (("suite", "s"), ("evaluator", "other"), ("classification", "full"))
    runs = [_run(prompt_level_strict_acc=0.26), _run(other, prompt_level_strict_acc=0.80)]
    assert not_comparable(runs) == "Not comparable: evaluator differs; no winner is marked."
    assert not_comparable(runs[:1]) is None and not_comparable([runs[0], runs[0]]) is None

    async def scenario() -> None:
        async with App().run_test():
            table = comparison_table((("A", runs[0]), ("B", runs[1])))
            styles = [str(style) for cell in _cells(table, "Prompt strict")
                      for style in (cell.style, *(span.style for span in cell.spans))]
            assert styles and all(palette.OK not in style for style in styles)

    asyncio.run(scenario())


def test_comparison_fields_match_exactly_when_comparison_keys_do() -> None:
    import copy

    from jaull.evaluation.quality_comparison import comparison_fields, comparison_key
    from tests.test_quality_generation_records import record

    base = record()
    other = copy.deepcopy(base)
    other["identity"]["artifact_sha256"] = "f" * 64  # Another model: still comparable.
    changed = copy.deepcopy(base)
    changed["identity"]["evaluator"]["context"] += 1
    assert comparison_fields(base) == comparison_fields(other)
    assert comparison_key(base) == comparison_key(other)
    assert comparison_key(base) != comparison_key(changed)
    differing = [name for (name, a), (_, b) in zip(
        comparison_fields(base), comparison_fields(changed), strict=True,
    ) if a != b]
    assert differing == ["evaluator"]


class _SuiteAwareAdvisor(ReadinessAdvisor):
    """IFEval candidates carry a stored run; any other suite has none, as in production."""

    def __init__(self, readiness: QualityReadiness, stored: Any) -> None:
        super().__init__(readiness)
        self.stored = stored

    def prepare_quality_candidates(self, state: Any, **kwargs: Any) -> Any:
        selection = super().prepare_quality_candidates(state, **kwargs)
        if kwargs.get("profile") is not QualityProfile.IFEVAL:
            return selection
        first, *rest = selection.candidates
        applicable = StoredQuality(self.stored, "r" * 64, "cohort", complete=1, distinct=1)
        return replace(selection, candidates=(replace(first, stored=applicable), *rest))


def test_changing_suite_clears_results_of_the_previous_one(monkeypatch) -> None:
    from textual.widgets import Select

    ready = QualityReadiness((("ok", "Evaluator image", "img"),), image="img")
    monkeypatch.setattr(PlanEvidence, "quality_details", lambda self: "details")

    async def scenario() -> None:
        app = JaullApp(advisor=_SuiteAwareAdvisor(ready, _run(prompt_level_strict_acc=0.5)))
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            assert screen.query("#quality-comparison")  # IFEval: its stored run shows.
            screen.query_one("#quality-profile", Select).value = QualityProfile.SMOKE.value
            await _settled(pilot, screen)
            for _ in range(20):
                await pilot.pause(0.02)
            assert not screen.query("#quality-comparison")

    asyncio.run(scenario())


def test_a_successful_prepare_checks_again_and_unblocks_start() -> None:
    blocked = QualityReadiness((("bad", "Evaluator image", "No image. Build it once: x"),))
    ready = QualityReadiness((("ok", "Evaluator image", "img"),), image="img")

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        start = screen.query_one("#quality-start", Button)
        assert start.disabled
        advisor = screen.app.advisor  # type: ignore[attr-defined]
        advisor.readiness = ready  # The user fixed the settings.
        for key in ("server", "root", "image", "output", "dataset"):
            screen.query_one(f"#quality-{key}", Input).value = screen.query_one(
                f"#quality-{key}", Input,
            ).value or "/x"
        await pilot.click("#quality-prepare")
        for _ in range(50):
            await pilot.pause(0.02)
            if not screen._busy and not screen._checking:
                break
        assert advisor.preparations and not start.disabled

    asyncio.run(_screen_with(blocked)(check))


def test_a_conflict_badge_shows_no_score_so_store_order_cannot_pick_one() -> None:
    from jaull.tui.screens.quality_evaluation import stored_badge

    conflict = StoredQuality(complete=2, distinct=2)
    text, style = stored_badge(conflict) or ("", "")
    assert style == "status-fail" and "2 complete runs disagree" in text
    assert "%" not in text and "/" not in text
    assert stored_badge(StoredQuality()) is None  # References only: offered as usual.
