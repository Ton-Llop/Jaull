"""Evaluate candidates follows the search's priority: speed, balanced or quality.

Synthetic advisor and assessments; nothing is measured. The screen must pick the
suite the priority asks for, run speed on the host without the quality evaluator,
and show what it measured beside what was already stored.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Checkbox, DataTable, Select, Static, TabbedContent

from jaull.advisor.quality_candidates import QualityCandidateOutcome, QualityCandidateSelection
from jaull.domain.recommendation import ApplicableSpeed, SpeedAssessment
from jaull.domain.requirements import RecommendationPriority
from jaull.runtime.quality_eval_runner import QualityProfile
from jaull.tui.app import JaullApp
from jaull.tui.screens import quality_evaluation
from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen
from tests.test_tui_ifeval import ProfileAdvisor, _state
from tests.test_tui_quality import _fill, _request, _settled, _wait_until

COHORT = "c" * 64


def _speed(value: float) -> ApplicableSpeed:
    return ApplicableSpeed(tokens_per_second=value, repetitions=5, cohort=COHORT,
                           benchmark_id=f"bench-{value}", run_count=3,
                           benchmark_ids=tuple(f"bench-{value}-{i}" for i in range(3)),
                           observed_min=value, observed_max=value)


class SpeedAdvisor(ProfileAdvisor):
    def __init__(self, *, measured_first: bool = False) -> None:
        super().__init__()
        self.speed_selections: list[Any] = []
        self.speed_runs: list[tuple[Any, ...]] = []
        if measured_first:
            first, *rest = self.candidates
            self.candidates = (replace(first, speed=SpeedAssessment(
                applicability="measured_plan", applicable=_speed(272.0),
            )), *rest)

    def prepare_speed_candidates(self, state: Any, **kwargs: Any) -> QualityCandidateSelection:
        self.speed_selections.append(kwargs.get("with_quality_profile"))
        return QualityCandidateSelection(self.candidates, (), len(self.candidates))

    def run_speed_candidates(self, candidates: Any, **kwargs: Any) -> Any:
        self.speed_runs.append(tuple(c.plan.plan_id for c in candidates))
        return tuple(QualityCandidateOutcome(c.plan, path=Path(f"/bench/bench-{i}.json"))
                     for i, c in enumerate(candidates))

    def load_benchmark_record(self, benchmark_id: str) -> Any:
        return benchmark_id


def _with(priority: RecommendationPriority) -> Any:
    state = _state()
    assert state.requirements is not None
    return state.model_copy(update={
        "requirements": state.requirements.model_copy(update={"priority": priority}),
    })


def _run(advisor: SpeedAdvisor, priority: RecommendationPriority, check: Any) -> None:
    async def scenario() -> None:
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_with(priority))
            await app.push_screen(screen)
            await _settled(pilot, screen)
            for _ in range(50):
                await pilot.pause(0.02)
                if not screen._checking:
                    break
            await check(screen, pilot)
    asyncio.run(scenario())


def test_a_speed_search_measures_speed_without_the_quality_evaluator(monkeypatch) -> None:
    measured = iter((_speed(164.5), _speed(271.7), _speed(235.1)))
    monkeypatch.setattr(quality_evaluation, "assess_speed", lambda plan, req, records:
                        SpeedAssessment(applicability="measured_plan", applicable=next(measured)))
    advisor = SpeedAdvisor()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        assert screen.query_one("#quality-profile", Select).value == "speed"
        assert advisor.speed_selections == [None] and advisor.profiles == []
        assert "no Docker" in str(screen.query_one("#quality-readiness", Static).render())
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert advisor.speed_runs == [tuple(c.plan.plan_id for c in advisor.candidates)]
        assert advisor.calls == []  # The quality evaluator never ran.
        table = screen.query_one("#quality-speed-comparison", DataTable)
        tg128 = [str(cell) for cell in table.get_row_at(0)[1:]]
        assert tg128 == ["164.5 tok/s", "271.7 tok/s", "235.1 tok/s"]
        fastest = table.get_row_at(0)[2]
        assert "bold" in str(fastest.style)  # One cohort: the fastest is marked.

    _run(advisor, RecommendationPriority.SPEED, check)


def test_a_balanced_search_selects_ifeval_candidates_with_their_speed() -> None:
    advisor = SpeedAdvisor()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        assert screen.query_one("#quality-profile", Select).value == "balanced"
        assert advisor.speed_selections == [QualityProfile.IFEVAL]
        assert screen._profile is QualityProfile.IFEVAL  # Quality side runs IFEval.

    _run(advisor, RecommendationPriority.BALANCED, check)


def test_a_plan_whose_speed_already_applies_starts_unchecked() -> None:
    advisor = SpeedAdvisor(measured_first=True)

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        boxes = [screen.query_one(f"#quality-candidate-{i}", Checkbox).value for i in range(3)]
        assert boxes == [False, True, True]
        badge = str(screen.query_one("#quality-row-0 .quality-measured", Static).render())
        assert "Speed already measured (tg128 272.0 tok/s)" in badge
        # Its stored figure is in Results before anything runs.
        assert screen.query("#quality-speed-comparison")

    _run(advisor, RecommendationPriority.SPEED, check)


def test_a_quality_search_keeps_offering_quality_first() -> None:
    advisor = SpeedAdvisor()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        assert screen.query_one("#quality-profile", Select).value == "ifeval"
        assert advisor.speed_selections == [] and advisor.profiles == [QualityProfile.IFEVAL]

    _run(advisor, RecommendationPriority.QUALITY, check)


def test_an_unreadable_benchmark_reports_the_error_and_frees_the_screen() -> None:
    class Broken(SpeedAdvisor):
        def load_benchmark_record(self, benchmark_id: str) -> Any:
            raise OSError("disk read failed")

    advisor = Broken()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert not screen._busy
        assert "Could not read it: disk read failed" in str(
            screen.query_one("#quality-result", Static).render()
        )

    _run(advisor, RecommendationPriority.SPEED, check)


def test_balanced_runs_only_the_axis_each_candidate_is_missing(tmp_path: Path) -> None:
    from jaull.recommendation.quality import StoredQuality

    class Balanced(SpeedAdvisor):
        def __init__(self) -> None:
            super().__init__(measured_first=True)  # Model-0: speed already applies.
            first, second, third = self.candidates
            # Model-1: two different IFEval results; a third run cannot settle it.
            self.candidates = (first, replace(second, stored=StoredQuality(distinct=2)), third)
            self.quality_runs: list[tuple[Any, ...]] = []

        def run_quality_candidates(self, candidates: Any, request: Any, **kwargs: Any) -> Any:
            self.quality_runs.append(tuple(c.plan.plan_id for c in candidates))
            return ()

    advisor = Balanced()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        ids = [c.plan.plan_id for c in advisor.candidates]
        _fill(screen, _request(tmp_path))
        await pilot.pause(0.6)
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert advisor.quality_runs == [(ids[0], ids[2])]
        assert advisor.speed_runs == [(ids[1], ids[2])]

    _run(advisor, RecommendationPriority.BALANCED, check)


def test_balanced_with_quality_settled_measures_speed_without_the_evaluator() -> None:
    from jaull.recommendation.quality import StoredQuality

    class SpeedOnly(SpeedAdvisor):
        def __init__(self) -> None:
            super().__init__()
            # Every plan's IFEval is settled (here in conflict); only speed is missing.
            self.candidates = tuple(replace(c, stored=StoredQuality(distinct=2))
                                    for c in self.candidates)

        def quality_readiness(self, *args: Any, **kwargs: Any) -> Any:
            from jaull.advisor.quality import QualityReadiness
            return QualityReadiness((("bad", "Docker", "Docker is not running."),))

    advisor = SpeedOnly()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        assert not screen.query_one("#quality-start").disabled  # Docker is not needed.
        await pilot.click("#quality-start")
        await _settled(pilot, screen)
        assert advisor.speed_runs == [tuple(c.plan.plan_id for c in advisor.candidates)]
        assert advisor.calls == []  # IFEval never ran.

    _run(advisor, RecommendationPriority.BALANCED, check)


def test_balanced_with_speed_settled_runs_only_quality(tmp_path: Path) -> None:
    from jaull.advisor.quality_candidates import run_speed_candidates

    class QualityOnly(SpeedAdvisor):
        def __init__(self) -> None:
            super().__init__()
            self.candidates = tuple(replace(c, speed=SpeedAssessment(
                applicability="measured_plan", applicable=_speed(200.0 + i),
            )) for i, c in enumerate(self.candidates))
            self.quality_runs: list[tuple[Any, ...]] = []

        def run_quality_candidates(self, candidates: Any, request: Any, **kwargs: Any) -> Any:
            self.quality_runs.append(tuple(c.plan.plan_id for c in candidates))
            return ()

        def run_speed_candidates(self, candidates: Any, **kwargs: Any) -> Any:
            self.speed_runs.append(tuple(c.plan.plan_id for c in candidates))
            return run_speed_candidates(self, candidates, **kwargs)  # type: ignore[arg-type]

    advisor = QualityOnly()

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        _fill(screen, _request(tmp_path))
        await pilot.pause()
        assert await pilot.click("#quality-start")
        await _wait_until(pilot, lambda: bool(advisor.quality_runs) and not screen._busy)
        assert advisor.quality_runs == [tuple(c.plan.plan_id for c in advisor.candidates)]
        assert advisor.speed_runs == []
        assert "Speed measurement failed" not in str(
            screen.query_one("#quality-result", Static).content
        )
        table = screen.query_one("#quality-speed-comparison", DataTable)
        assert [str(cell) for cell in table.get_row_at(0)[1:]] == [
            "200.0 tok/s", "201.0 tok/s", "202.0 tok/s",
        ]

    _run(advisor, RecommendationPriority.BALANCED, check)


@pytest.mark.parametrize("priority", [
    RecommendationPriority.SPEED, RecommendationPriority.QUALITY, RecommendationPriority.BALANCED,
])
def test_finished_measurements_refresh_candidates_without_reselecting(
    priority: RecommendationPriority, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaull.evaluation.quality_records import describe_record
    from jaull.recommendation.quality import StoredQuality
    from tests.test_ifeval_applicability import _full_record

    record = _full_record()
    evidence = describe_record(record)

    class Persistent(SpeedAdvisor):
        def __init__(self) -> None:
            super().__init__()
            self.quality_runs: list[tuple[Any, ...]] = []
            self.refreshes = 0

        def run_quality_candidates(self, candidates: Any, request: Any, **kwargs: Any) -> Any:
            self.quality_runs.append(tuple(c.plan.plan_id for c in candidates))
            ran = {c.plan.plan_id for c in candidates}
            self.candidates = tuple(replace(c, downloaded=True, stored=StoredQuality(
                applicable=evidence, complete=1, distinct=1,
            )) if c.plan.plan_id in ran else c for c in self.candidates)
            return tuple(QualityCandidateOutcome(c.plan, path=Path(f"/quality/quality-{i}.json"))
                         for i, c in enumerate(candidates))

        def load_quality_record(self, record_id: str) -> Any:
            return record

        def run_speed_candidates(self, candidates: Any, **kwargs: Any) -> Any:
            outcomes = super().run_speed_candidates(candidates, **kwargs)
            ran = {c.plan.plan_id for c in candidates}
            self.candidates = tuple(replace(c, downloaded=True, speed=SpeedAssessment(
                applicability="measured_plan", applicable=_speed(200.0),
            )) if c.plan.plan_id in ran else c for c in self.candidates)
            return outcomes

        def refresh_quality_candidates(self, selection: Any, requirements: Any, **kwargs: Any):
            self.refreshes += 1
            return replace(selection, candidates=self.candidates)

    advisor = Persistent()
    monkeypatch.setattr(quality_evaluation, "assess_speed", lambda *args:
                        SpeedAssessment(applicability="measured_plan", applicable=_speed(200.0)))

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        before = screen._search_state.model_dump_json()
        _fill(screen, _request(tmp_path))
        await pilot.pause()
        assert await pilot.click("#quality-start")
        await _wait_until(pilot, lambda: (
            bool(advisor.speed_runs or advisor.quality_runs) and not screen._busy
        ))
        assert advisor.refreshes == 1
        assert all(screen._settled(c) for c in screen._selection.candidates)
        assert all(c.downloaded for c in screen._selection.candidates)
        assert not screen._checked()
        assert len(screen.query(".quality-measured")) == (
            6 if priority is RecommendationPriority.BALANCED else 3
        )
        if priority is not RecommendationPriority.QUALITY:
            assert screen.query("#quality-speed-comparison")
        if priority is not RecommendationPriority.SPEED:
            assert screen.query("#quality-comparison")
        assert "Saved:" in str(screen.query_one("#quality-result", Static).content)
        # A second Start does not spend another benchmark or IFEval run.
        screen.query_one("#quality-tabs", TabbedContent).active = "quality-selection-tab"
        await pilot.pause()
        assert await pilot.click("#quality-start")
        await _wait_until(pilot, lambda: "Every candidate is already measured" in str(
            screen.query_one("#quality-status", Static).content,
        ))
        assert len(advisor.speed_runs) == (priority is not RecommendationPriority.QUALITY)
        assert len(advisor.quality_runs) == (priority is not RecommendationPriority.SPEED)
        assert advisor.selection_calls + len(advisor.speed_selections) == 1
        assert screen._search_state.model_dump_json() == before

    _run(advisor, priority, check)


def test_partial_measurement_refresh_keeps_failures_and_unchecked_candidates(monkeypatch) -> None:
    class Partial(SpeedAdvisor):
        def run_speed_candidates(self, candidates: Any, **kwargs: Any) -> Any:
            self.speed_runs.append(tuple(c.plan.plan_id for c in candidates))
            first, second = candidates
            self.candidates = tuple(replace(c, speed=SpeedAssessment(
                applicability="measured_plan", applicable=_speed(200.0),
            )) if c.plan.plan_id == first.plan.plan_id else c for c in self.candidates)
            return (
                QualityCandidateOutcome(first.plan, path=Path("/bench/completed.json")),
                QualityCandidateOutcome(second.plan, error="Synthetic measurement failure"),
            )

        def refresh_quality_candidates(self, selection: Any, requirements: Any, **kwargs: Any):
            return replace(selection, candidates=self.candidates)

    advisor = Partial()
    monkeypatch.setattr(quality_evaluation, "assess_speed", lambda *args:
                        SpeedAssessment(applicability="measured_plan", applicable=_speed(200.0)))

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        screen.query_one("#quality-candidate-2", Checkbox).value = False
        await pilot.pause()
        assert await pilot.click("#quality-start")
        await _wait_until(pilot, lambda: bool(advisor.speed_runs) and not screen._busy)
        assert [screen.query_one(f"#quality-candidate-{i}", Checkbox).value
                for i in range(3)] == [False, True, False]
        assert "Synthetic measurement failure" in str(
            screen.query_one("#quality-result", Static).content
        )
        assert "200.0 tok/s" in str(
            screen.query_one("#quality-speed-comparison", DataTable).get_row_at(0)[1]
        )
        assert advisor.selection_calls == 0 and len(advisor.speed_selections) == 1

    _run(advisor, RecommendationPriority.SPEED, check)


def test_a_refresh_failure_preserves_measurements_and_frees_the_screen(monkeypatch) -> None:
    class BrokenRefresh(SpeedAdvisor):
        def refresh_quality_candidates(self, *args: Any, **kwargs: Any):
            raise OSError("Synthetic store read failure")

    advisor = BrokenRefresh()
    monkeypatch.setattr(quality_evaluation, "assess_speed", lambda *args:
                        SpeedAssessment(applicability="measured_plan", applicable=_speed(200.0)))

    async def check(screen: QualityEvaluationScreen, pilot: Any) -> None:
        assert await pilot.click("#quality-start")
        await _wait_until(pilot, lambda: bool(advisor.speed_runs) and not screen._busy)
        text = str(screen.query_one("#quality-result", Static).content)
        assert "Saved:" in text and "Could not refresh candidate evidence" in text
        assert "Synthetic store read failure" in text
        assert screen.query("#quality-speed-comparison")
        assert not screen.query_one("#quality-start").disabled

    _run(advisor, RecommendationPriority.SPEED, check)


@pytest.mark.parametrize("case", ["overlap", "single", "separate"])
def test_speed_table_marks_only_a_separated_repeated_winner(case) -> None:
    first = _speed(230.0).model_copy(update={"observed_min": 220.0, "observed_max": 240.0})
    second = _speed(250.0).model_copy(update={
        "observed_min": 235.0 if case == "overlap" else 245.0, "observed_max": 260.0,
        "run_count": 1 if case == "single" else 3,
        "benchmark_ids": ("single",) if case == "single" else ("one", "two", "three"),
    })

    async def check(screen, pilot):
        table = quality_evaluation.speed_table((("A", first, "stored"), ("B", second, "stored")))
        await screen.mount(table)
        cells = table.get_row_at(0)[1:]
        assert ["bold" in str(cell.style) for cell in cells] == [False, case == "separate"]
        assert str(table.get_row_at(3)[2]) == str(second.run_count)
        assert "260.0" in str(table.get_row_at(4)[2])
        assert str(table.get_row_at(5)[2]) == f"{len(second.benchmark_ids)} records"

    _run(SpeedAdvisor(), RecommendationPriority.SPEED, check)


def test_finished_table_uses_the_refreshed_aggregate_not_the_last_run(monkeypatch) -> None:
    aggregate = _speed(237.3).model_copy(update={
        "run_count": 4, "benchmark_ids": ("old", "r1", "r2", "r3"),
        "observed_min": 218.6, "observed_max": 256.1,
    })

    class Aggregating(SpeedAdvisor):
        def refresh_quality_candidates(self, selection, requirements, **kwargs):
            return replace(selection, candidates=tuple(replace(candidate, speed=SpeedAssessment(
                applicability="measured_plan", applicable=aggregate,
            )) for candidate in selection.candidates))

    monkeypatch.setattr(quality_evaluation, "assess_speed", lambda *args:
                        SpeedAssessment(applicability="measured_plan", applicable=_speed(247.1)))

    async def check(screen, pilot):
        assert await pilot.click("#quality-start")
        await _wait_until(pilot, lambda: not screen._busy and bool(screen.query(
            "#quality-speed-comparison")))
        table = screen.query_one("#quality-speed-comparison", DataTable)
        assert str(table.get_row_at(0)[1]) == "237.3 tok/s"
        assert str(table.get_row_at(3)[1]) == "4"
        assert str(table.get_row_at(5)[1]) == "4 records"
        details = "\n".join(str(widget.content) for widget in screen.query(Static))
        assert "Benchmark IDs:\nold\nr1\nr2\nr3" in details

    _run(Aggregating(), RecommendationPriority.SPEED, check)
