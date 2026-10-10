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

from textual.widgets import Checkbox, DataTable, Select, Static

from jaull.advisor.quality_candidates import QualityCandidateOutcome, QualityCandidateSelection
from jaull.domain.recommendation import ApplicableSpeed, SpeedAssessment
from jaull.domain.requirements import RecommendationPriority
from jaull.runtime.quality_eval_runner import QualityProfile
from jaull.tui.app import JaullApp
from jaull.tui.screens import quality_evaluation
from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen
from tests.test_tui_ifeval import ProfileAdvisor, _state
from tests.test_tui_quality import _fill, _request, _settled

COHORT = "c" * 64


def _speed(value: float) -> ApplicableSpeed:
    return ApplicableSpeed(tokens_per_second=value, repetitions=5, cohort=COHORT,
                           benchmark_id=f"bench-{value}")


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
