"""Step 6 of the Fastest/Balanced plan: the measured policies order the active list.

Synthetic applicable evidence on real assessed plans; it tests the rule, not models.
"""

from __future__ import annotations

from typing import Any

import pytest

from jaull.domain.recommendation import ApplicableQuality, ApplicableSpeed
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.shadow import apply_measured_policy
from tests.test_shadow_policies import _pool

COHORT = "c" * 64


def _measured(item: Any, *, quality: float | None = None, speed: float | None = None) -> Any:
    """The plan with applicable evidence attached exactly where assess_plan puts it."""
    assessment = item.assessment
    if quality is not None:
        assessment = assessment.model_copy(update={"quality": assessment.quality.model_copy(
            update={"applicable": ApplicableQuality(
                suite="jaull-ifeval-chat-v1", metric="prompt_level_strict_acc", value=quality,
                correct=int(quality * 100), samples=100, cohort=COHORT, record_sha256="d" * 64,
            )},
        )})
    if speed is not None:
        assessment = assessment.model_copy(update={"speed": assessment.speed.model_copy(
            update={"applicability": "measured_plan", "applicable": ApplicableSpeed(
                tokens_per_second=speed, repetitions=5, cohort=COHORT,
                benchmark_id=f"bench-{speed}",
            )},
        )})
    return type(item)(item.evaluated, item.plan, assessment)


def _order(items: list[Any], requirements: Any) -> list[str]:
    return [item.evaluated.repo_id.split("/")[1][0]
            for item in apply_measured_policy(items, requirements)]


FASTEST = RecommendationPriority.SPEED
BALANCED = RecommendationPriority.BALANCED


def test_fastest_puts_the_faster_comparable_plan_first() -> None:
    requirements, (a, b) = _pool("A", "B", priority=FASTEST)
    assert _order([_measured(a, speed=60), _measured(b, speed=95)], requirements) == ["B", "A"]


def test_fastest_without_evidence_or_with_one_measurement_keeps_the_base() -> None:
    requirements, (a, b) = _pool("A", "B", priority=FASTEST)
    assert _order([a, b], requirements) == ["A", "B"]
    assert _order([a, _measured(b, speed=95)], requirements) == ["A", "B"]  # Isolated.


def test_balanced_moves_a_dominating_plan_up() -> None:
    requirements, (a, b) = _pool("A", "B", priority=BALANCED)
    items = [_measured(a, quality=0.4, speed=60), _measured(b, quality=0.5, speed=80)]
    assert _order(items, requirements) == ["B", "A"]  # B is better on both axes.


def test_balanced_keeps_a_real_trade_off_in_base_order() -> None:
    # The RTX 2060 case: Qwen 0.5B faster (272 tok/s), Qwen 1.5B better (41 %).
    requirements, (a, b) = _pool("A", "B", priority=BALANCED)
    items = [_measured(a, quality=0.26, speed=272), _measured(b, quality=0.41, speed=164)]
    assert _order(items, requirements) == ["A", "B"]  # Same Pareto front.


def test_balanced_needs_both_axes() -> None:
    requirements, (a, b) = _pool("A", "B", priority=BALANCED)
    only_quality = [_measured(a, quality=0.4), _measured(b, quality=0.9)]
    only_speed = [_measured(a, speed=60), _measured(b, speed=95)]
    assert _order(only_quality, requirements) == ["A", "B"]
    assert _order(only_speed, requirements) == ["A", "B"]


def test_memory_has_no_measured_policy() -> None:
    requirements, (a, b) = _pool("A", "B", priority=RecommendationPriority.MEMORY)
    items = [_measured(a, quality=0.4, speed=60), _measured(b, quality=0.9, speed=95)]
    assert apply_measured_policy(items, requirements) == items


# --- From the whole engine: a lone applicable speed never reaches the base order ---


@pytest.mark.parametrize("priority", [
    RecommendationPriority.SPEED, RecommendationPriority.BALANCED, RecommendationPriority.MEMORY,
])
def test_one_valid_speed_measurement_alone_changes_nothing(monkeypatch, priority) -> None:
    """Codex's repro: without benchmarks A->B; one valid measurement of B must keep A->B.

    recommend() runs for real: base key, diversity and the measured policy. Only the
    evidence source is replaced, so plan B carries one applicable speed and A none.
    """
    from jaull.application.recommendation.service import recommend
    from jaull.domain.recommendation import SpeedAssessment
    from jaull.recommendation import engine_v2
    from jaull.recommendation.engine_v2 import PlanRankingContext
    from tests._workflow_fixtures import hardware
    from tests.test_ifeval_applicability import _gguf
    from tests.test_recommendation_engine_v2 import _requirements

    requirements = _requirements(priority)
    candidates = [_gguf("org/A-7B-GGUF", "a" * 64), _gguf("org/B-7B-GGUF", "9" * 64)]

    def order() -> list[str]:
        shown = recommend(candidates, requirements, hardware=hardware(),
                          plan_context=PlanRankingContext(hardware=hardware()))
        return [rec.repo_id for rec in shown]

    before = order()
    real = engine_v2.assess_speed

    def b_is_measured(plan, requirements, records):
        if plan.artifact.sha256 != "9" * 64:
            return real(plan, requirements, records)
        return SpeedAssessment(applicability="measured_plan", applicable=ApplicableSpeed(
            tokens_per_second=500.0, repetitions=5, cohort=COHORT, benchmark_id="bench-b",
        ))

    monkeypatch.setattr(engine_v2, "assess_speed", b_is_measured)
    assert order() == before  # An isolated measurement proves no move.


def test_recompare_refreshes_speed_from_the_benchmarks_it_is_given(monkeypatch) -> None:
    from jaull.recommendation import shadow

    requirements, (a, b) = _pool("A", "B", priority=FASTEST)
    figures = {a.plan.plan_id: 60.0, b.plan.plan_id: 95.0}
    monkeypatch.setattr(shadow, "assess_speed", lambda plan, req, records: _measured(
        next(item for item in (a, b) if item.plan.plan_id == plan.plan_id),
        speed=figures[plan.plan_id],
    ).assessment.speed)
    # Search saw no speed; only the refreshed benchmarks can move anything.
    assert not shadow.recompare_quality([a, b], requirements, [], limit=5).moves
    assert shadow.recompare_quality(
        [a, b], requirements, [], limit=5, benchmark_records=[],
    ).moves
