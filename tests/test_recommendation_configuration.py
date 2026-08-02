from __future__ import annotations

import pytest

from local_ai_check.domain.inference import WeightPrecision
from local_ai_check.recommendation.configuration import select_configuration
from local_ai_check.workflow.models import RecommendationPriority
from local_ai_check.workflow.requirements import build_requirements
from tests._workflow_fixtures import (
    GIB,
    answers,
    gguf_analysis,
    hardware,
    size_driven_estimator,
    transformers_analysis,
)


def _req(priority: RecommendationPriority) -> object:
    return build_requirements(answers(priority=priority), hardware())


@pytest.mark.parametrize(
    ("priority", "expected"),
    [
        # Sizes are Q3_K_M=4, Q4_K_M=5, Q5_K_M=6, Q6_K=7 GiB against a 24 GiB
        # budget, so every rung fits and the ladder's first entry wins.
        (RecommendationPriority.QUALITY, "Q6_K"),
        (RecommendationPriority.BALANCED, "Q5_K_M"),
        (RecommendationPriority.SPEED, "Q4_K_M"),
        (RecommendationPriority.MEMORY, "Q3_K_M"),
    ],
)
def test_ladder_head_is_chosen_when_everything_fits(
    priority: RecommendationPriority, expected: str
) -> None:
    choice = select_configuration(
        gguf_analysis(),
        _req(priority),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=24 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.quantization == expected


def test_memory_priority_prefers_the_smallest_available_rung() -> None:
    choice = select_configuration(
        gguf_analysis(quantizations=("Q4_K_S", "Q4_K_M", "Q6_K")),
        _req(RecommendationPriority.MEMORY),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=24 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.quantization == "Q4_K_S"


def test_ladder_falls_through_when_the_preferred_rung_is_absent() -> None:
    """Q4_K_M missing must not break the balanced ladder."""
    choice = select_configuration(
        gguf_analysis(quantizations=("Q3_K_M", "Q5_K_M")),
        _req(RecommendationPriority.SPEED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=24 * GIB),
    )
    assert choice.configuration is not None
    # Speed ladder is Q4_K_M -> Q4_K_S -> Q3_K_M -> Q3_K_S; only Q3_K_M exists.
    assert choice.configuration.quantization == "Q3_K_M"


def test_smaller_rung_is_chosen_when_the_larger_one_does_not_fit() -> None:
    choice = select_configuration(
        gguf_analysis(),  # 4,5,6,7 GiB
        _req(RecommendationPriority.QUALITY),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=6 * GIB),
    )
    assert choice.configuration is not None
    # Q6_K (7 GiB) and Q5_K_M (6 GiB, tight at exactly the budget) are tried first.
    assert choice.configuration.quantization in {"Q5_K_M", "Q4_K_M"}


def test_no_compatible_quantization_reports_the_closest_option() -> None:
    choice = select_configuration(
        gguf_analysis(base_bytes=40 * GIB),
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=8 * GIB),
    )
    assert any("No GGUF variant fits" in w for w in choice.warnings)
    # Still returns something so the UI can explain how far off it was.
    assert choice.configuration is not None


def test_aggressive_quantization_is_flagged_when_it_is_the_only_fit() -> None:
    choice = select_configuration(
        gguf_analysis(quantizations=("Q2_K", "Q6_K"), base_bytes=2 * GIB),
        _req(RecommendationPriority.MEMORY),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=3 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.quantization == "Q2_K"
    assert any("aggressive quantization" in w for w in choice.warnings)


def test_gguf_repository_without_variants_degrades_cleanly() -> None:
    analysis = gguf_analysis(quantizations=())
    choice = select_configuration(
        analysis,
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(),
    )
    assert choice.configuration is None
    assert choice.estimate is None
    assert choice.warnings


def test_considered_rungs_are_recorded_for_the_report() -> None:
    choice = select_configuration(
        gguf_analysis(),
        _req(RecommendationPriority.MEMORY),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=24 * GIB),
    )
    assert choice.considered
    assert all(":" in entry for entry in choice.considered)


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------
def test_transformers_uses_float16_when_it_fits() -> None:
    choice = select_configuration(
        transformers_analysis(),  # 14 GiB
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=24 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.precision is WeightPrecision.FLOAT16
    assert choice.warnings == []


def test_transformers_falls_back_to_int8_and_warns_it_is_theoretical() -> None:
    choice = select_configuration(
        transformers_analysis(),  # float16 14 GiB, int8 7 GiB
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=10 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.precision is WeightPrecision.INT8
    assert any("theoretical estimate" in w for w in choice.warnings)


def test_transformers_int4_lowers_the_reported_confidence() -> None:
    choice = select_configuration(
        transformers_analysis(),  # int4 is 3.5 GiB
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=5 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.precision is WeightPrecision.INT4
    assert choice.estimate is not None
    # HIGH in the fixture, dropped one rung because nothing confirms the artifact.
    assert choice.estimate.assessment.confidence.value != "high"


def test_transformers_with_nothing_fitting_reports_the_closest() -> None:
    choice = select_configuration(
        transformers_analysis(),
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=1 * GIB),
    )
    assert any("No precision fits" in w for w in choice.warnings)


def test_guided_mode_never_asks_the_user_for_a_quantization() -> None:
    """The chosen config is fully derived: the user supplied no technical input."""
    choice = select_configuration(
        gguf_analysis(),
        _req(RecommendationPriority.BALANCED),  # type: ignore[arg-type]
        size_driven_estimator(vram_budget=24 * GIB),
    )
    assert choice.configuration is not None
    assert choice.configuration.batch_size == 1
    assert choice.reason
