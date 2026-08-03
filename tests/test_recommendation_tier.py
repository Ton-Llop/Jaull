from __future__ import annotations

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from jaull.recommendation.tier import RecommendationTier, choose_tier


def test_high_confidence_comfortable_is_best_match() -> None:
    result = choose_tier(
        CompatibilityStatus.COMFORTABLE,
        EstimationConfidence.HIGH,
        hard_penalty=1.0,
    )
    assert result is RecommendationTier.BEST_MATCH


def test_medium_confidence_downgrades_to_recommended() -> None:
    result = choose_tier(
        CompatibilityStatus.COMPATIBLE,
        EstimationConfidence.MEDIUM,
        hard_penalty=1.0,
    )
    assert result is RecommendationTier.RECOMMENDED


def test_offloading_becomes_closest_option_even_with_high_confidence() -> None:
    result = choose_tier(
        CompatibilityStatus.OFFLOADING_REQUIRED,
        EstimationConfidence.HIGH,
        hard_penalty=1.0,
    )
    assert result is RecommendationTier.CLOSEST_OPTION


def test_low_confidence_always_best_effort() -> None:
    result = choose_tier(
        CompatibilityStatus.COMFORTABLE,
        EstimationConfidence.LOW,
        hard_penalty=1.0,
    )
    assert result is RecommendationTier.BEST_EFFORT


def test_unmet_hard_requirement_forces_best_effort() -> None:
    result = choose_tier(
        CompatibilityStatus.COMFORTABLE,
        EstimationConfidence.HIGH,
        hard_penalty=0.6,
    )
    assert result is RecommendationTier.BEST_EFFORT
