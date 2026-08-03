"""Downgrade the "best match" label when the evidence is thin.

A recommendation card that always screams *BEST MATCH* is dishonest when the
top result is a low-confidence estimate for a model that only just fits with
offloading. This module maps the ``(status, confidence, unmet)`` signal onto
a small tier vocabulary so the card can pick an appropriate heading.
"""

from __future__ import annotations

from enum import StrEnum

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)


class RecommendationTier(StrEnum):
    BEST_MATCH = "best_match"
    RECOMMENDED = "recommended"
    CLOSEST_OPTION = "closest_option"
    BEST_EFFORT = "best_effort"


TIER_HEADINGS: dict[RecommendationTier, str] = {
    RecommendationTier.BEST_MATCH: "BEST MATCH",
    RecommendationTier.RECOMMENDED: "RECOMMENDED",
    RecommendationTier.CLOSEST_OPTION: "CLOSEST OPTION",
    RecommendationTier.BEST_EFFORT: "BEST-EFFORT SUGGESTION",
}


def choose_tier(
    status: CompatibilityStatus,
    confidence: EstimationConfidence,
    hard_penalty: float,
) -> RecommendationTier:
    """Pick a tier from the compatibility, confidence and hard-penalty signals.

    Ordered from strongest to weakest downgrade:

    1. Any hard requirement missed → ``BEST_EFFORT``.
    2. Low or unknown confidence → ``BEST_EFFORT``.
    3. Offloading required or unknown status → ``CLOSEST_OPTION``.
    4. Tight fit or only medium confidence → ``RECOMMENDED``.
    5. Comfortable/compatible + high confidence → ``BEST_MATCH``.
    """
    if hard_penalty < 1.0:
        return RecommendationTier.BEST_EFFORT
    if confidence in (EstimationConfidence.LOW, EstimationConfidence.UNKNOWN):
        return RecommendationTier.BEST_EFFORT
    if status in (
        CompatibilityStatus.OFFLOADING_REQUIRED,
        CompatibilityStatus.UNKNOWN,
    ):
        return RecommendationTier.CLOSEST_OPTION
    if status is CompatibilityStatus.TIGHT:
        return RecommendationTier.RECOMMENDED
    if confidence is EstimationConfidence.MEDIUM:
        return RecommendationTier.RECOMMENDED
    return RecommendationTier.BEST_MATCH


__all__ = ["TIER_HEADINGS", "RecommendationTier", "choose_tier"]
