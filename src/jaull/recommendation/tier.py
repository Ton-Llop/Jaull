"""Downgrade the "best match" label when the evidence is thin.

A recommendation card that always screams *BEST MATCH* is dishonest when the
top result is a low-confidence estimate for a model that only just fits with
offloading — or, worse, a theoretical int4 configuration on hardware that
cannot actually run it. This module maps the
``(status, confidence, actionability, unmet)`` signal onto a small tier
vocabulary so the card can pick an honest heading.
"""

from __future__ import annotations

from enum import StrEnum

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from jaull.recommendation.actionability import ActionabilityLevel


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
    actionability: ActionabilityLevel = ActionabilityLevel.ACTIONABLE,
) -> RecommendationTier:
    """Pick a tier from the compatibility, confidence, actionability and
    hard-penalty signals.

    Ordered from strongest to weakest downgrade:

    1. Any hard requirement missed → ``BEST_EFFORT``.
    2. Low or unknown confidence → ``BEST_EFFORT``.
    3. Speculative actionability (theoretical artifact / unknown runtime) →
       at most ``BEST_EFFORT``.
    4. Offloading required or unknown status → ``CLOSEST_OPTION``.
    5. Tight fit or only medium confidence → ``RECOMMENDED``.
    6. Likely-actionable but not confirmed actionable → at most
       ``RECOMMENDED``.
    7. Everything else with high confidence → ``BEST_MATCH``.
    """
    if hard_penalty < 1.0:
        return RecommendationTier.BEST_EFFORT
    if confidence in (EstimationConfidence.LOW, EstimationConfidence.UNKNOWN):
        return RecommendationTier.BEST_EFFORT
    if actionability is ActionabilityLevel.SPECULATIVE:
        # Ranker still ranks it, but it can never earn a BEST/RECOMMENDED
        # heading because the artifact/runtime path is not confirmed.
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
    if actionability is ActionabilityLevel.LIKELY_ACTIONABLE:
        # Runtime is *potentially* executable — good, but not "best-match" good.
        return RecommendationTier.RECOMMENDED
    return RecommendationTier.BEST_MATCH


__all__ = ["TIER_HEADINGS", "RecommendationTier", "choose_tier"]
