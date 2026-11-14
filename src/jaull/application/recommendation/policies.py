"""Recommendation workflow budgets and output limits."""

from __future__ import annotations

from jaull.domain.requirements import RecommendationPriority

SEARCH_RESULTS_PER_QUERY = 20
# The pool the shortlist chooses the inspected repositories from. Search results are already
# downloaded, so a larger pool costs nothing on the network; it only has to be wide enough that
# the queries (12 on a large machine, each cut at its own top ranks) are not trimmed to their best
# few hits, which dropped Qwen's own GGUF repositories.
MAX_UNIQUE_CANDIDATES = 100
MAX_DEEP_INSPECTION = 12
# Shortlist places kept for models that fit only at the lowest quantization of the priority's
# ladder (a 35B on 24 GiB). They come out of the exploration budget, not out of relevance.
# Balanced is where the default-quantization ceiling hides them; Quality already explores large
# models and Speed/Memory have no need to go past what fits comfortably.
REACH_EXPLORATION_SLOTS: dict[RecommendationPriority, int] = {RecommendationPriority.BALANCED: 1}
MAX_CONCURRENT_INSPECTIONS = 4
# Ranged GGUF header reads are small and independent; they only wait on the network.
MAX_CONCURRENT_HEADER_READS = 8
# Repository lookups for the task a re-quantization inherits from its declared base.
MAX_CONCURRENT_LINEAGE_READS = 8
MAX_VARIANT_DEEP_INSPECTION = 6
# One extra round of inspections for shortlist slots lost to failed or drafter
# candidates, so they are replaced without turning into an unbounded search.
MAX_REPLACEMENT_INSPECTIONS = 6
MAX_TARGETED_VARIANT_SEARCHES = 3

MAX_RECOMMENDATIONS = 5
MAX_ALTERNATIVES = MAX_RECOMMENDATIONS - 1


__all__ = [
    "MAX_ALTERNATIVES",
    "MAX_CONCURRENT_HEADER_READS",
    "MAX_CONCURRENT_INSPECTIONS",
    "MAX_CONCURRENT_LINEAGE_READS",
    "MAX_DEEP_INSPECTION",
    "MAX_RECOMMENDATIONS",
    "MAX_REPLACEMENT_INSPECTIONS",
    "MAX_TARGETED_VARIANT_SEARCHES",
    "MAX_UNIQUE_CANDIDATES",
    "MAX_VARIANT_DEEP_INSPECTION",
    "REACH_EXPLORATION_SLOTS",
    "SEARCH_RESULTS_PER_QUERY",
]
