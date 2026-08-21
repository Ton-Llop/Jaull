"""Compatibility shim for recommendation workflow budgets."""

from __future__ import annotations

from jaull.application.recommendation.policies import (
    MAX_ALTERNATIVES,
    MAX_CONCURRENT_INSPECTIONS,
    MAX_DEEP_INSPECTION,
    MAX_RECOMMENDATIONS,
    MAX_UNIQUE_CANDIDATES,
    MAX_VARIANT_DEEP_INSPECTION,
    SEARCH_RESULTS_PER_QUERY,
)

__all__ = [
    "MAX_ALTERNATIVES",
    "MAX_CONCURRENT_INSPECTIONS",
    "MAX_DEEP_INSPECTION",
    "MAX_RECOMMENDATIONS",
    "MAX_UNIQUE_CANDIDATES",
    "MAX_VARIANT_DEEP_INSPECTION",
    "SEARCH_RESULTS_PER_QUERY",
]
