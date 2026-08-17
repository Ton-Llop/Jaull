"""Guided-run budgets: how much work the orchestrator is willing to do.

Domain-facing constants (pipeline tag, context lengths, concurrency buckets,
compatibility ranks) live in ``jaull.domain.policies``. This module owns only
the numbers that decide *how big a run gets*, which is application concern.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Search budget.
#
# A guided run must stay reasonable on a home connection. Each query is a
# single paginated API call; the deep-inspection stage is the expensive one
# because every candidate there costs a model_info call plus (potentially) a
# config.json download, a safetensors header read and a GGUF Range read.
# --------------------------------------------------------------------------
SEARCH_RESULTS_PER_QUERY = 20
MAX_UNIQUE_CANDIDATES = 40
MAX_DEEP_INSPECTION = 12
MAX_CONCURRENT_INSPECTIONS = 4
MAX_VARIANT_DEEP_INSPECTION = 6

# --------------------------------------------------------------------------
# Output budget.
# --------------------------------------------------------------------------
MAX_RECOMMENDATIONS = 5
MAX_ALTERNATIVES = MAX_RECOMMENDATIONS - 1


__all__ = [
    "MAX_ALTERNATIVES",
    "MAX_CONCURRENT_INSPECTIONS",
    "MAX_DEEP_INSPECTION",
    "MAX_RECOMMENDATIONS",
    "MAX_UNIQUE_CANDIDATES",
    "MAX_VARIANT_DEEP_INSPECTION",
    "SEARCH_RESULTS_PER_QUERY",
]
