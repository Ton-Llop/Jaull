"""Centralised, documented limits for the guided recommendation workflow.

Every bound the workflow enforces lives here so the cost of a guided run is
auditable in one place instead of being spread across the search, filtering and
evaluation stages.
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

# --------------------------------------------------------------------------
# Output budget.
# --------------------------------------------------------------------------
MAX_RECOMMENDATIONS = 3
MAX_ALTERNATIVES = MAX_RECOMMENDATIONS - 1

# --------------------------------------------------------------------------
# Context lengths offered by the document wizard question, in tokens.
#
# These describe the *model* context window, which is not the size of a
# document collection: a RAG system retrieves a handful of chunks that must fit
# in this window, it does not load the corpus. The wizard copy says so and
# `requirements.py` records the same caveat as an assumption.
# --------------------------------------------------------------------------
DOCUMENT_CONTEXT_TOKENS: dict[str, int] = {
    "short": 4096,
    "medium": 8192,
    "long": 16384,
    "collection": 32768,
}

# Context used when the use case is not document-oriented.
DEFAULT_CONTEXT_TOKENS = 4096

# --------------------------------------------------------------------------
# Concurrency buckets. The integer is a representative value used as a ranking
# signal and a KV-cache headroom multiplier; the label is kept alongside it so
# reports never imply a precision the question did not capture.
# --------------------------------------------------------------------------
CONCURRENCY_BUCKETS: dict[str, tuple[int, str]] = {
    "single": (1, "One user"),
    "small": (3, "2-5 users"),
    "medium": (10, "6-20 users"),
    "large": (25, "More than 20 users"),
}

# How much extra KV-cache headroom to require per concurrent user beyond the
# first. Deliberately conservative: there is no real concurrency model yet, so
# this only nudges ranking away from models that barely fit.
CONCURRENCY_HEADROOM_PER_USER = 0.15

# Cap the multiplier so a "more than 20 users" answer cannot make every model
# look impossible; beyond this point the honest answer is a warning, not a number.
MAX_CONCURRENCY_HEADROOM = 2.0

# --------------------------------------------------------------------------
# The only pipeline this commit supports.
# --------------------------------------------------------------------------
TEXT_GENERATION_PIPELINE = "text-generation"

__all__ = [
    "CONCURRENCY_BUCKETS",
    "CONCURRENCY_HEADROOM_PER_USER",
    "DEFAULT_CONTEXT_TOKENS",
    "DOCUMENT_CONTEXT_TOKENS",
    "MAX_ALTERNATIVES",
    "MAX_CONCURRENCY_HEADROOM",
    "MAX_DEEP_INSPECTION",
    "MAX_RECOMMENDATIONS",
    "MAX_UNIQUE_CANDIDATES",
    "SEARCH_RESULTS_PER_QUERY",
    "TEXT_GENERATION_PIPELINE",
]
