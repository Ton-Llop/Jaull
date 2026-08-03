"""Domain-level constants shared between discovery, recommendation and workflow.

Anything about *how the model behaves* or *how the user's answers map to a
technical requirement* lives here. Things about *how the guided workflow
budgets its own work* (search page size, deep-inspection cap) stay in
``workflow/policies.py`` because they are application concerns.
"""

from __future__ import annotations

from jaull.domain.estimation import CompatibilityStatus

# --------------------------------------------------------------------------
# The only pipeline this commit supports.
# --------------------------------------------------------------------------
TEXT_GENERATION_PIPELINE = "text-generation"

# --------------------------------------------------------------------------
# Context lengths offered by the document wizard question, in tokens.
#
# These describe the *model* context window, which is not the size of a
# document collection: a RAG system retrieves a handful of chunks that must fit
# in this window, it does not load the corpus. The wizard copy says so and
# ``workflow/requirements.py`` records the same caveat as an assumption.
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
# Compatibility ordering. Lower rank is better. `insufficient` can never be a
# primary recommendation and `unknown` only ever appears as a flagged
# alternative — both rules are enforced in the ranker, not here.
# --------------------------------------------------------------------------
STATUS_RANK: dict[CompatibilityStatus, int] = {
    CompatibilityStatus.COMFORTABLE: 0,
    CompatibilityStatus.COMPATIBLE: 1,
    CompatibilityStatus.TIGHT: 2,
    CompatibilityStatus.OFFLOADING_REQUIRED: 3,
    CompatibilityStatus.UNKNOWN: 4,
    CompatibilityStatus.INSUFFICIENT: 5,
}

# How each status maps onto the hardware-fit sub-score.
STATUS_FIT_SCORE: dict[CompatibilityStatus, float] = {
    CompatibilityStatus.COMFORTABLE: 1.0,
    CompatibilityStatus.COMPATIBLE: 0.85,
    CompatibilityStatus.TIGHT: 0.55,
    CompatibilityStatus.OFFLOADING_REQUIRED: 0.30,
    CompatibilityStatus.UNKNOWN: 0.20,
    CompatibilityStatus.INSUFFICIENT: 0.0,
}


__all__ = [
    "CONCURRENCY_BUCKETS",
    "CONCURRENCY_HEADROOM_PER_USER",
    "DEFAULT_CONTEXT_TOKENS",
    "DOCUMENT_CONTEXT_TOKENS",
    "MAX_CONCURRENCY_HEADROOM",
    "STATUS_FIT_SCORE",
    "STATUS_RANK",
    "TEXT_GENERATION_PIPELINE",
]
