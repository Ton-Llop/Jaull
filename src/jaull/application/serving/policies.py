"""How cautious a long-running model server is planned. Product policy, not physics.

The estimator says how many bytes a server configuration allocates; these values
decide how much of the machine a server may take. A server shares the PC with
other work for hours, so it gets a margin on top of the estimator's own policy.
"""

from __future__ import annotations

from jaull.domain.estimation import CompatibilityStatus

# Fraction of the accelerator memory the scan reports that a server may plan to
# allocate. HYPOTHESIS from two points on one machine (Intel Arc Pro B60, 24 GiB,
# Qwen3.6-35B-A3B; docs/TFG/concurrencia-b60.md): about 22.1 GiB (0.92) hung the
# shared PC under load, about 20.8 GiB (0.87) ran. Revisit with more evidence.
SERVE_GPU_MAX_USAGE_FRACTION = 0.875

# A CPU-only estimate carries no hardware-fit placement, only the assessment.
# A server runs for hours next to other work, so TIGHT and worse are refused.
SERVE_CPU_ACCEPTED_STATUSES = frozenset({
    CompatibilityStatus.COMFORTABLE,
    CompatibilityStatus.COMPATIBLE,
})

# Context sizes are planned in steps of this many tokens.
SERVE_CONTEXT_STEP_TOKENS = 4096

# Defaults for ``jaull serve start``: product decisions, not measurements.
SERVE_DEFAULT_USERS = 1
SERVE_DEFAULT_CONTEXT_PER_USER = 32768
