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

# Upper bound of the "largest context that fits" search when the model does not
# declare its native context. Search bound, not a claim about any model.
SERVE_CONTEXT_SEARCH_CEILING_TOKENS = 1_048_576

# llama.cpp's prompt cache keeps idle conversations in system RAM, 8 GiB by default
# (--cache-ram). The server's own sizing never counted it, so Jaull sizes it at start:
# what is free, minus what the model itself puts in RAM, minus a share left to the
# rest of the PC, up to llama.cpp's default. HYPOTHESIS: the share is a product
# decision, not measured. Below the minimum the RAM cache is not worth having.
SERVE_PROMPT_CACHE_MAX_BYTES = 8 * 1024**3
SERVE_PROMPT_CACHE_MIN_BYTES = 512 * 1024**2
SERVE_RAM_LEFT_FOR_SYSTEM_FRACTION = 0.20
# The server monitor warns when free system RAM falls below this share of the total.
SERVE_RAM_LOW_FRACTION = 0.10

# A dedicated Intel GPU left idle for ~70 s had its VRAM copied into system RAM by Windows
# or the driver, which froze a PC that could not hold it (B60, docs/TFG/DECISIONS-2.md).
# Releasing the model after 45 s idle avoided it. Reported on a B50 and other Intel cards
# in forums, not tested here. The cause was never confirmed, so this is a mitigation.
# The cost is ~33 s on the first request after a pause (the model reloads).
# Measured on native Windows only. Under WSL2 with Dozen it slept after ~93 s and the PC
# lagged all the same, so it is not applied by default there.
SERVE_INTEL_SLEEP_IDLE_SECONDS = 45

# llama.cpp loads a GGUF with mmap, which also puts the whole file in system RAM although every
# layer ends on the GPU: on the B60 free RAM went from 13 to 0.1 GB in 10 s. Reading it with
# direct I/O (-lm dio) kept the server at ~0.4 GB resident in 8 runs. Measured on native
# Windows only (docs/TFG/DECISIONS-2.md), so applied there, for a model fully on the GPU.
# What it does to generation speed was not compared.
SERVE_DIRECT_IO_LOAD_MODE = "dio"
# What the person types to get llama.cpp's own default back.
SERVE_LOAD_MODE_DEFAULT = "default"

# The opencode configuration written at start. With a shared pool, a client given the whole
# pool can fill it alone and end every other chat (measured, docs/TFG/concurrencia-b60.md),
# so it gets this share of the pool. HEURISTIC: half leaves room for a few web chats while an
# agent keeps a long context; not measured with real use.
SERVE_CLIENT_POOL_SHARE = 0.5
# Largest answer the client asks for in one turn, and at most this share of its context so
# the prompt keeps the rest. Product decisions, not measurements.
SERVE_CLIENT_OUTPUT_TOKENS = 8192
SERVE_CLIENT_OUTPUT_MAX_SHARE = 0.25

# The API key shown as a QR hides itself after this long, in case it is left open.
# Product decision: long enough to point a phone at it, short enough to be forgotten.
SERVE_KEY_QR_VISIBLE_SECONDS = 30.0

# Context sizes are planned in steps of this many tokens.
SERVE_CONTEXT_STEP_TOKENS = 4096

# Defaults for ``jaull serve start``: product decisions, not measurements.
SERVE_DEFAULT_USERS = 1
SERVE_DEFAULT_CONTEXT_PER_USER = 32768
