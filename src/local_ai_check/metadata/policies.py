"""Documented constants for the GGUF enrichment pipeline."""

from __future__ import annotations

KiB = 1024
MiB = 1024 * 1024

# --------------------------------------------------------------------------
# HTTP Range strategy for fetching GGUF headers without downloading weights.
# --------------------------------------------------------------------------
INITIAL_HEADER_RANGE_BYTES = 256 * KiB
RANGE_GROWTH_FACTOR = 2
MAX_HEADER_DOWNLOAD_BYTES = 8 * MiB
HTTP_TIMEOUT_SECONDS = 10.0

# --------------------------------------------------------------------------
# Base model resolver hint prefixes: repository name suffixes that *might*
# indicate a GGUF conversion of another repo. Never used as authoritative
# resolution; only surfaced as evidence when nothing else is available.
# --------------------------------------------------------------------------
GGUF_SUFFIX_HINTS = ("-GGUF", "-gguf", ".gguf")
