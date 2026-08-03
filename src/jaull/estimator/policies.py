"""Centralised, documented constants for the memory estimator.

Every "magic number" the estimator uses lives here. Modules import these instead
of hard-coding values so the assumptions are auditable and easy to tune.
"""

from __future__ import annotations

from jaull.domain.inference import WeightPrecision

MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024

# --------------------------------------------------------------------------
# Weight dtype -> bytes per parameter.
#
# These are the theoretical footprints. Real quantized formats add scales,
# zero-points, block metadata, etc. that this table does NOT model — those
# extras are typically well under 10 % of the weight bytes for GGUF/AWQ/GPTQ,
# but the estimator flags the result as DERIVED (not EXACT) when it uses this
# table instead of an actual file size.
# --------------------------------------------------------------------------
BYTES_PER_PARAM: dict[WeightPrecision, float] = {
    WeightPrecision.FLOAT32: 4.0,
    WeightPrecision.FLOAT16: 2.0,
    WeightPrecision.BFLOAT16: 2.0,
    WeightPrecision.INT8: 1.0,
    WeightPrecision.INT4: 0.5,
}


def bytes_per_parameter(precision: WeightPrecision) -> float:
    return BYTES_PER_PARAM[precision]


# --------------------------------------------------------------------------
# KV cache dtype default. Modern runtimes overwhelmingly keep KV in fp16 even
# when weights are int4/int8. The user can override with --kv-dtype.
# --------------------------------------------------------------------------
KV_DTYPE_DEFAULT = WeightPrecision.FLOAT16

# --------------------------------------------------------------------------
# Runtime overhead heuristic. Meant to cover: allocator arenas, kernel
# workspaces, activation buffers, tokenizer/state tensors, small caches.
# Overhead = max(min, base + fraction * weights).
# --------------------------------------------------------------------------
OVERHEAD_BASE_BYTES = 512 * MiB
OVERHEAD_WEIGHT_FRACTION = 0.10
OVERHEAD_MIN_BYTES = 256 * MiB

# --------------------------------------------------------------------------
# Device reserve: memory we tell the user to LEAVE unused on the device
# (display server, other processes, driver working set). Applied only for
# GPU/AUTO with a GPU present. The CLI exposes this as --device-reserve-gib.
# --------------------------------------------------------------------------
DEVICE_RESERVE_DEFAULT_BYTES = 512 * MiB

# --------------------------------------------------------------------------
# Safety margin percentage applied on top of the sum of everything else.
# Exposed via --safety-margin-percent (0 to disable).
# --------------------------------------------------------------------------
SAFETY_MARGIN_DEFAULT_PERCENT = 10.0

# --------------------------------------------------------------------------
# Compatibility ratio thresholds (fraction of available memory).
# --------------------------------------------------------------------------
COMFORTABLE_MAX = 0.75
COMPATIBLE_MAX = 0.90
TIGHT_MAX = 1.00

# --------------------------------------------------------------------------
# Default GGUF quantization when the user does not specify one. Q4_K_M is
# the quality/size sweet spot most repositories publish; if it is not
# present, gguf_selection falls back to the median-sized variant.
# --------------------------------------------------------------------------
DEFAULT_GGUF_QUANTIZATION = "Q4_K_M"
