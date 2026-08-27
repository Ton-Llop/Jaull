"""Compatibility import for artifact quantization semantics."""

from __future__ import annotations

from jaull.domain.artifact_profile import (
    is_packed_transformers_quantization_config,
    packed_transformers_quantization_bits,
    packed_transformers_quantization_label,
)

__all__ = [
    "is_packed_transformers_quantization_config",
    "packed_transformers_quantization_bits",
    "packed_transformers_quantization_label",
]
