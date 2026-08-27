"""Confirmed vs theoretical artifact information for a candidate.

Discovery already classifies a repository as ``GGUF``/``TRANSFORMERS``/…, but
that says nothing about *which* pre-quantized artifact (if any) is available.
An AWQ repo with real weight files is a different beast from a bare
Transformers repo that the recommender chose to *estimate* at int4: both live
under ``RepositoryType.TRANSFORMERS`` but only the first is safe to promote
as a top pick on hardware that supports it. This module exposes that
distinction as first-class data.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.model import ModelConfig


class ArtifactFormat(StrEnum):
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
    BITSANDBYTES = "bitsandbytes"
    COMPRESSED_TENSORS = "compressed_tensors"
    NATIVE = "native"
    UNKNOWN = "unknown"


class ArtifactConfirmation(StrEnum):
    """How much evidence backs the claim that this artifact is real."""

    # File(s) or explicit metadata prove the artifact exists in the repo.
    CONFIRMED = "confirmed"
    # Strong secondary signal (tag, id suffix) suggests the artifact exists.
    INFERRED = "inferred"
    # The recommender picked a precision the repo does not publish.
    THEORETICAL = "theoretical"
    UNKNOWN = "unknown"


class ArtifactProfile(BaseModel):
    """A structured description of the recommended artifact."""

    model_config = ConfigDict(frozen=True)

    format: ArtifactFormat
    quantization: str | None = None
    confirmation: ArtifactConfirmation = ArtifactConfirmation.UNKNOWN
    reasons: list[str] = Field(default_factory=list)


_PACKED_QUANTIZATION_LABELS = {
    "awq": "AWQ",
    "gptq": "GPTQ",
    "compressed-tensors": "Compressed Tensors",
    "compressed_tensors": "Compressed Tensors",
}


def is_packed_transformers_quantization_config(config: ModelConfig | None) -> bool:
    """Return whether config metadata describes packed Transformers weights."""

    return packed_transformers_quantization_label(config) is not None


def packed_transformers_quantization_label(config: ModelConfig | None) -> str | None:
    """Return the artifact label for config-declared packed Transformers weights."""

    if config is None or not isinstance(config.quantization_config, dict):
        return None

    qcfg = config.quantization_config
    method_raw = qcfg.get("quant_method") or qcfg.get("quantization_method")
    method = str(method_raw).strip().casefold() if method_raw is not None else ""
    label = next(
        (
            method_label
            for token, method_label in _PACKED_QUANTIZATION_LABELS.items()
            if token in method
        ),
        None,
    )
    if label is None:
        return None

    bits = packed_transformers_quantization_bits(config)
    return f"{label} {bits}-bit" if bits is not None else label


def packed_transformers_quantization_bits(config: ModelConfig | None) -> int | None:
    """Read common top-level and compressed-tensors nested bit fields."""

    if config is None or not isinstance(config.quantization_config, dict):
        return None
    qcfg = config.quantization_config
    bits = _coerce_positive_int(
        qcfg.get("bits") or qcfg.get("w_bit") or qcfg.get("load_in_bits")
    )
    if bits is not None:
        return bits

    groups = qcfg.get("config_groups")
    if not isinstance(groups, dict):
        return None
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if not isinstance(weights, dict):
            continue
        bits = _coerce_positive_int(
            weights.get("num_bits") or weights.get("bits") or weights.get("w_bit")
        )
        if bits is not None:
            return bits
    return None


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "ArtifactConfirmation",
    "ArtifactFormat",
    "ArtifactProfile",
    "is_packed_transformers_quantization_config",
    "packed_transformers_quantization_bits",
    "packed_transformers_quantization_label",
]
