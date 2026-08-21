"""Artifact quantization semantics shared by execution-plan builders."""

from __future__ import annotations

from jaull.domain.model import ModelConfig

_PACKED_QUANTIZATION_LABELS = {
    "awq": "AWQ",
    "gptq": "GPTQ",
}


def packed_transformers_quantization_label(config: ModelConfig | None) -> str | None:
    """Return the artifact label for config-declared packed Transformers weights.

    ``quantization_config`` describes the stored artifact, not the runtime
    compute dtype. Only explicit config metadata is used here; repository names
    and tags remain weaker evidence handled elsewhere.
    """

    if config is None or not isinstance(config.quantization_config, dict):
        return None

    method_raw = config.quantization_config.get("quant_method") or (
        config.quantization_config.get("quantization_method")
    )
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

    bits = _coerce_positive_int(
        config.quantization_config.get("bits")
        or config.quantization_config.get("w_bit")
        or config.quantization_config.get("load_in_bits")
    )
    return f"{label} {bits}-bit" if bits is not None else label


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
