from __future__ import annotations

import pytest

from local_ai_check.domain.model import GgufVariant, ModelFile
from local_ai_check.estimator import gguf_selection
from local_ai_check.exceptions import QuantizationNotFoundError


def _variant(quant: str, size: int) -> GgufVariant:
    return GgufVariant(
        quantization=quant,
        files=[ModelFile(path=f"model-{quant}.gguf", size_bytes=size)],
        total_bytes=size,
    )


def test_explicit_selection_matches() -> None:
    variants = [_variant("Q4_K_M", 4), _variant("Q8_0", 8)]
    choice = gguf_selection.select_variant(variants, requested="Q4_K_M")
    assert choice.variant.quantization == "Q4_K_M"
    assert "explicitly" in choice.reason.lower()


def test_selection_is_case_insensitive() -> None:
    variants = [_variant("Q4_K_M", 4)]
    choice = gguf_selection.select_variant(variants, requested="q4_k_m")
    assert choice.variant.quantization == "Q4_K_M"


def test_missing_quantization_raises_with_list() -> None:
    variants = [_variant("Q4_K_M", 4), _variant("Q8_0", 8)]
    with pytest.raises(QuantizationNotFoundError) as info:
        gguf_selection.select_variant(variants, requested="Q3_K_L")
    assert info.value.available == ["Q4_K_M", "Q8_0"]


def test_auto_selects_default_q4_k_m() -> None:
    variants = [
        _variant("Q3_K_S", 3),
        _variant("Q4_K_M", 4),
        _variant("Q8_0", 8),
    ]
    choice = gguf_selection.select_variant(variants, requested=None)
    assert choice.variant.quantization == "Q4_K_M"
    assert "Auto-selected" in choice.reason


def test_auto_falls_back_to_median_when_default_missing() -> None:
    variants = [_variant("Q3_K_S", 3), _variant("Q5_K_M", 5), _variant("Q8_0", 8)]
    choice = gguf_selection.select_variant(variants, requested=None)
    assert choice.variant.quantization == "Q5_K_M"
    assert "median" in choice.reason.lower()


def test_empty_variants_raises() -> None:
    with pytest.raises(QuantizationNotFoundError):
        gguf_selection.select_variant([], requested="Q4_K_M")
