"""Pick a GGUF variant to estimate against."""

from __future__ import annotations

from dataclasses import dataclass

from jaull.domain.gguf_variants import preferred_gguf_variants, unique_gguf_representation
from jaull.domain.model import GgufVariant
from jaull.estimator.policies import DEFAULT_GGUF_QUANTIZATION
from jaull.exceptions import GgufVariantSelectionError, QuantizationNotFoundError


@dataclass(frozen=True)
class VariantChoice:
    variant: GgufVariant
    reason: str  # human-readable explanation for the pick


def select_variant(variants: list[GgufVariant], requested: str | None) -> VariantChoice:
    if not variants:
        raise QuantizationNotFoundError(
            requested=requested or "",
            available=[],
        )

    if requested:
        needle = requested.lower()
        matching = [variant for variant in variants if variant.quantization.lower() == needle]
        if matching:
            return VariantChoice(
                variant=unique_gguf_representation(matching),
                reason=f"Selected explicitly by user: {matching[0].quantization}.",
            )
        raise QuantizationNotFoundError(
            requested=requested,
            available=[v.quantization for v in variants],
        )

    for variant in variants:
        if variant.quantization.upper() == DEFAULT_GGUF_QUANTIZATION:
            return VariantChoice(
                variant=select_variant(variants, DEFAULT_GGUF_QUANTIZATION).variant,
                reason=(
                    f"Auto-selected {DEFAULT_GGUF_QUANTIZATION} as the default "
                    "quality/size trade-off."
                ),
            )

    ordered = sorted(
        preferred_gguf_variants(variants),
        key=lambda v: (v.total_bytes is None, v.total_bytes or 0, v.quantization),
    )
    if not ordered:
        raise GgufVariantSelectionError("No unambiguous complete GGUF representation is available.")
    median = ordered[len(ordered) // 2]
    return VariantChoice(
        variant=median,
        reason=(
            f"Auto-selected {median.quantization} as the median-sized variant "
            f"(no {DEFAULT_GGUF_QUANTIZATION} available)."
        ),
    )
