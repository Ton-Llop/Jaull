"""Pick a GGUF variant to estimate against."""

from __future__ import annotations

from dataclasses import dataclass

from jaull.domain.model import GgufVariant
from jaull.estimator.policies import DEFAULT_GGUF_QUANTIZATION
from jaull.exceptions import QuantizationNotFoundError


@dataclass(frozen=True)
class VariantChoice:
    variant: GgufVariant
    reason: str  # human-readable explanation for the pick


def select_variant(
    variants: list[GgufVariant], requested: str | None
) -> VariantChoice:
    if not variants:
        raise QuantizationNotFoundError(
            requested=requested or "",
            available=[],
        )

    if requested:
        needle = requested.lower()
        for variant in variants:
            if variant.quantization.lower() == needle:
                return VariantChoice(
                    variant=variant,
                    reason=f"Selected explicitly by user: {variant.quantization}.",
                )
        raise QuantizationNotFoundError(
            requested=requested,
            available=[v.quantization for v in variants],
        )

    for variant in variants:
        if variant.quantization.upper() == DEFAULT_GGUF_QUANTIZATION:
            return VariantChoice(
                variant=variant,
                reason=(
                    f"Auto-selected {DEFAULT_GGUF_QUANTIZATION} as the default "
                    "quality/size trade-off."
                ),
            )

    ordered = sorted(variants, key=lambda v: v.total_bytes)
    median = ordered[len(ordered) // 2]
    return VariantChoice(
        variant=median,
        reason=(
            f"Auto-selected {median.quantization} as the median-sized variant "
            f"(no {DEFAULT_GGUF_QUANTIZATION} available)."
        ),
    )
