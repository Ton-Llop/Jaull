"""Resolve GGUF representations without equating quantization with artifact identity."""

from jaull.domain.model import GgufVariant
from jaull.exceptions import GgufVariantSelectionError


def unique_gguf_representation(variants: list[GgufVariant]) -> GgufVariant:
    complete = [variant for variant in variants if variant.is_complete]
    whole = [variant for variant in complete if len(variant.files) == 1 and not variant.is_split]
    candidates = whole or complete
    if len(candidates) != 1:
        names = ", ".join(file.path for variant in variants for file in variant.files)
        reason = (
            "Multiple artifacts share this quantization"
            if candidates
            else "Incomplete GGUF shard set"
        )
        raise GgufVariantSelectionError(f"{reason}: {names}. No artifact was selected.")
    return candidates[0]


def preferred_gguf_variants(variants: list[GgufVariant]) -> list[GgufVariant]:
    """One complete unambiguous representation per quantization for plan generation."""
    result: list[GgufVariant] = []
    for quant in sorted({variant.quantization for variant in variants}):
        try:
            result.append(
                unique_gguf_representation(
                    [variant for variant in variants if variant.quantization == quant]
                )
            )
        except GgufVariantSelectionError:
            continue
    return result
