"""Turn "the estimator picked int4" into "does that artifact actually exist?".

Two signals feed the answer:

- **The selected configuration** — did the estimator pick a real GGUF variant,
  a native dtype, or an in-flight quantisation with no artifact behind it?
- **The repository itself** — does its ``config.json`` declare a
  ``quantization_config`` (AWQ, GPTQ, bitsandbytes) and/or do its files
  match a known pre-quantized layout?

An AWQ repo whose ``config.json`` has ``quantization_config.quant_method =
"awq"`` is CONFIRMED even when the recommender happened to render it as an
int4 estimate. That is the concrete regression the report language used to
misdescribe as "no pre-quantized artifact was found".
"""

from __future__ import annotations

from jaull.domain.artifact_profile import (
    ArtifactConfirmation,
    ArtifactFormat,
    ArtifactProfile,
    packed_transformers_quantization_bits,
)
from jaull.domain.candidates import ModelCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis
from jaull.estimator.policies import THEORETICAL_DTYPES

_AWQ_TAG_TOKENS = frozenset({"awq"})
_GPTQ_TAG_TOKENS = frozenset({"gptq"})
_BNB_TAG_TOKENS = frozenset({"bnb-4bit", "bnb-8bit", "bitsandbytes"})
_COMPRESSED_TAG_TOKENS = frozenset({"compressed-tensors"})


def analyze_artifact(
    candidate: ModelCandidate,
    analysis: ModelAnalysis | None,
    selected_configuration: InferenceConfiguration | None,
) -> ArtifactProfile:
    """Classify the artifact backing this recommendation."""
    reasons: list[str] = []

    # GGUF is unambiguous: the estimator only picks a ``quantization`` when a
    # matching variant exists on the Hub, so both the format and the confirmation
    # are decided by the configuration.
    if selected_configuration and selected_configuration.quantization:
        reasons.append(
            f"GGUF variant {selected_configuration.quantization!r} exists in the "
            "repository."
        )
        return ArtifactProfile(
            format=ArtifactFormat.GGUF,
            quantization=selected_configuration.quantization,
            confirmation=ArtifactConfirmation.CONFIRMED,
            reasons=reasons,
        )

    # Config-level detection wins over tag hints — the repo itself declared it.
    from_config = _detect_from_config(analysis)
    if from_config is not None:
        format_, quant, note = from_config
        reasons.append(note)
        return ArtifactProfile(
            format=format_,
            quantization=quant,
            confirmation=ArtifactConfirmation.CONFIRMED,
            reasons=reasons,
        )

    tag_hint = _detect_from_tags(candidate, analysis)
    precision = selected_configuration.precision if selected_configuration else None
    theoretical = (
        precision in THEORETICAL_DTYPES
        and (tag_hint is None or tag_hint[0] is ArtifactFormat.UNKNOWN)
    )

    if tag_hint is not None and tag_hint[0] is not ArtifactFormat.UNKNOWN:
        format_, quant, note = tag_hint
        reasons.append(note)
        # A tag alone is weaker than a config declaration — mark inferred so
        # the caller can still discount it if it wants to.
        return ArtifactProfile(
            format=format_,
            quantization=quant,
            confirmation=ArtifactConfirmation.INFERRED,
            reasons=reasons,
        )

    if theoretical:
        assert precision is not None
        reasons.append(
            f"{precision.value} is a theoretical estimate: no pre-quantized "
            "artifact was found for this repository."
        )
        return ArtifactProfile(
            format=ArtifactFormat.UNKNOWN,
            quantization=precision.value,
            confirmation=ArtifactConfirmation.THEORETICAL,
            reasons=reasons,
        )

    if precision is not None:
        reasons.append(
            f"Loaded at native precision {precision.value} (repository "
            "publishes weights at this dtype)."
        )
        return ArtifactProfile(
            format=ArtifactFormat.NATIVE,
            quantization=precision.value,
            confirmation=ArtifactConfirmation.CONFIRMED,
            reasons=reasons,
        )

    if analysis is not None:
        primary = analysis.classification.primary_type
        if primary is RepositoryType.TRANSFORMERS:
            reasons.append("Repository publishes native Transformers weights.")
            return ArtifactProfile(
                format=ArtifactFormat.NATIVE,
                quantization=None,
                confirmation=ArtifactConfirmation.CONFIRMED,
                reasons=reasons,
            )

    reasons.append("No confirmable artifact metadata was found.")
    return ArtifactProfile(
        format=ArtifactFormat.UNKNOWN,
        quantization=None,
        confirmation=ArtifactConfirmation.UNKNOWN,
        reasons=reasons,
    )


def _detect_from_config(
    analysis: ModelAnalysis | None,
) -> tuple[ArtifactFormat, str | None, str] | None:
    """Read ``config.json``'s ``quantization_config`` block, if present."""
    if analysis is None or analysis.config is None:
        return None
    qcfg = analysis.config.quantization_config
    if not isinstance(qcfg, dict):
        return None

    method_raw = qcfg.get("quant_method") or qcfg.get("quantization_method")
    method = str(method_raw).lower() if method_raw else ""
    bits = packed_transformers_quantization_bits(analysis.config)
    quant_label = f"int{bits}" if bits else None

    if "awq" in method:
        return ArtifactFormat.AWQ, quant_label, (
            "config.json declares an AWQ quantization_config."
        )
    if "gptq" in method:
        return ArtifactFormat.GPTQ, quant_label, (
            "config.json declares a GPTQ quantization_config."
        )
    if "bitsandbytes" in method or "bnb" in method:
        return ArtifactFormat.BITSANDBYTES, quant_label, (
            "config.json declares a bitsandbytes quantization_config."
        )
    if "compressed-tensors" in method or "compressed_tensors" in method:
        return ArtifactFormat.COMPRESSED_TENSORS, quant_label, (
            "config.json declares a compressed-tensors quantization_config."
        )
    return None


def _detect_from_tags(
    candidate: ModelCandidate,
    analysis: ModelAnalysis | None,
) -> tuple[ArtifactFormat, str | None, str] | None:
    tags = {tag.lower() for tag in candidate.tags}
    if analysis is not None:
        tags.update(tag.lower() for tag in analysis.repo.tags)

    name = candidate.repo_id.lower()

    if tags & _AWQ_TAG_TOKENS or name.endswith("-awq") or "-awq-" in name:
        return ArtifactFormat.AWQ, None, (
            "Repository name/tags advertise an AWQ artifact."
        )
    if tags & _GPTQ_TAG_TOKENS or name.endswith("-gptq") or "-gptq-" in name:
        return ArtifactFormat.GPTQ, None, (
            "Repository name/tags advertise a GPTQ artifact."
        )
    if tags & _BNB_TAG_TOKENS:
        return ArtifactFormat.BITSANDBYTES, None, (
            "Repository tags advertise a bitsandbytes artifact."
        )
    if tags & _COMPRESSED_TAG_TOKENS:
        return ArtifactFormat.COMPRESSED_TENSORS, None, (
            "Repository tags advertise a compressed-tensors artifact."
        )
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):  # bool is a subclass of int — guard first
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


__all__ = ["analyze_artifact"]
