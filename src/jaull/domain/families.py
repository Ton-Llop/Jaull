"""Model-family detection and parameter-count guessing.

Pure heuristics over a candidate's repo id and any inspected configuration.
No scoring logic here — those live in ``recommendation.capability`` — but the
family key and parameter count are shared contracts used by discovery
(series grouping) as well, so they sit in ``domain`` to keep the boundary
clean.
"""

from __future__ import annotations

import re

from jaull.domain.artifact_profile import is_packed_transformers_quantization_config
from jaull.domain.candidates import ModelCandidate
from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    WeightEstimate,
)
from jaull.domain.model import ModelAnalysis, ModelConfig
from jaull.domain.parameters import (
    ParameterCount,
    ParameterCountSource,
    estimate_total_parameters_from_config,
)

# Tokens that split a repository name into "family" (kept) and "variant"
# (bucketed by series). Format tokens (gguf/awq/…) are stripped when building
# the family key so ``Qwen2.5-3B-Instruct`` and ``Qwen2.5-3B-Instruct-GGUF``
# collapse. Functional specialisations (``coder``, ``vision``, ``embedding``)
# stay in the family key so ``Qwen2.5-Coder-7B`` is not grouped with
# ``Qwen2.5-7B`` general.
_VARIANT_TOKENS = {
    "base",
    "chat",
    "instruct",
    "it",
    "sft",
    "rlhf",
    "dpo",
}

_FORMAT_TOKENS = {
    "gguf",
    "awq",
    "gptq",
    "onnx",
    "hf",
    "quantized",
    "bnb",
}

_PARAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])"),
    re.compile(r"(\d+(?:\.\d+)?)[-_]?([bBmM])(?![a-zA-Z])"),
)

_SIZE_MARKER_RE = re.compile(r"[-_]\d+(?:\.\d+)?[bBmM](?![a-zA-Z])")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9.]+")
_VERSION_RE = re.compile(r"^(?:v?\d+(?:\.\d+)*|r\d+)$")


def detect_family(
    candidate: ModelCandidate, analysis: ModelAnalysis | None = None
) -> str:
    """Best-effort family key for series grouping, not a quality tier."""
    config_type = analysis.config.model_type if analysis and analysis.config else None
    source = candidate.base_model_repo_id or candidate.repo_id
    name_family = _family_from_repo_id(source)

    if config_type:
        base = _normalise_family(config_type)
        specialization = _extract_specialization(source)
        if specialization and specialization not in base:
            return f"{base}-{specialization}"
        return base

    return name_family or "unknown"


def _extract_specialization(repo_id: str) -> str | None:
    """Return a functional-specialisation token (``coder``, ``vision``, …).

    Present in the name → the family key must keep it so ``Qwen2.5-Coder``
    never collapses into general ``Qwen2.5``. Distinct from *variant* tokens
    (instruct/chat/base) which affect the series bucket, not the family.
    """
    name = repo_id.split("/")[-1].lower()
    for keyword in ("coder", "code", "vision", "embedding", "embed", "reranker"):
        if keyword in name:
            return keyword
    return None


def resolve_parameter_count(
    candidate: ModelCandidate,
    analysis: ModelAnalysis | None,
    weight_estimate: WeightEstimate | None = None,
) -> ParameterCount:
    """Layered source hierarchy — the strongest evidence wins.

    1. ``safetensors_summary.total_parameters`` on the analysis (HIGH) *or* a
       ``WeightEstimate`` whose ``num_parameters`` came from safetensors
       metadata (also HIGH — same source, different plumbing).
    2. A full sum from ``ModelConfig`` (embeddings + attention + FFN, GQA
       aware). Only when the config carries every dimension we need (MEDIUM).
    3. GGUF header ``general.parameter_count`` — future work; not populated
       today, but the hierarchy is prepared for it.
    4. ``…-7B-…`` suffix in the repo id (LOW).
    5. Nothing → UNKNOWN.
    """
    packed_quantization = (
        analysis is not None
        and is_packed_transformers_quantization_config(analysis.config)
    )
    if analysis is not None and not packed_quantization:
        summary = getattr(analysis, "safetensors_summary", None)
        if summary is not None and summary.total_parameters > 0:
            return ParameterCount(
                count=summary.total_parameters,
                source=ParameterCountSource.SAFETENSORS_METADATA,
                confidence=EstimationConfidence.HIGH,
            )

    if (
        weight_estimate is not None
        and weight_estimate.num_parameters
        and not packed_quantization
    ):
        source = weight_estimate.component.source
        if source is EstimateSource.METADATA:
            return ParameterCount(
                count=weight_estimate.num_parameters,
                source=ParameterCountSource.SAFETENSORS_METADATA,
                confidence=EstimationConfidence.HIGH,
            )

    if analysis is not None:
        config_count = _from_config(analysis.config)
        if config_count is not None:
            return ParameterCount(
                count=config_count,
                source=ParameterCountSource.MODEL_CONFIG,
                confidence=EstimationConfidence.MEDIUM,
            )

    name_count = _from_repo_id(candidate.repo_id)
    if name_count is not None:
        return ParameterCount(
            count=name_count,
            source=ParameterCountSource.NAME_INFERENCE,
            confidence=EstimationConfidence.LOW,
        )

    return ParameterCount(
        count=None,
        source=ParameterCountSource.UNKNOWN,
        confidence=EstimationConfidence.UNKNOWN,
    )


def parameter_count(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> int | None:
    """Backwards-compatible facade returning just the count (or ``None``).

    New code should prefer :func:`resolve_parameter_count` so it can inspect
    the source and confidence.
    """
    return resolve_parameter_count(candidate, analysis).count


def _from_config(config: ModelConfig | None) -> int | None:
    """Approximate total parameters from a Transformers ``config.json``.

    Includes token embeddings, attention (Q/K/V/O with GQA when
    ``num_key_value_heads`` differs from ``num_attention_heads``), FFN (SwiGLU
    when ``intermediate_size`` is declared, classic 4h fallback otherwise) and
    the LM head (skipped when tied to the input embedding). Rough enough to
    stand in for a real measurement — it lands within a few percent of the
    manufacturer-reported size — but honestly under ``MEDIUM`` confidence.
    """
    return estimate_total_parameters_from_config(config)


def _from_repo_id(repo_id: str) -> int | None:
    for pattern in _PARAM_PATTERNS:
        match = pattern.search(repo_id)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            multiplier = 1_000_000_000 if unit == "b" else 1_000_000
            return int(value * multiplier)
    return None


def _family_from_repo_id(repo_id: str) -> str | None:
    name = repo_id.split("/")[-1].lower()
    owner = repo_id.split("/", 1)[0].lower() if "/" in repo_id else ""
    if _SIZE_MARKER_RE.search(name) is None:
        return None
    without_size = _SIZE_MARKER_RE.sub("", name)
    tokens = [token for token in _TOKEN_SPLIT_RE.split(without_size) if token]
    if not tokens:
        return None

    owner_tokens = {token for token in _TOKEN_SPLIT_RE.split(owner) if token}
    if len(tokens) > 2 and tokens[0] in owner_tokens:
        tokens = tokens[1:]

    family_tokens: list[str] = []
    for token in tokens:
        if token in _VARIANT_TOKENS or token in _FORMAT_TOKENS:
            break
        family_tokens.append(token)
        if len(family_tokens) >= 3 and not _VERSION_RE.match(token):
            break

    if not family_tokens:
        family_tokens = [tokens[0]]
    return _normalise_family("-".join(family_tokens))


def _normalise_family(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-") or "unknown"


__all__ = [
    "detect_family",
    "parameter_count",
    "resolve_parameter_count",
]
