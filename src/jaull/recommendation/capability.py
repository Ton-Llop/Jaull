"""Capability signal derived from inspected Hugging Face metadata.

No benchmarks are run and no external leaderboard is queried. The score is a
transparent heuristic over signals the workflow already has: parameter count,
metadata completeness, confirmed artifact/runtime hints and Hub activity. That
keeps recommendation tests offline while avoiding a hand-maintained quality
table for named model families.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from jaull.domain.candidates import ModelCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.families import detect_family, parameter_count
from jaull.domain.model import ModelAnalysis

DEFAULT_CAPABILITY_SCORE = 0.5

_CONFIDENCE_SIGNAL: dict[EstimationConfidence, float] = {
    EstimationConfidence.HIGH: 0.85,
    EstimationConfidence.MEDIUM: 0.65,
    EstimationConfidence.LOW: 0.45,
    EstimationConfidence.UNKNOWN: 0.25,
}

_REPOSITORY_TYPE_SIGNAL: dict[RepositoryType, float] = {
    RepositoryType.TRANSFORMERS: 0.75,
    RepositoryType.GGUF: 0.80,
    RepositoryType.ONNX: 0.65,
    RepositoryType.ADAPTER: 0.35,
    RepositoryType.DIFFUSERS: 0.20,
    RepositoryType.UNKNOWN: 0.50,
}

_REAL_ARTIFACT_TAGS = {
    "gguf",
    "safetensors",
    "4-bit",
    "8-bit",
    "awq",
    "gptq",
    "bnb-4bit",
    "bitsandbytes",
}

@dataclass(frozen=True)
class CapabilitySignal:
    """Explainable capability assessment for one candidate."""

    score: float
    parameter_count: int | None = None
    scale_tier: str | None = None
    family: str | None = None
    instruction_tuned: bool | None = None
    chat_or_base: str | None = None
    confidence: EstimationConfidence = EstimationConfidence.MEDIUM
    reasons: tuple[str, ...] = ()


class CapabilityAnalyzer(Protocol):
    """Anything that can score a candidate's expected model capability."""

    def analyze(
        self, candidate: ModelCandidate, analysis: ModelAnalysis | None
    ) -> CapabilitySignal: ...


class MetadataCapabilityAnalyzer:
    """Default analyzer using only already-fetched Hugging Face metadata."""

    def analyze(
        self, candidate: ModelCandidate, analysis: ModelAnalysis | None
    ) -> CapabilitySignal:
        params = parameter_count(candidate, analysis)
        family = detect_family(candidate, analysis)
        size_signal = _size_curve(params)
        metadata_signal = _metadata_signal(candidate, analysis)
        artifact_signal = _artifact_signal(candidate, analysis)
        activity_signal = _activity_signal(candidate)

        score = _clamp(
            0.45 * size_signal
            + 0.30 * metadata_signal
            + 0.15 * artifact_signal
            + 0.10 * activity_signal
        )

        return CapabilitySignal(
            score=score,
            parameter_count=params,
            scale_tier=_scale_tier(params),
            family=family,
            instruction_tuned=_instruction_tuned(candidate),
            chat_or_base=_chat_or_base(candidate),
            confidence=candidate.metadata_confidence,
            reasons=tuple(
                reason
                for reason in (
                    _size_reason(params),
                    _instruction_reason(candidate),
                    _metadata_reason(candidate, analysis),
                    _artifact_reason(candidate, analysis),
                    _activity_reason(candidate),
                )
                if reason is not None
            ),
        )


def capability_score(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> float:
    """Compatibility wrapper for older callers."""
    return MetadataCapabilityAnalyzer().analyze(candidate, analysis).score


def _metadata_signal(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> float:
    score = _CONFIDENCE_SIGNAL[candidate.metadata_confidence]
    if candidate.license:
        score += 0.05
    if candidate.languages:
        score += 0.05
    if analysis is not None:
        if analysis.config is not None:
            score += 0.08
        if analysis.total_size_bytes:
            score += 0.04
        score -= min(0.15, 0.03 * len(analysis.warnings))
    score -= min(0.15, 0.03 * len(candidate.penalties))
    return _clamp(score)


def _artifact_signal(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> float:
    repository_type = candidate.repository_type
    if analysis is not None:
        repository_type = analysis.classification.primary_type
    score = (
        _REPOSITORY_TYPE_SIGNAL[repository_type]
        if repository_type is not None
        else _REPOSITORY_TYPE_SIGNAL[RepositoryType.UNKNOWN]
    )

    tags = {tag.lower() for tag in candidate.tags}
    if tags & _REAL_ARTIFACT_TAGS:
        score += 0.10
    if candidate.pipeline_tag == "text-generation":
        score += 0.05
    if analysis is not None and analysis.relevant_files:
        score += 0.05
    return _clamp(score)


def _activity_signal(candidate: ModelCandidate) -> float:
    downloads = math.log1p(max(0, candidate.downloads)) / math.log1p(1_000_000)
    likes = math.log1p(max(0, candidate.likes)) / math.log1p(20_000)
    return _clamp(0.75 * min(downloads, 1.0) + 0.25 * min(likes, 1.0))


def _size_curve(params: int | None) -> float:
    """Sigmoid over parameter count: 0.5B -> ~0.2, 8B -> ~0.7, 70B -> ~0.95."""
    if params is None or params <= 0:
        return DEFAULT_CAPABILITY_SCORE
    x = math.log10(params) - 9.5
    return 1.0 / (1.0 + math.exp(-1.8 * x))


def _scale_tier(params: int | None) -> str | None:
    """Coarse capacity prior for explanation, never a hard quality rule."""
    if params is None or params <= 0:
        return None
    if params < 300_000_000:
        return "lightweight"
    if params < 1_500_000_000:
        return "small"
    if params < 4_000_000_000:
        return "balanced"
    if params < 7_000_000_000:
        return "mid_size"
    return "higher_capacity"


def _size_reason(params: int | None) -> str | None:
    if params is None:
        return "Parameter count could not be confirmed."
    if params >= 1_000_000_000:
        return f"Estimated around {params / 1_000_000_000:.1f}B parameters."
    return f"Estimated around {params / 1_000_000:.0f}M parameters."


def _instruction_tuned(candidate: ModelCandidate) -> bool | None:
    tokens = _model_tokens(candidate)
    if any(
        _has_signal(tokens, token)
        for token in ("instruct", "instruction", "chat", "conversational")
    ):
        return True
    if any(_has_signal(tokens, token) for token in ("base", "pretrain", "foundation")):
        return False
    if _has_signal(tokens, "assistant"):
        return True
    return None


def _chat_or_base(candidate: ModelCandidate) -> str | None:
    tokens = _model_tokens(candidate)
    if _has_signal(tokens, "chat") or _has_signal(tokens, "conversational"):
        return "chat"
    if _has_signal(tokens, "instruct") or _has_signal(tokens, "instruction"):
        return "instruction"
    if any(_has_signal(tokens, token) for token in ("base", "pretrain", "foundation")):
        return "base"
    if _has_signal(tokens, "assistant"):
        return "chat"
    return None


def _instruction_reason(candidate: ModelCandidate) -> str | None:
    kind = _chat_or_base(candidate)
    if kind == "chat":
        return "Metadata/name suggests chat tuning."
    if kind == "instruction":
        return "Metadata/name suggests instruction tuning."
    if kind == "base":
        return "Metadata/name suggests a base model rather than an assistant-tuned model."
    return None


def _model_tokens(candidate: ModelCandidate) -> set[str]:
    values = [
        candidate.repo_id.split("/")[-1],
        candidate.pipeline_tag or "",
        candidate.library_name or "",
        *candidate.tags,
    ]
    tokens: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.add(normalized.replace("_", "-"))
        tokens.update(token for token in re.split(r"[^a-z0-9.]+", normalized) if token)
    return tokens


def _has_signal(tokens: set[str], signal: str) -> bool:
    normalized = signal.lower()
    return normalized in tokens or normalized.replace("_", "-") in tokens


def _metadata_reason(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> str | None:
    if analysis is not None and analysis.config is not None:
        return "Repository exposes model configuration metadata."
    if candidate.metadata_confidence is EstimationConfidence.LOW:
        return "Model card metadata is incomplete."
    return None


def _artifact_reason(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> str | None:
    repository_type = candidate.repository_type
    if analysis is not None:
        repository_type = analysis.classification.primary_type
    if repository_type is RepositoryType.GGUF:
        return "Repository contains a runnable GGUF artifact."
    if repository_type is RepositoryType.TRANSFORMERS:
        return "Repository can be inspected as a Transformers model."
    return None


def _activity_reason(candidate: ModelCandidate) -> str | None:
    if candidate.downloads <= 0 and candidate.likes <= 0:
        return None
    return "Hub activity provides a secondary maturity signal."


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "DEFAULT_CAPABILITY_SCORE",
    "CapabilityAnalyzer",
    "CapabilitySignal",
    "MetadataCapabilityAnalyzer",
    "capability_score",
    "detect_family",
    "parameter_count",
]
