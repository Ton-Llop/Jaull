"""Model-family detection and parameter-count guessing.

Pure heuristics over a candidate's repo id and any inspected configuration.
No scoring logic here — those live in ``recommendation.capability`` — but the
family key and parameter count are shared contracts used by discovery
(series grouping) as well, so they sit in ``domain`` to keep the boundary
clean.
"""

from __future__ import annotations

import re

from jaull.domain.candidates import ModelCandidate
from jaull.domain.model import ModelAnalysis

_VARIANT_TOKENS = {
    "base",
    "chat",
    "coder",
    "gguf",
    "hf",
    "instruct",
    "it",
    "onnx",
    "quantized",
    "sft",
    "vision",
}

_PARAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])"),
    re.compile(r"(\d+(?:\.\d+)?)[-_]?[bB](?![a-zA-Z])"),
)

_SIZE_MARKER_RE = re.compile(r"[-_]\d+(?:\.\d+)?[bB](?![a-zA-Z])")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9.]+")
_VERSION_RE = re.compile(r"^(?:v?\d+(?:\.\d+)*|r\d+)$")


def detect_family(
    candidate: ModelCandidate, analysis: ModelAnalysis | None = None
) -> str:
    """Best-effort family key for series grouping, not a quality tier."""
    config_type = analysis.config.model_type if analysis and analysis.config else None
    if config_type:
        return _normalise_family(config_type)

    source = candidate.base_model_repo_id or candidate.repo_id
    family = _family_from_repo_id(source)
    return family or "unknown"


def parameter_count(
    candidate: ModelCandidate, analysis: ModelAnalysis | None
) -> int | None:
    """Best-effort parameter count, from config or repo-id suffix. Never 0."""
    if analysis is not None:
        config = analysis.config
        if config and config.num_hidden_layers and config.hidden_size:
            approx = 12 * config.num_hidden_layers * config.hidden_size * config.hidden_size
            if approx > 0:
                return approx

    for pattern in _PARAM_PATTERNS:
        match = pattern.search(candidate.repo_id)
        if match:
            billions = float(match.group(1))
            return int(billions * 1_000_000_000)
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
        if token in _VARIANT_TOKENS:
            break
        family_tokens.append(token)
        if len(family_tokens) >= 2 and not _VERSION_RE.match(token):
            break

    if not family_tokens:
        family_tokens = [tokens[0]]
    return _normalise_family("-".join(family_tokens))


def _normalise_family(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-") or "unknown"


__all__ = ["detect_family", "parameter_count"]
