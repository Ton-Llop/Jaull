"""Resolve the base repository behind a GGUF variant, with explicit provenance."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from huggingface_hub.hf_api import ModelInfo

from local_ai_check.domain.enrichment import GgufHeaderMetadata
from local_ai_check.domain.estimation import (
    BaseModelResolution,
    EstimationConfidence,
    MetadataSource,
)

_HF_URL_RE = re.compile(
    r"https?://huggingface\.co/([^/\s#?]+/[^/\s#?]+)", re.IGNORECASE
)
_RELATION_KEYS = ("finetune", "quantized_by", "merge", "adapter", "base_model")


def resolve_base_model(
    model_info: ModelInfo | None,
    gguf_header: GgufHeaderMetadata | None,
    repo_id: str,
) -> BaseModelResolution:
    """Try each source in priority order until one gives a confident answer.

    Order:
      1. Model card ``base_model`` field (highest confidence).
      2. GGUF header ``general.source.huggingface.repository``.
      3. Explicit ``https://huggingface.co/...`` URL in the model card.
      4. Nothing — return ``UNRESOLVED``. Repository name heuristics
         (``X-GGUF`` → ``X``) are only surfaced as evidence, never as a
         resolved answer.
    """
    card_data = _extract_card_data(model_info)

    from_card = _from_card_data(card_data)
    if from_card is not None:
        return from_card

    from_gguf = _from_gguf_header(gguf_header)
    if from_gguf is not None:
        return from_gguf

    from_url = _from_url(card_data)
    if from_url is not None:
        return from_url

    hint = _name_hint(repo_id)
    warnings = [
        "Base model could not be resolved from model card, GGUF header or URLs.",
    ]
    evidence: list[str] = []
    if hint:
        evidence.append(
            f"Repository name suggests conversion of {hint!r} but this is only a hint."
        )
    return BaseModelResolution(
        repo_id=None,
        source=MetadataSource.UNRESOLVED,
        confidence=EstimationConfidence.LOW,
        evidence=evidence,
        warnings=warnings,
    )


def _extract_card_data(model_info: ModelInfo | None) -> dict[str, Any]:
    if model_info is None:
        return {}
    card = getattr(model_info, "card_data", None)
    if card is None:
        return {}
    if hasattr(card, "to_dict"):
        result = card.to_dict()
        return dict(result) if isinstance(result, dict) else {}
    if isinstance(card, dict):
        return dict(card)
    return {}


def _from_card_data(card_data: Mapping[str, Any]) -> BaseModelResolution | None:
    if not card_data:
        return None
    value = card_data.get("base_model")
    if value is None:
        return None

    if isinstance(value, str):
        candidate = _normalise_ref(value)
        if candidate is None:
            return None
        return BaseModelResolution(
            repo_id=candidate,
            source=MetadataSource.MODEL_CARD_METADATA,
            confidence=EstimationConfidence.HIGH,
            evidence=[f"Model card `base_model: {candidate}`."],
        )

    if isinstance(value, list):
        candidates = [
            c
            for c in (_normalise_ref(item) for item in value if isinstance(item, str))
            if c
        ]
        if len(candidates) == 1:
            return BaseModelResolution(
                repo_id=candidates[0],
                source=MetadataSource.MODEL_CARD_METADATA,
                confidence=EstimationConfidence.HIGH,
                evidence=[f"Model card `base_model` list with one entry: {candidates[0]}."],
            )
        if len(candidates) > 1:
            return BaseModelResolution(
                repo_id=None,
                source=MetadataSource.UNRESOLVED,
                confidence=EstimationConfidence.LOW,
                candidates=candidates,
                warnings=[
                    "Model card lists multiple base models; refusing to pick one automatically."
                ],
                evidence=[f"Candidates: {', '.join(candidates)}."],
            )
        return None

    if isinstance(value, dict):
        picked: list[str] = []
        for key in _RELATION_KEYS:
            entry = value.get(key)
            if isinstance(entry, str):
                normalised = _normalise_ref(entry)
                if normalised:
                    picked.append(normalised)
            elif isinstance(entry, list):
                picked.extend(
                    c
                    for c in (_normalise_ref(x) for x in entry if isinstance(x, str))
                    if c
                )
        unique = list(dict.fromkeys(picked))
        if len(unique) == 1:
            return BaseModelResolution(
                repo_id=unique[0],
                source=MetadataSource.MODEL_CARD_METADATA,
                confidence=EstimationConfidence.HIGH,
                evidence=[f"Model card relation entry resolved to {unique[0]}."],
            )
        if len(unique) > 1:
            return BaseModelResolution(
                repo_id=None,
                source=MetadataSource.UNRESOLVED,
                confidence=EstimationConfidence.LOW,
                candidates=unique,
                warnings=[
                    "Model card relations point to multiple base models; ambiguous."
                ],
            )
    return None


def _from_gguf_header(header: GgufHeaderMetadata | None) -> BaseModelResolution | None:
    if header is None or not header.source_repository:
        return None
    candidate = _normalise_ref(header.source_repository)
    if candidate is None:
        return None
    return BaseModelResolution(
        repo_id=candidate,
        source=MetadataSource.GGUF_METADATA,
        confidence=EstimationConfidence.HIGH,
        evidence=[
            f"GGUF header `general.source.huggingface.repository` = {candidate}."
        ],
    )


def _from_url(card_data: Mapping[str, Any]) -> BaseModelResolution | None:
    for key in ("source", "source_url", "homepage", "url"):
        value = card_data.get(key)
        if isinstance(value, str):
            match = _HF_URL_RE.search(value)
            if match:
                candidate = match.group(1)
                return BaseModelResolution(
                    repo_id=candidate,
                    source=MetadataSource.EXPLICIT_REPOSITORY_REFERENCE,
                    confidence=EstimationConfidence.MEDIUM,
                    evidence=[
                        f"Model card field {key!r} references {candidate}."
                    ],
                )
    return None


def _name_hint(repo_id: str) -> str | None:
    from local_ai_check.metadata.policies import GGUF_SUFFIX_HINTS

    for suffix in GGUF_SUFFIX_HINTS:
        if repo_id.endswith(suffix):
            base = repo_id[: -len(suffix)]
            if "/" in base:
                return base
    return None


def _normalise_ref(value: str) -> str | None:
    ref = value.strip()
    if not ref:
        return None
    # URLs → owner/name
    match = _HF_URL_RE.match(ref)
    if match:
        return match.group(1)
    if "/" in ref and not ref.startswith(("http", "www")):
        return ref
    return None


__all__ = ["resolve_base_model"]
