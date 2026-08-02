"""Search side of the Hugging Face API.

Kept apart from :class:`~local_ai_check.huggingface.client.HfClient` on purpose:
that client's protocol describes per-repository reads, and widening it would
invalidate every existing test double. Search is a different capability, so it
gets its own narrow protocol.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import ModelInfo

from local_ai_check.discovery.models import ModelCandidate, SearchQuery
from local_ai_check.domain.estimation import EstimationConfidence
from local_ai_check.exceptions import HuggingFaceUnavailableError

logger = logging.getLogger(__name__)


class ModelSearchClient(Protocol):
    """Anything that can turn a :class:`SearchQuery` into candidates."""

    def search(self, query: SearchQuery) -> list[ModelCandidate]: ...


class HfSearchClient:
    """Real search client, wrapping ``HfApi.list_models``.

    Note on the API surface: ``list_models`` in huggingface_hub 1.x has no
    ``direction``, ``language``, ``library``, ``tags`` or ``task`` parameters —
    those were folded into ``filter`` (an iterable of tag strings) and
    ``search``. ``sort`` is always descending.
    """

    def __init__(self, token: str | None = None, api: HfApi | None = None) -> None:
        # Token is honoured when present (higher rate limits, gated visibility)
        # but is never logged or surfaced.
        self._token = token or os.environ.get("HF_TOKEN")
        self._api = api or HfApi(token=self._token)

    def search(self, query: SearchQuery) -> list[ModelCandidate]:
        kwargs: dict[str, Any] = {
            "limit": query.limit,
            "sort": query.sort,
            "cardData": True,
            # Exclude gated repositories server-side: without access they cannot
            # be inspected, so spending a slot of the budget on them is waste.
            "gated": False,
        }
        if query.search:
            kwargs["search"] = query.search
        if query.pipeline_tag:
            kwargs["pipeline_tag"] = query.pipeline_tag
        if query.filter_tags:
            kwargs["filter"] = list(query.filter_tags)

        try:
            results = list(self._api.list_models(**kwargs))
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                raise HuggingFaceUnavailableError(
                    "Hugging Face rate limit reached while searching for models. "
                    "Try again in a moment, or set HF_TOKEN for higher limits."
                ) from exc
            raise HuggingFaceUnavailableError(
                f"Hugging Face returned HTTP {status} while searching for models."
            ) from exc
        except OSError as exc:
            raise HuggingFaceUnavailableError(
                "Unable to reach Hugging Face while searching for models."
            ) from exc

        return [candidate_from_model_info(info, query.label) for info in results]


def candidate_from_model_info(info: ModelInfo, query_label: str) -> ModelCandidate:
    """Map a raw ``ModelInfo`` onto our candidate model, tolerating missing fields."""
    card_data = _card_data(info)
    repo_id = str(getattr(info, "id", None) or getattr(info, "modelId", "") or "")

    languages = _string_list(card_data.get("language"))
    license_value = _first_string(card_data.get("license"))
    tags = [str(tag) for tag in (getattr(info, "tags", None) or [])]

    confidence = EstimationConfidence.MEDIUM
    if not card_data:
        confidence = EstimationConfidence.LOW
    elif license_value and languages:
        confidence = EstimationConfidence.HIGH

    return ModelCandidate(
        repo_id=repo_id,
        pipeline_tag=getattr(info, "pipeline_tag", None),
        library_name=getattr(info, "library_name", None),
        license=license_value,
        languages=languages,
        tags=tags,
        downloads=int(getattr(info, "downloads", None) or 0),
        likes=int(getattr(info, "likes", None) or 0),
        gated=getattr(info, "gated", False) or False,
        private=bool(getattr(info, "private", False)),
        base_model_repo_id=_first_string(card_data.get("base_model")),
        source_queries=[query_label],
        metadata_confidence=confidence,
    )


def _card_data(info: ModelInfo) -> dict[str, Any]:
    card_data = getattr(info, "card_data", None) or {}
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    if isinstance(card_data, dict):
        return card_data
    return {}


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str | int | float)]
    return []


def _first_string(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, list) and value:
        first = value[0]
        return str(first) if first else None
    if isinstance(value, dict):
        # base_model can be {"finetune": "org/model"} and similar shapes.
        for key in ("finetune", "quantized", "quantized_by", "base_model"):
            nested = value.get(key)
            if isinstance(nested, str) and nested:
                return nested
    return None


__all__ = [
    "HfSearchClient",
    "ModelSearchClient",
    "candidate_from_model_info",
]
