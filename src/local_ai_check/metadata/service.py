"""Orchestrator that resolves a GGUF variant's base model and merges its config."""

from __future__ import annotations

import json
import logging
from pathlib import PurePosixPath
from typing import Protocol

from huggingface_hub import hf_hub_url

from local_ai_check.analyzers.transformers import _model_config_from_dict
from local_ai_check.domain.enrichment import (
    EnrichmentResult,
    GgufHeaderMetadata,
)
from local_ai_check.domain.estimation import (
    BaseModelResolution,
    EstimationConfidence,
    MetadataSource,
)
from local_ai_check.domain.model import GgufVariant, ModelAnalysis, ModelConfig
from local_ai_check.exceptions import (
    HuggingFaceUnavailableError,
    LocalAiCheckError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)
from local_ai_check.huggingface.client import HfClientProtocol
from local_ai_check.metadata import base_model_resolver, config_merger, range_reader
from local_ai_check.metadata.range_reader import HttpRangeClient

logger = logging.getLogger(__name__)


class RangeClientFactory(Protocol):
    """Provides an :class:`HttpRangeClient` for a given repository access token."""

    def __call__(self, token: str | None) -> HttpRangeClient: ...


def enrich(
    analysis: ModelAnalysis,
    variant: GgufVariant | None,
    client: HfClientProtocol,
    *,
    range_client: HttpRangeClient | None = None,
    hf_token: str | None = None,
) -> EnrichmentResult:
    """Resolve the base model and combine its config with the GGUF header.

    Args:
        analysis: Result of :func:`inspect_model`.
        variant: The chosen GGUF variant whose header will be read. ``None`` skips
            header reading entirely (still tries to resolve the base model).
        client: HF client (for ``model_info`` and ``download_small_file``).
        range_client: Injectable HTTP client for Range requests. When ``None``,
            no GGUF header will be read.
        hf_token: Optional token forwarded to the range client (unused here but
            declared for symmetry with future callers).

    Never raises for expected failures — degrades to a partial result with
    warnings so the caller can keep computing weights-only estimates.
    """
    warnings: list[str] = []
    model_info = None
    try:
        model_info = client.model_info(analysis.repo.repo_id)
    except LocalAiCheckError as exc:
        warnings.append(f"Could not re-read model info while enriching: {exc}")

    gguf_header: GgufHeaderMetadata | None = None
    if variant is not None and range_client is not None:
        gguf_header = _read_variant_header(
            variant=variant,
            repo_id=analysis.repo.repo_id,
            range_client=range_client,
            warnings=warnings,
        )

    base_resolution = base_model_resolver.resolve_base_model(
        model_info=model_info,
        gguf_header=gguf_header,
        repo_id=analysis.repo.repo_id,
    )
    warnings.extend(base_resolution.warnings)

    base_config = None
    if (
        base_resolution.repo_id
        and base_resolution.confidence
        in {EstimationConfidence.HIGH, EstimationConfidence.MEDIUM}
    ):
        base_config = _fetch_base_config(
            repo_id=base_resolution.repo_id,
            client=client,
            warnings=warnings,
        )

    enriched = config_merger.merge(gguf_header=gguf_header, base_config=base_config)
    if enriched is not None:
        warnings.extend(enriched.conflicts)

    return EnrichmentResult(
        enriched_config=enriched,
        base_model_resolution=base_resolution,
        gguf_header=gguf_header,
        warnings=warnings,
    )


def _read_variant_header(
    variant: GgufVariant,
    repo_id: str,
    range_client: HttpRangeClient,
    warnings: list[str],
) -> GgufHeaderMetadata | None:
    if not variant.files:
        return None
    # Multipart GGUFs put the metadata KV in the first shard.
    target_file = sorted(variant.files, key=lambda f: f.path)[0]
    filename = target_file.path
    if _looks_like_multipart_non_first(filename):
        warnings.append(
            f"Variant {variant.quantization} looks multipart; using first shard "
            f"{PurePosixPath(filename).name} for header read."
        )

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    result = range_reader.fetch_gguf_header(url=url, http_client=range_client)
    warnings.extend(result.warnings)
    return result.header


def _fetch_base_config(
    repo_id: str,
    client: HfClientProtocol,
    warnings: list[str],
) -> ModelConfig | None:
    try:
        path = client.download_small_file(repo_id=repo_id, filename="config.json")
    except ModelNotFoundError:
        warnings.append(f"Base repository {repo_id} does not exist; skipping enrichment.")
        return None
    except ModelAccessDeniedError:
        warnings.append(
            f"Base repository {repo_id} is gated; set HF_TOKEN to enable enrichment."
        )
        return None
    except HuggingFaceUnavailableError as exc:
        warnings.append(f"Could not reach {repo_id} for base config: {exc}.")
        return None
    except LocalAiCheckError as exc:
        warnings.append(f"Could not read base config from {repo_id}: {exc}.")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"config.json for {repo_id} could not be parsed: {exc}.")
        return None

    return _model_config_from_dict(data)


def _looks_like_multipart_non_first(filename: str) -> bool:
    name = PurePosixPath(filename).name
    # GGUF multipart convention: name-00001-of-00003.gguf
    if "-of-" not in name:
        return False
    try:
        index_part = name.split("-of-")[0].split("-")[-1]
        return index_part.isdigit() and int(index_part) != 1
    except (IndexError, ValueError):
        return False


def unresolved_result(warnings: list[str] | None = None) -> EnrichmentResult:
    """Convenience: a zero-info EnrichmentResult for the disabled-flag path."""
    return EnrichmentResult(
        enriched_config=None,
        base_model_resolution=BaseModelResolution(
            repo_id=None,
            source=MetadataSource.UNRESOLVED,
            confidence=EstimationConfidence.UNKNOWN,
            warnings=warnings or [],
        ),
        gguf_header=None,
        warnings=warnings or [],
    )


__all__ = ["RangeClientFactory", "enrich", "unresolved_result"]
