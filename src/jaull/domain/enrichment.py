"""GGUF metadata enrichment domain objects.

Base-model resolution enums and models (``MetadataSource``,
``BaseModelResolution``, ``ConfigurationSource``) live in ``domain.estimation``
so that :class:`~jaull.domain.estimation.MemoryEstimate` can reference
them without a circular import; this module carries only what is specific to
the enrichment pipeline itself.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.estimation import (
    BaseModelResolution,
    ConfigurationSource,
)
from jaull.domain.model import ModelConfig


def per_layer_kv_heads(value: object, layers: int | None) -> tuple[int, ...] | None:
    """The KV heads of every layer, when a header lists them and the list can be trusted.

    Trusted means: one non-negative integer per layer (a layer without attention may say 0) and
    at least one layer with attention. Anything else, such as a list that does not cover every
    layer, is not interpreted: the header is then read to the end instead of being guessed.
    """
    if layers is None or layers < 1 or not isinstance(value, list) or len(value) != layers:
        return None
    heads: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        heads.append(item)
    return tuple(heads) if max(heads) >= 1 else None


class GgufHeaderMetadata(BaseModel):
    """Structured subset of the GGUF KV metadata table we care about."""

    model_config = ConfigDict(frozen=True)

    architecture: str | None = None
    name: str | None = None
    quantization_version: int | None = None
    file_type: int | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    rope_dim: int | None = None
    # `general.size_label` ("14B", "12B-A2.5B"): the size the artifact itself declares.
    size_label: str | None = None
    source_repository: str | None = None
    # False when the read stopped inside the tokenizer: every key the memory estimate reads is
    # present, but keys not read yet are unknown, never absent.
    is_complete: bool = True
    raw_kv: dict[str, object] = Field(default_factory=dict)


class EnrichedConfig(BaseModel):
    """A merged ``ModelConfig`` with per-field provenance."""

    model_config = ConfigDict(frozen=True)

    config: ModelConfig
    sources: dict[str, ConfigurationSource] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)


class EnrichmentResult(BaseModel):
    """Everything the metadata service produces for one GGUF variant."""

    model_config = ConfigDict(frozen=True)

    enriched_config: EnrichedConfig | None
    base_model_resolution: BaseModelResolution
    gguf_header: GgufHeaderMetadata | None = None
    warnings: list[str] = Field(default_factory=list)
