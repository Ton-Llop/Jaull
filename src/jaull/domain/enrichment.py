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
    source_repository: str | None = None
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
