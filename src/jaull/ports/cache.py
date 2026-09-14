"""Cache ports used by application services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jaull.domain.candidates import ModelCandidate
from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.model import GgufVariant, ModelAnalysis


@dataclass
class ModelAnalysisCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    read_errors: int = 0
    write_errors: int = 0
    unsupported_schema: int = 0
    expired: int = 0


class ModelAnalysisCacheProtocol(Protocol):
    stats: ModelAnalysisCacheStats

    def get(self, candidate: ModelCandidate) -> ModelAnalysis | None: ...
    def put(self, candidate: ModelCandidate, analysis: ModelAnalysis) -> None: ...


@dataclass
class GgufHeaderCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    read_errors: int = 0
    write_errors: int = 0
    unsupported_schema: int = 0
    expired: int = 0


class GgufHeaderCacheProtocol(Protocol):
    """Persistent cache for successfully parsed GGUF artifact headers."""

    stats: GgufHeaderCacheStats

    def get(
        self,
        analysis: ModelAnalysis,
        variant: GgufVariant,
    ) -> GgufHeaderMetadata | None: ...

    def put(
        self,
        analysis: ModelAnalysis,
        variant: GgufVariant,
        header: GgufHeaderMetadata,
    ) -> None: ...


__all__ = [
    "GgufHeaderCacheProtocol",
    "GgufHeaderCacheStats",
    "ModelAnalysisCacheProtocol",
    "ModelAnalysisCacheStats",
]
