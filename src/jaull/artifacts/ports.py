"""Protocols the artifact service depends on.

Splitting the resolver behind a Protocol keeps the service layer free of any
Hugging Face dependency: tests supply a hand-written fake, production wires
in ``jaull.huggingface.artifact_resolver.HuggingFaceArtifactResolver``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from jaull.domain.artifacts import ModelArtifact


class ArtifactResolverProtocol(Protocol):
    def resolve(
        self,
        repo_id: str,
        *,
        quantization: str | None,
        revision: str | None = None,
    ) -> ModelArtifact: ...


@dataclass(frozen=True)
class PublishedVersion:
    """A digest a repository has published for one file, with the newest commit that had it."""

    sha256: str
    revision: str
    date: datetime


class ArtifactHistoryProtocol(Protocol):
    def find_version(
        self, repo_id: str, filename: str, sha256: str, *, limit: int,
    ) -> PublishedVersion | None:
        """The newest of the last ``limit`` commits whose copy of the file has ``sha256``."""
        ...


__all__ = ["ArtifactHistoryProtocol", "ArtifactResolverProtocol", "PublishedVersion"]
