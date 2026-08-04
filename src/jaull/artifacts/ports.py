"""Protocols the artifact service depends on.

Splitting the resolver behind a Protocol keeps the service layer free of any
Hugging Face dependency: tests supply a hand-written fake, production wires
in ``jaull.huggingface.artifact_resolver.HuggingFaceArtifactResolver``.
"""

from __future__ import annotations

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


__all__ = ["ArtifactResolverProtocol"]
