"""Concrete model artifact — the resolved file behind an abstract recommendation.

A ``ModelArtifact`` is the bridge between "the user wants model X at
quantization Q" and "there is a file at this local path, of this size, with
this hash". It travels from the ``AdvisorService`` up to the CLI/TUI, so it
lives in ``domain/`` to satisfy the layering rule established in Fase 6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict


class ModelArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_id: str
    revision: str
    filename: str
    format: str
    quantization: str | None = None
    size_bytes: int | None = None
    local_path: Path | None = None
    sha256: str | None = None
    is_downloaded: bool = False
    is_verified: bool = False


class ArtifactIdentity(BaseModel):
    """What names the artifact, with no machine-local state attached.

    ``ModelArtifact`` mixes identity with the state of one machine's copy:
    ``local_path``, ``is_downloaded`` and ``is_verified`` say where the file
    landed here, not which file it is. ``size_bytes`` and ``sha256`` describe
    the same file but are frequently unknown (the Transformers path never
    resolves a digest), so they corroborate identity rather than establish it —
    see :func:`artifact_identity_matches`.
    """

    model_config = ConfigDict(frozen=True)

    repo_id: str
    revision: str
    filename: str
    format: str
    quantization: str | None = None

    @classmethod
    def of(cls, artifact: ModelArtifact) -> Self:
        return cls(
            repo_id=artifact.repo_id,
            revision=artifact.revision,
            filename=artifact.filename,
            format=artifact.format,
            quantization=artifact.quantization,
        )


def artifact_identity_matches(left: ModelArtifact, right: ModelArtifact) -> bool:
    """Do these two references name the same artifact?

    Identity fields must be equal. ``size_bytes`` and ``sha256`` only
    corroborate: a value known on one side and absent on the other is not a
    mismatch, because absence is common and is not evidence of difference. Two
    *known* and different values are a mismatch — that is a real conflict.

    Matching metadata is not proof of identical file contents. A plan resolved
    from discovery carries no digest, and a mutable revision such as ``main``
    is not an immutable identity.
    """

    if ArtifactIdentity.of(left) != ArtifactIdentity.of(right):
        return False
    return all(
        a is None or b is None or a == b
        for a, b in ((left.size_bytes, right.size_bytes), (left.sha256, right.sha256))
    )


__all__ = ["ArtifactIdentity", "ModelArtifact", "artifact_identity_matches"]
