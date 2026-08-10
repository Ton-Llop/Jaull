"""Confirmed vs theoretical artifact information for a candidate.

Discovery already classifies a repository as ``GGUF``/``TRANSFORMERS``/…, but
that says nothing about *which* pre-quantized artifact (if any) is available.
An AWQ repo with real weight files is a different beast from a bare
Transformers repo that the recommender chose to *estimate* at int4: both live
under ``RepositoryType.TRANSFORMERS`` but only the first is safe to promote
as a top pick on hardware that supports it. This module exposes that
distinction as first-class data.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ArtifactFormat(StrEnum):
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
    BITSANDBYTES = "bitsandbytes"
    COMPRESSED_TENSORS = "compressed_tensors"
    NATIVE = "native"
    UNKNOWN = "unknown"


class ArtifactConfirmation(StrEnum):
    """How much evidence backs the claim that this artifact is real."""

    # File(s) or explicit metadata prove the artifact exists in the repo.
    CONFIRMED = "confirmed"
    # Strong secondary signal (tag, id suffix) suggests the artifact exists.
    INFERRED = "inferred"
    # The recommender picked a precision the repo does not publish.
    THEORETICAL = "theoretical"
    UNKNOWN = "unknown"


class ArtifactProfile(BaseModel):
    """A structured description of the recommended artifact."""

    model_config = ConfigDict(frozen=True)

    format: ArtifactFormat
    quantization: str | None = None
    confirmation: ArtifactConfirmation = ArtifactConfirmation.UNKNOWN
    reasons: list[str] = Field(default_factory=list)


__all__ = ["ArtifactConfirmation", "ArtifactFormat", "ArtifactProfile"]
