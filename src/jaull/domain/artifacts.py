"""Concrete model artifact — the resolved file behind an abstract recommendation.

A ``ModelArtifact`` is the bridge between "the user wants model X at
quantization Q" and "there is a file at this local path, of this size, with
this hash". It travels from the ``AdvisorService`` up to the CLI/TUI, so it
lives in ``domain/`` to satisfy the layering rule established in Fase 6.
"""

from __future__ import annotations

from pathlib import Path

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


__all__ = ["ModelArtifact"]
