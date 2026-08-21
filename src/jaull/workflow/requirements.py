"""Compatibility shim for requirement normalization."""

from __future__ import annotations

from jaull.application.requirements import (
    build_requirements,
    normalize_language,
    normalize_languages,
)

__all__ = ["build_requirements", "normalize_language", "normalize_languages"]
