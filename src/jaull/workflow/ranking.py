"""Compatibility shim for recommendation application service."""

from __future__ import annotations

from jaull.application.recommendation.service import (
    enrich_candidate_features,
    recommend,
    score_all,
)
from jaull.recommendation import ranker

__all__ = ["enrich_candidate_features", "ranker", "recommend", "score_all"]
