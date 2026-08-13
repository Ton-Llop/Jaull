"""Shared artifact preparation helpers for TUI execution workflows."""

from __future__ import annotations

from collections.abc import Callable

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.recommendation.models import ModelRecommendation


def prepare_recommendation_artifact(
    *,
    advisor: AdvisorService,
    recommendation: ModelRecommendation,
    report_step: Callable[[str], None],
) -> ModelArtifact:
    """Resolve, download, and verify the artifact for a recommendation."""

    config = recommendation.evaluated.selected_configuration
    quantization = config.quantization if config is not None else None
    report_step(
        f"Resolve artifact for {recommendation.repo_id}"
        + (f" ({quantization})" if quantization else ""),
    )
    artifact = advisor.resolve_artifact(
        repo_id=recommendation.repo_id,
        quantization=quantization,
        revision=None,
    )
    if not artifact.is_downloaded:
        report_step("Downloading artifact")
        artifact = advisor.download_artifact(artifact)
    else:
        report_step(f"Reuse local artifact {artifact.filename}")
    report_step("Verifying artifact")
    artifact = advisor.verify_artifact(artifact)
    report_step("Model ready")
    return artifact


__all__ = ["prepare_recommendation_artifact"]
