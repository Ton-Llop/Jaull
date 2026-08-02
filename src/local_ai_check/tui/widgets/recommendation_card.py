from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from local_ai_check.recommendation.models import ModelRecommendation
from local_ai_check.tui.widgets.assessment_badge import AssessmentBadge

_STATUS_LABEL = {
    "comfortable": "Comfortable",
    "compatible": "Compatible",
    "tight": "Tight",
    "offloading_required": "Needs offloading",
    "insufficient": "Insufficient",
    "unknown": "Unknown",
}


class RecommendationCard(Vertical):
    """One recommendation: what it is, how it fits, and why it was chosen."""

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._rec = recommendation
        self.add_class("recommendation-card" if recommendation.is_primary else "card")

    def compose(self) -> ComposeResult:
        rec = self._rec
        heading = (
            "BEST MATCH"
            if rec.is_primary
            else (rec.alternative_label or f"Alternative {rec.rank - 1}")
        )
        yield Static(heading, classes="card-title")
        yield Static(f"[b]{rec.repo_id}[/b]")

        for label, value in self._facts():
            yield Static(f"[bold]{label.ljust(20)}[/bold] {value}")

        yield AssessmentBadge(rec.status)

        if rec.reasons:
            yield Static("Why this model?", classes="card-title")
            for reason in rec.reasons:
                yield Static(f"✓ {reason}", classes="reason-line")

        if rec.warnings:
            yield Static("Limitations and warnings", classes="card-title")
            for warning in rec.warnings:
                yield Static(f"⚠ {warning}", classes="warning-line")

        if rec.related_repositories:
            related = ", ".join(rec.related_repositories[:3])
            yield Static(
                f"[dim]Same model, other formats: {related}[/dim]",
                classes="text-muted",
            )

    def _facts(self) -> list[tuple[str, str]]:
        rec = self._rec
        config = rec.evaluated.selected_configuration
        estimate = rec.evaluated.memory_estimate

        rows: list[tuple[str, str]] = [
            ("Repository", rec.repo_id),
        ]
        if rec.evaluated.candidate.repository_type is not None:
            rows.append(
                ("Selected format", rec.evaluated.candidate.repository_type.value)
            )
        if config is not None:
            if config.quantization:
                rows.append(("Quantization", config.quantization))
            if config.precision:
                rows.append(("Precision", config.precision.value))
            rows.append(("Suggested context", f"{config.context_length} tokens"))
        if rec.evaluated.compatibility is not None:
            rows.append(
                (
                    "Target device",
                    rec.evaluated.compatibility.effective_device.value,
                )
            )
        rows.append(
            ("Compatibility", _STATUS_LABEL.get(rec.status.value, rec.status.value))
        )
        rows.append(("Confidence", rec.confidence.value))
        rows.append(
            (
                "License",
                f"{rec.evaluated.candidate.license or 'not declared'} "
                f"({rec.license_category.value.replace('_', ' ')})",
            )
        )
        rows.append(("Score", f"{rec.score.out_of_100}/100"))
        if estimate is not None and estimate.total_bytes is not None:
            rows.append(("Estimated total", _gib(estimate.total_bytes)))
        return rows


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


__all__ = ["RecommendationCard"]
