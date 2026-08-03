from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from jaull.domain.estimation import MemoryEstimate
from jaull.domain.runtime import RuntimeName
from jaull.recommendation.models import ModelRecommendation
from jaull.recommendation.tier import (
    TIER_HEADINGS,
    RecommendationTier,
)
from jaull.tui.widgets.assessment_badge import AssessmentBadge
from jaull.tui.widgets.compact_score_strip import CompactScoreStrip

_STATUS_LABEL = {
    "comfortable": "Comfortable",
    "compatible": "Compatible",
    "tight": "Tight",
    "offloading_required": "Needs offloading",
    "insufficient": "Insufficient",
    "unknown": "Unknown",
}

_TIER_CLASS = {
    RecommendationTier.BEST_MATCH: "tier-best-match",
    RecommendationTier.RECOMMENDED: "tier-recommended",
    RecommendationTier.CLOSEST_OPTION: "tier-closest-option",
    RecommendationTier.BEST_EFFORT: "tier-best-effort",
}


class RecommendationCard(Vertical):
    """One recommendation: what it is, how it fits, and why it was chosen."""

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._rec = recommendation
        self.add_class(
            "recommendation-card" if recommendation.is_primary else "card"
        )

    def compose(self) -> ComposeResult:
        rec = self._rec
        tier = _tier_from(rec.tier)
        heading = self._heading(tier)
        yield Static(
            heading,
            classes=f"card-title {_TIER_CLASS.get(tier, 'tier-recommended')}",
        )
        yield Static(f"[b]{rec.repo_id}[/b]")

        for label, value in self._facts():
            yield Static(f"[bold]{label.ljust(20)}[/bold] {value}")

        yield AssessmentBadge(rec.status)
        yield CompactScoreStrip(rec.score)

        estimate = rec.evaluated.memory_estimate
        if estimate is not None and estimate.runtime_recommendation is not None:
            yield _RuntimeBlock(estimate)

        if rec.series_alternatives:
            yield _SeriesLadder(rec)

        if rec.reasons:
            yield Static("Why this model?", classes="card-title")
            for reason in rec.reasons:
                yield Static(f"✓ {reason}", classes="reason-line")

        if rec.warnings:
            yield Static("Limitations and warnings", classes="card-title")
            for warning in rec.warnings:
                yield Static(f"⚠ {warning}", classes="warning-line")

        if rec.score.unmet_requirements:
            yield Static("Unmet requirements", classes="card-title")
            for label in rec.score.unmet_requirements:
                yield Static(f"✗ {label}", classes="warning-line")

        if rec.related_repositories:
            related = ", ".join(rec.related_repositories[:3])
            yield Static(
                f"[dim]Same model, other formats: {related}[/dim]",
                classes="text-muted",
            )

    # ------------------------------------------------------------------
    def _heading(self, tier: RecommendationTier) -> str:
        if not self._rec.is_primary:
            return self._rec.alternative_label or f"Alternative {self._rec.rank - 1}"
        return TIER_HEADINGS[tier]

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
            if config.concurrent_users > 1:
                rows.append(("Concurrent users", str(config.concurrent_users)))
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
        rows.append(("Runtime", _runtime_summary(estimate)))
        if estimate is not None and estimate.total_bytes is not None:
            rows.append(("Estimated total", _gib(estimate.total_bytes)))
        return rows


class _RuntimeBlock(Vertical):
    DEFAULT_CLASSES = "card"

    def __init__(self, estimate: MemoryEstimate) -> None:
        super().__init__()
        self._estimate = estimate

    def compose(self) -> ComposeResult:
        runtime = self._estimate.runtime_recommendation
        assert runtime is not None
        yield Static("How to run it", classes="card-title")
        if runtime.runtime is RuntimeName.UNKNOWN:
            yield Static(
                "No runtime yet for repositories of this type.",
                classes="text-muted",
            )
            return
        if runtime.command_preview:
            yield Static(f"$ {runtime.command_preview}", classes="runtime-code")
        if runtime.python_snippet:
            yield Static(runtime.python_snippet, classes="runtime-code")
        for reason in runtime.reasons[:2]:
            yield Static(f"- {reason}", classes="text-muted")


class _SeriesLadder(Vertical):
    DEFAULT_CLASSES = "card"

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._rec = recommendation

    def compose(self) -> ComposeResult:
        yield Static("Series ladder", classes="card-title")
        siblings = self._rec.series_alternatives
        ladder = " · ".join(alt.parameter_label for alt in siblings)
        yield Static(f"Same series, other sizes: {ladder}")
        for alt in siblings[:4]:
            yield Static(f"  · {alt.repo_id} — {alt.note}", classes="text-muted")


def _tier_from(value: str) -> RecommendationTier:
    try:
        return RecommendationTier(value)
    except ValueError:
        return RecommendationTier.RECOMMENDED


def _runtime_summary(estimate: MemoryEstimate | None) -> str:
    if estimate is None or estimate.runtime_recommendation is None:
        return "not detected"
    return estimate.runtime_recommendation.runtime.value


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


__all__ = ["RecommendationCard"]
