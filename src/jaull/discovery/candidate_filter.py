"""Preliminary filtering of search results, before any expensive inspection.

Two rules shape this module:

* Reject only what is genuinely unusable — private repos, the wrong modality, a
  license that contradicts a hard requirement.
* Never reject a model merely for having thin metadata. Incomplete cards are the
  norm on the Hub; those candidates continue with a recorded penalty and a lower
  confidence instead of disappearing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from jaull.discovery.models import ModelCandidate
from jaull.domain.estimation import EstimationConfidence
from jaull.recommendation import policies as rec_policies
from jaull.workflow import policies
from jaull.workflow.models import UserRequirements

# Tags that mean "this is not a plain text-generation model". This commit only
# supports text, so these are rejections rather than penalties.
_MULTIMODAL_TAGS: frozenset[str] = frozenset(
    {
        "image-to-text",
        "text-to-image",
        "text-to-speech",
        "text-to-video",
        "automatic-speech-recognition",
        "audio-classification",
        "image-classification",
        "visual-question-answering",
        "image-text-to-text",
        "video-text-to-text",
        "multimodal",
        "vision",
    }
)

# An adapter needs a base model to mean anything; without one there is nothing
# to estimate.
_ADAPTER_TAGS: frozenset[str] = frozenset({"peft", "lora", "adapter", "adapter-transformers"})


@dataclass
class FilterOutcome:
    """What survived, and why the rest did not."""

    kept: list[ModelCandidate] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def reject(self, repo_id: str, reason: str) -> None:
        self.rejected.append((repo_id, reason))


def deduplicate(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    """Collapse repeated sightings of a repo, preserving first-seen order.

    A repository showing up in several queries is a signal, not a reason to
    inspect it twice — the merged candidate keeps every query label.
    """
    merged: dict[str, ModelCandidate] = {}
    order: list[str] = []
    for candidate in candidates:
        if not candidate.repo_id:
            continue
        existing = merged.get(candidate.repo_id)
        if existing is None:
            merged[candidate.repo_id] = candidate
            order.append(candidate.repo_id)
        else:
            merged[candidate.repo_id] = existing.merged_with(candidate)
    return [merged[repo_id] for repo_id in order]


def filter_candidates(
    candidates: list[ModelCandidate], requirements: UserRequirements
) -> FilterOutcome:
    """Apply the preliminary rules, returning survivors and rejection reasons."""
    outcome = FilterOutcome()

    for candidate in candidates:
        rejection = _rejection_reason(candidate, requirements)
        if rejection is not None:
            outcome.reject(candidate.repo_id, rejection)
            continue
        outcome.kept.append(_apply_penalties(candidate))

    return outcome


def _rejection_reason(
    candidate: ModelCandidate, requirements: UserRequirements
) -> str | None:
    if candidate.private:
        return "Repository is private."
    if candidate.gated:
        return "Repository is gated and cannot be inspected without access."

    tags = {tag.lower() for tag in candidate.tags}

    if tags & _MULTIMODAL_TAGS:
        return "Model is multimodal; this version only recommends text models."

    pipeline = candidate.pipeline_tag
    if pipeline and pipeline != policies.TEXT_GENERATION_PIPELINE:
        return f"Pipeline {pipeline!r} is not text generation."

    if tags & _ADAPTER_TAGS and not candidate.base_model_repo_id:
        return "Adapter without a resolvable base model."

    if requirements.commercial_use_required:
        category = rec_policies.classify_license(candidate.license)
        if category is rec_policies.LicenseCategory.COMMERCIAL_RESTRICTED:
            return (
                f"License {candidate.license!r} is not generally suitable for "
                "commercial use."
            )

    return None


def _apply_penalties(candidate: ModelCandidate) -> ModelCandidate:
    """Keep the candidate but record what makes it less trustworthy."""
    penalties = list(candidate.penalties)
    confidence = candidate.metadata_confidence

    if not candidate.license:
        penalties.append("No license declared in the model card.")
        confidence = _lower(confidence)
    if not candidate.languages:
        penalties.append("No languages declared in the model card.")
        confidence = _lower(confidence)
    if candidate.pipeline_tag is None:
        penalties.append("No pipeline tag declared; task match is inferred from tags.")
        confidence = _lower(confidence)

    if not penalties:
        return candidate
    return candidate.model_copy(
        update={"penalties": penalties, "metadata_confidence": confidence}
    )


# Parameter counts are conventionally encoded in the repository name
# ("Qwen2.5-Coder-7B-Instruct"). Matched case-insensitively on a word boundary
# so "7B" is found but "B7" or a bare "7" is not.
_PARAM_COUNT_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def parameter_count_hint(repo_id: str) -> float | None:
    """Best-effort parameter count in billions, read from the repository name.

    A naming *heuristic*, used only to decide which candidates are worth
    inspecting. It never reaches a reported figure: once a candidate is
    inspected, every number comes from real file sizes and configuration
    metadata. Returns ``None`` when the name says nothing.
    """
    matches = _PARAM_COUNT_RE.findall(repo_id)
    if not matches:
        return None
    try:
        # Largest match wins: "Qwen3-Coder-30B-A3B" is a 30B model.
        return max(float(value) for value in matches)
    except ValueError:  # pragma: no cover - regex already constrains the shape
        return None


def shortlist(
    candidates: list[ModelCandidate],
    requirements: UserRequirements,
    limit: int,
    budget_bytes: int | None = None,
) -> list[ModelCandidate]:
    """Choose which candidates are worth the cost of deep inspection.

    Deep inspection is the expensive stage, so the budget has to go to the most
    plausible repositories. This is a *cheap* pre-ranking over search metadata
    only — the real, hardware-aware ranking happens later and can still demote
    anything selected here.

    Popularity carries real weight at this stage, and deliberately so: it is the
    only pre-inspection signal separating a maintained release from a
    19-download experiment. It stays a secondary signal in the final ranking,
    where hardware fit dominates.

    When ``budget_bytes`` is known, models whose name implies they cannot
    possibly fit are pushed down. Without this the whole inspection budget goes
    to the most-downloaded 30B repositories on a machine that can only run a
    3B one, and the run returns almost nothing.
    """
    preferred = requirements.preferred_formats[:1]
    budget_gib = budget_bytes / 1024**3 if budget_bytes else None

    def key(candidate: ModelCandidate) -> tuple[float, str]:
        score = math.log1p(max(0, candidate.downloads))
        score += 0.5 * math.log1p(max(0, candidate.likes))
        # A repository matching the format this machine can actually run is
        # worth inspecting ahead of one that probably will not fit.
        tags = {tag.lower() for tag in candidate.tags}
        if preferred and preferred[0] in tags:
            score += 2.0
        # Appearing in several independent queries is corroboration.
        score += 0.75 * (len(candidate.source_queries) - 1)
        score -= 0.5 * len(candidate.penalties)
        score += _size_bonus(candidate, budget_gib)
        return (-score, candidate.repo_id)

    return sorted(candidates, key=key)[:limit]


def _size_bonus(candidate: ModelCandidate, budget_gib: float | None) -> float:
    """Reward names that suggest the model fits; penalise ones that cannot."""
    if budget_gib is None:
        return 0.0
    billions = parameter_count_hint(candidate.repo_id)
    if billions is None:
        return 0.0
    # ~0.6 GiB per billion parameters at 4-bit, the smallest quantization we
    # would realistically recommend. Anything above that cannot fit at all.
    smallest_plausible_gib = billions * 0.6
    if smallest_plausible_gib > budget_gib:
        return -6.0
    if smallest_plausible_gib > budget_gib * 0.7:
        return -1.5
    return 1.5


_CONFIDENCE_LADDER: tuple[EstimationConfidence, ...] = (
    EstimationConfidence.HIGH,
    EstimationConfidence.MEDIUM,
    EstimationConfidence.LOW,
    EstimationConfidence.UNKNOWN,
)


def _lower(confidence: EstimationConfidence) -> EstimationConfidence:
    index = _CONFIDENCE_LADDER.index(confidence)
    return _CONFIDENCE_LADDER[min(index + 1, len(_CONFIDENCE_LADDER) - 1)]


__all__ = [
    "FilterOutcome",
    "deduplicate",
    "filter_candidates",
    "parameter_count_hint",
    "shortlist",
]
