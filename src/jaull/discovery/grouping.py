"""Group candidates that are really the same model into families.

A search for one model routinely returns the original repository plus several
third-party GGUF conversions. Presenting those as separate recommendations would
be noise, so they are collapsed into a family and the ranker emits at most one
entry per family.

Grouping is **evidence-based**: two repositories are only related when the model
card says so via ``base_model``. Similar names are not enough — ``Qwen2.5-7B``
and ``Qwen2.5-7B-Coder`` are different models, and guessing from a shared prefix
would silently merge them.
"""

from __future__ import annotations

from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate


def family_key(candidate: ModelCandidate) -> str:
    """Return the identifier of the family this candidate belongs to.

    The declared base model wins when present; otherwise the repository is its
    own family. No name-similarity heuristics.
    """
    base = candidate.base_model_repo_id
    if base:
        return base.strip().lower()
    return candidate.repo_id.strip().lower()


def group_candidates(
    candidates: list[ModelCandidate],
) -> dict[str, list[ModelCandidate]]:
    """Bucket candidates by family, preserving input order within each bucket."""
    families: dict[str, list[ModelCandidate]] = {}
    for candidate in candidates:
        families.setdefault(family_key(candidate), []).append(candidate)
    return families


def collapse_families(
    evaluated: list[EvaluatedCandidate],
    score_of: dict[str, float],
) -> tuple[list[EvaluatedCandidate], dict[str, list[str]]]:
    """Keep the best-scoring member of each family.

    Returns the survivors plus, for each kept ``repo_id``, the sibling repos that
    were folded into it so the report can still show them as technical
    alternatives.
    """
    best: dict[str, EvaluatedCandidate] = {}
    siblings: dict[str, list[str]] = {}
    order: list[str] = []

    for item in evaluated:
        key = family_key(item.candidate)
        if key not in best:
            best[key] = item
            order.append(key)
            continue
        incumbent = best[key]
        if score_of.get(item.repo_id, 0.0) > score_of.get(incumbent.repo_id, 0.0):
            best[key] = item

    for item in evaluated:
        key = family_key(item.candidate)
        winner = best[key]
        if item.repo_id == winner.repo_id:
            continue
        siblings.setdefault(winner.repo_id, []).append(item.repo_id)

    return [best[key] for key in order], siblings


__all__ = ["collapse_families", "family_key", "group_candidates"]
