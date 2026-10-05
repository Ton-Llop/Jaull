"""Audit how discovery's own candidates reach a repository the catalogue names.

Published results are attached only when the canonical repository is confirmed
— the candidate is that repository, or its metadata names it as the base model.
A repository arrived at by stripping a suffix off a name is a guess, and guesses
do not inherit a publisher's numbers.

That rule is only as useful as the lineage that real candidates actually carry,
and nothing in the repository records it. This runs one real guided search, takes
the first N candidates it produced, and classifies each one, so the question
"why is there no evidence here" has an answer per candidate rather than a
suspicion.

It is a sample, not a coverage measurement: one request, one day, one search
strategy. It reads metadata only — no weights are downloaded and nothing is
evaluated.

    uv run python -m scripts.audit_published_evidence_lineage --limit 12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jaull.bootstrap.container import ServiceContainer
from jaull.domain.execution_plans import ModelIdentityEvidenceKind
from jaull.domain.requirements import (
    CommercialUse,
    ConcurrencyLevel,
    RecommendationPriority,
    UseCase,
    UserAnswers,
    WorkloadMode,
)
from jaull.execution_plans.service import resolve_model_identity
from jaull.hardware.detector import detect_hardware
from jaull.recommendation.capability_catalog import (
    default_catalog_path,
    load_capability_catalog,
)
from jaull.workflow.orchestrator import run_workflow

#: The four ways a candidate can relate to the repository a publisher measured.
DIRECT = "direct_repository"
DECLARED = "declared_base_model"
GUESSED = "name_heuristic_only"
NONE = "no_lineage_evidence"


def _answers() -> UserAnswers:
    """An ordinary request, so the candidates are the ordinary ones."""
    return UserAnswers(
        use_case=UseCase.GENERAL_CHAT,
        workload_mode=WorkloadMode.INTERACTIVE,
        priority=RecommendationPriority.BALANCED,
        languages=["English"],
        concurrency=ConcurrencyLevel.SINGLE,
        commercial_use=CommercialUse.NO_PREFERENCE,
    )


def _lineage(identity: Any) -> str:
    """Which of the four relations this identity's evidence actually supports."""
    canonical = (identity.canonical_repo_id or "").strip().casefold()
    if not canonical:
        return NONE
    kinds = set()
    for item in identity.evidence:
        if item.value.strip().casefold() == canonical:
            kinds.add(item.kind)
    if ModelIdentityEvidenceKind.REPOSITORY_ID in kinds:
        return DIRECT
    if ModelIdentityEvidenceKind.BASE_MODEL_METADATA in kinds:
        return DECLARED
    if any(
        item.kind is ModelIdentityEvidenceKind.NAME_HEURISTIC
        for item in identity.evidence
    ):
        return GUESSED
    return NONE


def _catalogue_subjects() -> dict[str, set[str | None]]:
    """Every repository the catalogue names, and the variants it names it with."""
    result = load_capability_catalog(default_catalog_path())
    subjects: dict[str, set[str | None]] = {}
    if result.status != "loaded" or result.catalog is None:
        return subjects
    for evaluation in result.catalog.evaluations:
        subject = evaluation.subject
        subjects.setdefault(subject.repo_id.strip().casefold(), set()).add(subject.variant)
    return subjects


def run(limit: int, output: Path | None) -> None:
    services = ServiceContainer.default()
    hardware = detect_hardware()
    state = run_workflow(_answers(), hardware, services)
    subjects = _catalogue_subjects()

    rows: list[dict[str, Any]] = []
    for candidate in state.candidates[:limit]:
        identity = resolve_model_identity(candidate=candidate)
        canonical = identity.canonical_repo_id
        lineage = _lineage(identity)
        key = (canonical or "").strip().casefold()
        variants = subjects.get(key)
        if variants is None:
            catalogue = "no_catalogue_entry"
        elif identity.variant in variants:
            catalogue = "entry_matches"
        else:
            catalogue = "entry_variant_mismatch"
        rows.append(
            {
                "repo_id": candidate.repo_id,
                "declared_base_model": candidate.base_model_repo_id,
                "canonical_repo_id": canonical,
                "variant": identity.variant,
                "lineage": lineage,
                "catalogue": catalogue,
                # Why this candidate shows nothing, in the order the code decides it.
                "published_evidence_blocked_by": (
                    "unconfirmed_lineage"
                    if lineage in {GUESSED, NONE}
                    else catalogue
                    if catalogue != "entry_matches"
                    else None
                ),
            }
        )

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "sample_size": len(rows),
        "limitations": [
            "One request, one search strategy, one day: a sample, not coverage.",
            "Metadata only. No weights were downloaded and nothing was evaluated.",
            "Lineage is read from discovery's own identity resolution, not re-derived.",
        ],
        "request": _answers().model_dump(mode="json"),
        "search_queries": state.search_queries,
        "candidates_found": len(state.candidates),
        "warnings": state.warnings,
        "lineage_counts": dict(Counter(row["lineage"] for row in rows)),
        "catalogue_counts": dict(Counter(row["catalogue"] for row in rows)),
        "blocked_by_counts": dict(
            Counter(str(row["published_evidence_blocked_by"]) for row in rows)
        ),
        "candidates": rows,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.limit, args.output)


if __name__ == "__main__":
    main()
