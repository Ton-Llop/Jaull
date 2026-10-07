"""Bibliographic model references, never a match to an execution protocol."""

from __future__ import annotations

import json

from jaull.domain.capability_evidence import CapabilityEvaluation
from jaull.domain.execution_plans import ModelIdentity, ModelIdentityEvidenceKind
from jaull.recommendation.capability_catalog import CatalogReadResult, canonical_repo_is_confirmed


def published_reference_text(
    identity: ModelIdentity, result: CatalogReadResult | None,
) -> tuple[str, str]:
    """Display exact-repository references without inheriting their scores.

    Unlike attach_capability_evidence, this is a bibliography: each record's
    variant and protocol remain visible, not asserted to match the current plan.
    """
    if result is None:
        return "No published references loaded.", ""
    if result.status != "loaded" or result.catalog is None:
        return result.diagnostic or "Published capability catalog unavailable.", ""
    if not canonical_repo_is_confirmed(identity):
        return "Published references unavailable: model lineage is not confirmed.", ""
    assert identity.canonical_repo_id is not None
    repo = identity.canonical_repo_id
    entries = tuple(
        entry for entry in result.catalog.evaluations
        if entry.repo_id.casefold() == repo.strip().casefold()
    )
    if not entries:
        return "No published evaluations for this model. Absence is not poor quality.", ""
    declared_base = any(
        item.kind is ModelIdentityEvidenceKind.BASE_MODEL_METADATA
        and item.value.strip().casefold() == repo.strip().casefold()
        for item in identity.evidence
    )
    lines = [
        repo,
        "Base-model relationship declared in metadata." if declared_base
        else "Direct repository reference.",
        "Historical model results; not measured on this artifact "
        "or verified for the current protocol.",
    ]
    for entry in entries:
        lines.extend(["", *_entry_lines(entry)])
    details = [
        f"Catalog version: {result.catalog.catalog_version}",
        f"Catalog SHA256: {result.sha256 or 'unknown'}",
        "Missing conditions remain unknown. These references do not determine ranking.",
        *[json.dumps(entry.model_dump(mode="json"), indent=2) for entry in entries],
    ]
    return "\n".join(lines), "\n\n".join(details)


def _entry_lines(entry: CapabilityEvaluation) -> list[str]:
    source_kind = (
        "publisher-reported" if entry.source_kind == "publisher_reported" else "independent"
    )
    value = f"{entry.value:g}%" if entry.unit == "percent" else f"{entry.value:g} {entry.unit}"
    return [
        f"{entry.benchmark} / {entry.benchmark_version or 'version unknown'}: "
        f"{value} {entry.metric} ({entry.task}; {source_kind})",
        f"Variant: {entry.variant or 'unknown'}; reasoning mode: "
        f"{entry.protocol.reasoning_mode or 'unknown'}",
        f"Revision: {entry.revision or 'unknown'}; precision: {entry.precision or 'unknown'}",
        f"Source: {entry.source} (assessed {entry.date})",
    ]
