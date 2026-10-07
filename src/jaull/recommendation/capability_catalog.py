"""Explicit, offline evidence lookup; never a recommendation score input."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from jaull.domain.capability_evidence import CapabilityCatalog, CapabilitySubject
from jaull.domain.execution_plans import ModelIdentity, ModelIdentityEvidenceKind
from jaull.recommendation.capability import CapabilitySignal


@dataclass(frozen=True)
class CatalogReadResult:
    status: Literal["loaded", "missing", "invalid"]
    catalog: CapabilityCatalog | None = None
    sha256: str | None = None
    diagnostic: str | None = None


def canonical_repo_is_confirmed(identity: ModelIdentity) -> bool:
    """Confirm an exact repository claim, not a suffix-based grouping key."""
    canonical = identity.canonical_repo_id
    if canonical is None:
        return False
    return any(
        item.kind in {
            ModelIdentityEvidenceKind.REPOSITORY_ID,
            ModelIdentityEvidenceKind.BASE_MODEL_METADATA,
        }
        and item.value.strip().casefold() == canonical.strip().casefold()
        for item in identity.evidence
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON object keys")
    return result


def default_catalog_path() -> Path:
    """The catalogue shipped with the package, beside this module."""
    return Path(__file__).with_name("capability_catalog.json")


def load_capability_catalog(path: Path) -> CatalogReadResult:
    """Read optional data once; absence/corruption cannot break recommending."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return CatalogReadResult("missing", diagnostic="Capability catalog is missing.")
    except OSError as exc:
        return CatalogReadResult(
            "invalid", diagnostic=f"Capability catalog could not be read ({type(exc).__name__})."
        )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (ValueError, RecursionError):
        return CatalogReadResult(
            "invalid", sha256=digest,
            diagnostic=(
                "Invalid capability catalog JSON: malformed data, excessive nesting "
                "or duplicate object keys."
            ),
        )
    try:
        catalog = CapabilityCatalog.model_validate(data)
    except ValidationError as exc:
        # Surface locations/types, not arbitrary input values or credentials.
        detail = "; ".join(
            f'{error["loc"]}: {error["type"]}'
            for error in exc.errors(include_url=False, include_context=False, include_input=False)
        )
        return CatalogReadResult(
            "invalid", sha256=digest, diagnostic=f"Invalid capability catalog: {detail}"
        )
    return CatalogReadResult("loaded", catalog=catalog, sha256=digest)


def attach_capability_evidence(
    signal: CapabilitySignal,
    subject: CapabilitySubject,
    result: CatalogReadResult,
) -> CapabilitySignal:
    """Attach only exact subject matches, preserving every scoring field.

    No base-model/family lookup, revision resolution, quantization inheritance,
    network access or claims about the current execution plan occur here.
    """
    catalog = result.catalog
    evaluations = (
        tuple(item for item in catalog.evaluations if item.subject == subject)
        if result.status == "loaded" and catalog is not None
        else ()
    )
    diagnostics = []
    if result.diagnostic is not None:
        diagnostics.append(result.diagnostic)
    elif not evaluations:
        diagnostics.append("No exact capability evidence match; absence is not poor quality.")
    elif subject.revision is None:
        diagnostics.append(
            "Evidence is not pinned to a model revision; it is not revision-confirmed."
        )
    return replace(
        signal,
        evaluation_evidence=evaluations,
        evidence_catalog_version=catalog.catalog_version if catalog is not None else None,
        evidence_catalog_sha256=result.sha256,
        evidence_diagnostics=tuple(diagnostics),
    )
