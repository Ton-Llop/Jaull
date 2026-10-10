"""Offline references are a bibliography, not artifact or protocol validation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jaull.advisor.service import AdvisorService
from jaull.domain.capability_evidence import CapabilityCatalog
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution_plans import (
    ModelIdentity,
    ModelIdentityEvidence,
    ModelIdentityEvidenceKind,
)
from jaull.presentation.published_evaluations import published_reference_text
from jaull.recommendation import capability_catalog
from jaull.recommendation.capability_catalog import CatalogReadResult
from tests.test_capability_catalog import _evaluation


def _identity(
    repo: str = "synthetic/Example-32B-Instruct", *,
    kind: ModelIdentityEvidenceKind = ModelIdentityEvidenceKind.REPOSITORY_ID,
    value: str | None = None,
) -> ModelIdentity:
    return ModelIdentity(
        canonical_repo_id=repo, model_name=repo.split("/")[-1], variant="instruct",
        evidence=[ModelIdentityEvidence(
            kind=kind, source="synthetic/Repack-GGUF", value=value or repo,
            confidence=EstimationConfidence.HIGH,
        )],
    )


def _catalog() -> CatalogReadResult:
    return CatalogReadResult(
        "loaded", sha256="a" * 64,
        catalog=CapabilityCatalog(catalog_version="synthetic-v1", evaluations=(
            _evaluation(
                variant="post-trained/non-thinking", revision=None, precision=None,
                protocol={"reasoning_mode": "non-thinking"},
            ),
            _evaluation(
                evaluation_id="independent", source_kind="independent", value=65.0,
                source="https://independent.invalid/conflicting-result",
            ),
            _evaluation(evaluation_id="unrelated", repo_id="other/Unrelated"),
        )),
    )


@pytest.mark.parametrize("kind", [
    ModelIdentityEvidenceKind.REPOSITORY_ID, ModelIdentityEvidenceKind.BASE_MODEL_METADATA,
])
def test_references_preserve_variants_conflicting_sources_and_unknown_conditions(
    kind: ModelIdentityEvidenceKind,
) -> None:
    identity = _identity(kind=kind)
    before = identity.model_dump_json()
    text, provenance = published_reference_text(identity, _catalog())
    assert "75% accuracy" in text and "65% accuracy" in text
    assert "publisher-reported" in text and "independent" in text
    assert "post-trained/non-thinking; reasoning mode: non-thinking" in text
    assert "Revision: unknown; precision: unknown" in text
    assert "not measured on this artifact" in text and "current protocol" in text
    assert "other/Unrelated" not in provenance
    assert "synthetic-v1" in provenance and "a" * 64 in provenance
    assert '"few_shot": null' in provenance and '"direction": "higher_is_better"' in provenance
    assert identity.model_dump_json() == before


@pytest.mark.parametrize("identity", [
    _identity(kind=ModelIdentityEvidenceKind.NAME_HEURISTIC),
    _identity(value="other/Unrelated"),
    _identity(value="synthetic/Example-32B-Instruct-GGUF"),
    ModelIdentity(model_name="unresolved"),
])
def test_unconfirmed_lineage_never_displays_a_published_score(identity: ModelIdentity) -> None:
    text, provenance = published_reference_text(identity, _catalog())
    assert "lineage is not confirmed" in text
    assert "75" not in text and not provenance


def test_exact_fine_tune_does_not_inherit_base_results_and_absence_is_not_poor_quality() -> None:
    identity = _identity("synthetic/FineTune")
    text, provenance = published_reference_text(identity, _catalog())
    assert "Absence is not poor quality" in text and not provenance


def test_qwen3_non_thinking_is_a_historical_reference_not_a_variant_match() -> None:
    result = capability_catalog.load_capability_catalog(capability_catalog.default_catalog_path())
    identity = _identity("Qwen/Qwen3-32B").model_copy(update={"variant": None})
    text, provenance = published_reference_text(identity, result)
    assert "MMLU-Redux" in text and "85.7%" in text
    assert "post-trained/non-thinking; reasoning mode: non-thinking" in text
    assert "not measured on this artifact" in text
    assert '"max_generation_tokens": 32768' in provenance
    # Bibliographic display must not relax the existing strict subject lookup.
    from jaull.domain.capability_evidence import CapabilitySubject
    from jaull.recommendation.capability import CapabilitySignal
    from jaull.recommendation.capability_catalog import attach_capability_evidence

    signal = CapabilitySignal(score=0.5)
    attached = attach_capability_evidence(
        signal, CapabilitySubject(repo_id="Qwen/Qwen3-32B"), result,
    )
    assert attached.evaluation_evidence == ()


@pytest.mark.parametrize("status", ["missing", "invalid"])
def test_catalog_failure_is_visible_without_claiming_measurements(status: str) -> None:
    result = CatalogReadResult(status, diagnostic="Synthetic catalog failure")  # type: ignore[arg-type]
    text, provenance = published_reference_text(_identity(), result)
    assert text == "Synthetic catalog failure" and not provenance


def test_service_caches_catalog_with_provenance_and_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.json"
    monkeypatch.setattr(capability_catalog, "default_catalog_path", lambda: path)
    service = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]
    assert service.capability_catalog().status == "missing"
    assert not service._published_evaluations()
    path.write_text(_catalog().catalog.model_dump_json(), encoding="utf-8")  # type: ignore[union-attr]
    assert service.capability_catalog().status == "missing"  # Cached until the next request.
    evaluations = service._published_evaluations()
    result = service.capability_catalog()
    assert result.status == "loaded" and result.sha256
    assert evaluations == result.catalog.evaluations  # type: ignore[union-attr]
    assert service.capability_catalog() is result
    path.write_text("not JSON", encoding="utf-8")
    assert service.capability_catalog() is result
    assert not service._published_evaluations()
    invalid = service.capability_catalog()
    assert invalid.status == "invalid" and invalid.diagnostic
