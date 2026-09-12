"""Persistence of case manifests, and the line between parsing and validating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.cases.errors import (
    CaseManifestNotFoundError,
    CaseStoreError,
    InvalidCaseIdError,
)
from jaull.cases.storage import CaseStore
from jaull.cases.validation import identity_for_experiment
from jaull.domain.cases import (
    CaseConsistencyStatus,
    EvidenceFileReference,
    EvidenceFileRole,
    ExperimentalCaseManifest,
)
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.model import ModelAnalysis, SafetensorsSummary
from tests._case_fixtures import benchmark_record, experiment_record


def _manifest(**overrides: Any) -> ExperimentalCaseManifest:
    experiment = overrides.pop("experiment", None) or experiment_record()
    return ExperimentalCaseManifest(
        identity=identity_for_experiment(experiment, label="qwen-2060"),
        experiment_record_id=experiment.identity.experiment_id,
        **overrides,
    )


def test_a_manifest_survives_a_save_and_load(tmp_path: Path) -> None:
    store = CaseStore(root=tmp_path)
    manifest = _manifest(
        evidence_files=(
            EvidenceFileReference(
                path="validation/report.json", role=EvidenceFileRole.REPORT
            ),
        ),
        notes=("first baseline",),
    )

    path = store.save(manifest)

    assert path == store.path_for(manifest.identity.case_id)
    assert store.load(manifest.identity.case_id) == manifest
    assert store.list_ids() == [manifest.identity.case_id]
    assert store.exists(manifest.identity.case_id)


def test_the_file_on_disk_carries_a_schema_version(tmp_path: Path) -> None:
    store = CaseStore(root=tmp_path)
    manifest = _manifest()

    payload = json.loads(store.save(manifest).read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["case"]["experiment_record_id"] == manifest.experiment_record_id


def test_saving_the_same_manifest_twice_is_idempotent(tmp_path: Path) -> None:
    store = CaseStore(root=tmp_path)
    manifest = _manifest()

    assert store.save(manifest) == store.save(manifest)


def test_a_case_id_cannot_be_reused_for_different_content(tmp_path: Path) -> None:
    store = CaseStore(root=tmp_path)
    manifest = _manifest()
    store.save(manifest)
    altered = manifest.model_copy(update={"notes": ("rewritten",)})

    with pytest.raises(CaseStoreError, match="already exists with different"):
        store.save(altered)


def test_loading_an_absent_case_names_it(tmp_path: Path) -> None:
    with pytest.raises(CaseManifestNotFoundError, match="case-missing"):
        CaseStore(root=tmp_path).load("case-missing")


def test_a_malformed_manifest_is_a_parsing_error_not_a_verdict(tmp_path: Path) -> None:
    """Corrupt bytes are a store failure; a broken *reference* is not."""
    store = CaseStore(root=tmp_path)
    (tmp_path / "case-broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(CaseStoreError, match="invalid JSON/domain data"):
        store.load("case-broken")


def test_an_unsupported_schema_version_is_refused(tmp_path: Path) -> None:
    store = CaseStore(root=tmp_path)
    manifest = _manifest()
    path = store.save(manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CaseStoreError, match="Unsupported case schema_version"):
        store.load(manifest.identity.case_id)


@pytest.mark.parametrize("case_id", ["", "../escape", "with/slash", ".hidden"])
def test_unsafe_case_ids_are_refused(tmp_path: Path, case_id: str) -> None:
    with pytest.raises(InvalidCaseIdError):
        CaseStore(root=tmp_path).path_for(case_id)


def test_listing_an_absent_store_is_empty_not_an_error(tmp_path: Path) -> None:
    assert CaseStore(root=tmp_path / "nothing-here").list_ids() == []


def test_the_default_root_lives_beside_the_other_evidence_stores(
    monkeypatch: Any,
) -> None:
    captured: list[tuple[str, ...]] = []

    def fake_user_data_dir(*subdirs: str) -> Path:
        captured.append(subdirs)
        return Path("/tmp/jaull-data").joinpath(*subdirs)

    monkeypatch.setattr("jaull.cases.storage.user_data_dir", fake_user_data_dir)

    assert CaseStore().root == Path("/tmp/jaull-data/cases")
    assert captured == [("cases",)]


def test_the_advisor_can_build_save_load_and_validate_a_case(tmp_path: Path) -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()
    advisor = _advisor(tmp_path, experiment, benchmark)

    manifest = advisor.build_case_manifest(
        experiment_id=experiment.identity.experiment_id,
        benchmark_ids=[benchmark.identity.benchmark_id],
        label="qwen-2060",
    )
    advisor.save_case_manifest(manifest)

    assert advisor.list_case_ids() == [manifest.identity.case_id]
    assert advisor.load_case_manifest(manifest.identity.case_id) == manifest

    result = advisor.validate_case(manifest.identity.case_id)
    assert result.case_id == manifest.identity.case_id
    # Provenance is genuinely absent on every record Jaull writes today.
    assert result.status is CaseConsistencyStatus.PARTIAL


def test_the_advisor_derives_identity_from_the_experiment(tmp_path: Path) -> None:
    experiment = experiment_record()
    advisor = _advisor(tmp_path, experiment, benchmark_record())

    manifest = advisor.build_case_manifest(experiment_id=experiment.identity.experiment_id)

    assert manifest.identity.artifact.repo_id == experiment.artifact.repo_id
    assert manifest.identity.runtime is experiment.runtime.runtime
    assert manifest.identity.machine_fingerprint
    assert manifest.identity.case_id.startswith("case-")


def _advisor(tmp_path: Path, experiment: Any, benchmark: Any) -> AdvisorService:
    from jaull.benchmarks.storage import BenchmarkStore
    from jaull.experiments.storage import ExperimentStore

    experiment_store = ExperimentStore(root=tmp_path / "experiments")
    experiment_store.save(experiment)
    benchmark_store = BenchmarkStore(root=tmp_path / "benchmarks")
    benchmark_store.save(benchmark)

    return AdvisorService.build(
        hf_client=_FakeHfClient(),  # type: ignore[arg-type]
        detect_hardware=_unused_hardware,
        inspect_model=_unused_inspect,
        estimate_memory=_unused_estimate,
        experiment_store=experiment_store,
        benchmark_store=benchmark_store,
        case_store=CaseStore(root=tmp_path / "cases"),
    )


class _FakeHfClient:
    def model_info(self, repo_id: str) -> object:
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


def _unused_hardware(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("case handling must not detect hardware")


def _unused_inspect(repo_id: str, client: object | None = None) -> ModelAnalysis:
    del repo_id, client
    raise AssertionError("case handling must not inspect models")


def _unused_estimate(**kwargs: Any) -> MemoryEstimate:
    del kwargs
    raise AssertionError("case handling must not re-estimate")
