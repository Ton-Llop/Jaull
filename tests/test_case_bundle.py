"""Portable experimental-case bundles stay independent from local stores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jaull.cases.bundle import CaseBundleService
from jaull.cases.errors import CaseBundleError
from jaull.cases.validation import identity_for_experiment
from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.cases import (
    CaseConsistencyStatus,
    EvidenceFileReference,
    EvidenceFileRole,
    ExperimentalCaseManifest,
)
from jaull.domain.experiments import ExperimentRecord
from tests._case_fixtures import benchmark_record, experiment_record


def _service(
    experiment: ExperimentRecord,
    benchmarks: tuple[BenchmarkRecord, ...],
) -> CaseBundleService:
    by_id = {record.identity.benchmark_id: record for record in benchmarks}

    def load_experiment(record_id: str) -> ExperimentRecord:
        assert record_id == experiment.identity.experiment_id
        return experiment

    def load_benchmark(record_id: str) -> BenchmarkRecord:
        return by_id[record_id]

    return CaseBundleService(
        load_experiment=load_experiment,
        load_benchmark=load_benchmark,
    )


def _case(
    experiment: ExperimentRecord,
    benchmarks: tuple[BenchmarkRecord, ...],
    evidence: tuple[EvidenceFileReference, ...] = (),
) -> ExperimentalCaseManifest:
    return ExperimentalCaseManifest(
        identity=identity_for_experiment(experiment, label="RTX 2060 baseline"),
        experiment_record_id=experiment.identity.experiment_id,
        benchmark_record_ids=tuple(record.identity.benchmark_id for record in benchmarks),
        evidence_files=evidence,
    )


def test_exported_bundle_validates_without_local_record_loaders(tmp_path: Path) -> None:
    experiment = experiment_record()
    benchmark = benchmark_record(context_length=4096)
    evidence_root = tmp_path / "evidence-source"
    source = evidence_root / "validation" / "raw.log"
    source.parent.mkdir(parents=True)
    raw = b"llama.cpp original output  \n\x1b[0m"
    source.write_bytes(raw)
    case = _case(
        experiment,
        (benchmark,),
        (
            EvidenceFileReference(
                path="validation/raw.log",
                role=EvidenceFileRole.RUNTIME_LOG,
            ),
        ),
    )
    bundle_root = tmp_path / "nested" / "portable-case"

    _service(experiment, (benchmark,)).export(
        case,
        destination=bundle_root,
        evidence_root=evidence_root,
    )

    def unavailable(_: str) -> ExperimentRecord:
        raise AssertionError("portable validation must not read a local store")

    def unavailable_benchmark(_: str) -> BenchmarkRecord:
        raise AssertionError("portable validation must not read a local store")

    portable = CaseBundleService(
        load_experiment=unavailable,
        load_benchmark=unavailable_benchmark,
    )
    loaded = portable.load(bundle_root)
    result = portable.validate(bundle_root)

    assert loaded.case == case
    assert loaded.experiment == experiment
    assert loaded.benchmarks == (benchmark,)
    assert (bundle_root / "evidence" / "validation" / "raw.log").read_bytes() == raw
    assert result.status is CaseConsistencyStatus.PARTIAL
    assert result.checks[0].name == "bundle_integrity"
    assert result.checks[0].status is CaseConsistencyStatus.VALID


def test_export_refuses_missing_evidence_without_publishing_destination(
    tmp_path: Path,
) -> None:
    experiment = experiment_record()
    case = _case(
        experiment,
        (),
        (EvidenceFileReference(path="missing/runtime.log"),),
    )
    target = tmp_path / "bundle"

    with pytest.raises(CaseBundleError, match="Evidence file not found"):
        _service(experiment, ()).export(
            case,
            destination=target,
            evidence_root=tmp_path,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".bundle.*"))


def test_export_refuses_evidence_with_a_different_recorded_checksum(
    tmp_path: Path,
) -> None:
    experiment = experiment_record()
    source = tmp_path / "source" / "output.log"
    source.parent.mkdir()
    original = b"original"
    source.write_bytes(b"changed!")
    case = _case(
        experiment,
        (),
        (
            EvidenceFileReference(
                path="output.log",
                sha256=hashlib.sha256(original).hexdigest(),
                size_bytes=len(original),
            ),
        ),
    )
    target = tmp_path / "bundle"

    with pytest.raises(CaseBundleError, match="checksum differs from case reference"):
        _service(experiment, ()).export(
            case,
            destination=target,
            evidence_root=source.parent,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".bundle.*"))


def test_export_refuses_evidence_with_a_different_recorded_size(tmp_path: Path) -> None:
    experiment = experiment_record()
    source = tmp_path / "source" / "output.log"
    source.parent.mkdir()
    source.write_bytes(b"short")
    case = _case(
        experiment,
        (),
        (EvidenceFileReference(path="output.log", size_bytes=len(b"original")),),
    )
    target = tmp_path / "bundle"

    with pytest.raises(CaseBundleError, match="size differs from case reference"):
        _service(experiment, ()).export(
            case,
            destination=target,
            evidence_root=source.parent,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".bundle.*"))


def test_bundle_validation_keeps_the_case_evidence_identity(tmp_path: Path) -> None:
    experiment = experiment_record()
    source = tmp_path / "source" / "output.log"
    source.parent.mkdir()
    raw = b"original"
    source.write_bytes(raw)
    case = _case(
        experiment,
        (),
        (
            EvidenceFileReference(
                path="output.log",
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
            ),
        ),
    )
    target = tmp_path / "bundle"
    service = _service(experiment, ())
    service.export(case, destination=target, evidence_root=source.parent)

    # A malformed historical identity is distinct from bundle transport
    # integrity. Keep the copied bytes valid, but make the frozen reference
    # wrong and update its two enclosing bundle index entries accordingly.
    case_path = target / "case.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["evidence_files"][0]["sha256"] = "0" * 64
    case_path.write_text(json.dumps(case_payload, indent=2) + "\n", encoding="utf-8")

    bundle_path = target / "bundle.json"
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle_payload["case_file"]["sha256"] = hashlib.sha256(
        case_path.read_bytes()
    ).hexdigest()
    bundle_payload["case_file"]["size_bytes"] = case_path.stat().st_size
    bundle_payload["evidence_files"][0]["reference"]["sha256"] = "0" * 64
    bundle_path.write_text(
        json.dumps(bundle_payload, indent=2) + "\n", encoding="utf-8"
    )

    result = service.validate(target)

    evidence_check = next(check for check in result.checks if check.name == "evidence_files")
    assert evidence_check.status is CaseConsistencyStatus.PARTIAL
    assert "does not match its recorded digest" in evidence_check.detail


def test_bundle_validation_rejects_tampered_raw_evidence(tmp_path: Path) -> None:
    experiment = experiment_record()
    source = tmp_path / "source" / "output.log"
    source.parent.mkdir()
    source.write_bytes(b"original")
    case = _case(
        experiment,
        (),
        (EvidenceFileReference(path="output.log"),),
    )
    target = tmp_path / "bundle"
    service = _service(experiment, ())
    service.export(case, destination=target, evidence_root=source.parent)
    (target / "evidence" / "output.log").write_bytes(b"changed!")

    with pytest.raises(CaseBundleError, match="checksum differs"):
        service.validate(target)


def test_bundle_refuses_to_overwrite_an_existing_destination(tmp_path: Path) -> None:
    experiment = experiment_record()
    target = tmp_path / "bundle"
    target.mkdir()

    with pytest.raises(CaseBundleError, match="already exists"):
        _service(experiment, ()).export(
            _case(experiment, ()),
            destination=target,
            evidence_root=tmp_path,
        )
