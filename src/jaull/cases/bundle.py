"""Create and validate portable snapshots of experimental cases.

A local case manifest deliberately links immutable records held by separate
stores. A bundle is the transport form: it copies those records and referenced
raw evidence into one directory, records the hashes of every byte, and can be
validated without a local Jaull data directory, network access or hardware.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from jaull import __version__
from jaull.cases.errors import CaseBundleError
from jaull.cases.validation import CaseValidationService
from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.cases import (
    CaseBundleBenchmarkFile,
    CaseBundleEvidenceFile,
    CaseBundleFile,
    CaseCheck,
    CaseConsistencyStatus,
    CaseValidationResult,
    EvidenceFileReference,
    ExperimentalCaseBundleManifest,
    ExperimentalCaseManifest,
)
from jaull.domain.experiments import ExperimentRecord
from jaull.experiments.errors import ExperimentRecordNotFoundError
from jaull.observability.provenance import capture_git_commit

_BUNDLE_FILENAME = "bundle.json"
_CASE_FILENAME = "case.json"
_EXPERIMENT_FILENAME = "records/experiment.json"
_BENCHMARK_DIRECTORY = "records/benchmarks"
_EVIDENCE_DIRECTORY = "evidence"
_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class LoadedCaseBundle:
    """Verified, in-memory contents of a portable experimental case."""

    root: Path
    manifest: ExperimentalCaseBundleManifest
    case: ExperimentalCaseManifest
    experiment: ExperimentRecord
    benchmarks: tuple[BenchmarkRecord, ...]


@dataclass(frozen=True)
class CaseBundleService:
    """Transport service whose only dependencies are record loaders."""

    load_experiment: Callable[[str], ExperimentRecord]
    load_benchmark: Callable[[str], BenchmarkRecord]

    def export(
        self,
        case: ExperimentalCaseManifest,
        *,
        destination: Path,
        evidence_root: Path,
    ) -> Path:
        """Write a new bundle atomically, refusing to overwrite a destination."""

        target = destination.expanduser()
        if target.exists():
            raise CaseBundleError(f"Bundle destination already exists: {target}.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaseBundleError(
                f"Could not create bundle parent directory {target.parent}: {exc}"
            ) from exc

        experiment, benchmarks = self._load_records(case)
        source_root = evidence_root.expanduser().resolve()
        evidence_sources = self._resolve_evidence(case.evidence_files, source_root)

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
        )
        try:
            case_file = _write_json(temporary, temporary / _CASE_FILENAME, case)
            experiment_file = _write_json(
                temporary, temporary / _EXPERIMENT_FILENAME, experiment
            )
            benchmark_files = tuple(
                CaseBundleBenchmarkFile(
                    benchmark_id=record.identity.benchmark_id,
                    **_write_json(
                        temporary,
                        temporary
                        / _BENCHMARK_DIRECTORY
                        / f"{record.identity.benchmark_id}.json",
                        record,
                    ).model_dump(),
                )
                for record in benchmarks
            )
            evidence_files = tuple(
                CaseBundleEvidenceFile(
                    reference=reference,
                    **_copy_evidence(
                        temporary,
                        source,
                        temporary / _EVIDENCE_DIRECTORY / reference.path,
                    ).model_dump(),
                )
                for reference, source in evidence_sources
            )
            bundle = ExperimentalCaseBundleManifest(
                exported_at=datetime.now(UTC),
                jaull_version=__version__,
                git_commit=capture_git_commit(),
                case_file=case_file,
                experiment_file=experiment_file,
                benchmark_files=benchmark_files,
                evidence_files=evidence_files,
            )
            _write_json(temporary, temporary / _BUNDLE_FILENAME, bundle)
            os.replace(temporary, target)
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(exc, OSError):
                raise CaseBundleError(
                    f"Could not create case bundle at {target}: {exc}"
                ) from exc
            raise
        return target

    def load(self, root: Path) -> LoadedCaseBundle:
        """Load a bundle and verify every declared snapshot and evidence byte."""

        directory = root.expanduser()
        if not directory.is_dir():
            raise CaseBundleError(f"Case bundle directory not found: {directory}.")
        bundle = _parse_json(
            _read_json_file(directory / _BUNDLE_FILENAME),
            ExperimentalCaseBundleManifest,
            "bundle manifest",
        )
        case = _parse_json(
            _read_json_verified(directory, bundle.case_file),
            ExperimentalCaseManifest,
            "case manifest",
        )
        experiment = _parse_json(
            _read_json_verified(directory, bundle.experiment_file),
            ExperimentRecord,
            "experiment record",
        )
        benchmarks = tuple(
            _parse_json(
                _read_json_verified(directory, item),
                BenchmarkRecord,
                f"benchmark record {item.benchmark_id}",
            )
            for item in bundle.benchmark_files
        )
        self._verify_relationships(bundle, case, experiment, benchmarks)
        for item in bundle.evidence_files:
            _verify_file(directory, item)
        return LoadedCaseBundle(
            root=directory,
            manifest=bundle,
            case=case,
            experiment=experiment,
            benchmarks=benchmarks,
        )

    def validate(self, root: Path) -> CaseValidationResult:
        """Validate a bundle offline, including bundle-file integrity."""

        loaded = self.load(root)
        benchmark_by_id = {
            record.identity.benchmark_id: record for record in loaded.benchmarks
        }

        def load_experiment(experiment_id: str) -> ExperimentRecord:
            if experiment_id != loaded.experiment.identity.experiment_id:
                raise ExperimentRecordNotFoundError(
                    f"Experiment record not found in bundle: {experiment_id}."
                )
            return loaded.experiment

        def load_benchmark(benchmark_id: str) -> BenchmarkRecord:
            try:
                return benchmark_by_id[benchmark_id]
            except KeyError as exc:
                from jaull.benchmarks.errors import BenchmarkRecordNotFoundError

                raise BenchmarkRecordNotFoundError(
                    f"Benchmark record not found in bundle: {benchmark_id}."
                ) from exc

        bundled_case = loaded.case.model_copy(
            update={
                "evidence_files": tuple(
                    reference.model_copy(
                        update={
                            "path": evidence.path,
                            "sha256": evidence.sha256,
                            "size_bytes": evidence.size_bytes,
                        }
                    )
                    for reference, evidence in zip(
                        loaded.case.evidence_files,
                        loaded.manifest.evidence_files,
                        strict=True,
                    )
                )
            }
        )
        result = CaseValidationService(
            load_experiment=load_experiment,
            load_benchmark=load_benchmark,
            evidence_root=loaded.root,
        ).validate(bundled_case)
        return result.model_copy(
            update={
                "checks": (
                    CaseCheck(
                        name="bundle_integrity",
                        status=CaseConsistencyStatus.VALID,
                        detail="all declared files verified",
                    ),
                    *result.checks,
                )
            }
        )

    def _load_records(
        self, case: ExperimentalCaseManifest
    ) -> tuple[ExperimentRecord, tuple[BenchmarkRecord, ...]]:
        try:
            experiment = self.load_experiment(case.experiment_record_id)
            benchmarks = tuple(
                self.load_benchmark(record_id) for record_id in case.benchmark_record_ids
            )
        except Exception as exc:
            raise CaseBundleError(f"Could not load case records: {exc}") from exc
        if experiment.identity.experiment_id != case.experiment_record_id:
            raise CaseBundleError("Loaded experiment id does not match the case manifest.")
        benchmark_ids = tuple(record.identity.benchmark_id for record in benchmarks)
        if benchmark_ids != case.benchmark_record_ids:
            raise CaseBundleError("Loaded benchmark ids do not match the case manifest.")
        return experiment, benchmarks

    @staticmethod
    def _resolve_evidence(
        references: tuple[EvidenceFileReference, ...],
        root: Path,
    ) -> tuple[tuple[EvidenceFileReference, Path], ...]:
        sources: list[tuple[EvidenceFileReference, Path]] = []
        for reference in references:
            source = (root / reference.path).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise CaseBundleError(
                    f"Evidence path escapes its root: {reference.path}."
                ) from exc
            if not source.is_file():
                raise CaseBundleError(f"Evidence file not found: {source}.")
            sources.append((reference, source))
        return tuple(sources)

    @staticmethod
    def _verify_relationships(
        bundle: ExperimentalCaseBundleManifest,
        case: ExperimentalCaseManifest,
        experiment: ExperimentRecord,
        benchmarks: tuple[BenchmarkRecord, ...],
    ) -> None:
        if experiment.identity.experiment_id != case.experiment_record_id:
            raise CaseBundleError("Bundled experiment does not match the case manifest.")
        expected_ids = case.benchmark_record_ids
        found_ids = tuple(record.identity.benchmark_id for record in benchmarks)
        indexed_ids = tuple(item.benchmark_id for item in bundle.benchmark_files)
        if found_ids != expected_ids or indexed_ids != expected_ids:
            raise CaseBundleError("Bundled benchmark records do not match the case manifest.")
        expected_evidence = case.evidence_files
        found_evidence = tuple(item.reference for item in bundle.evidence_files)
        if found_evidence != expected_evidence:
            raise CaseBundleError("Bundled evidence does not match the case manifest.")


def _write_json(root: Path, path: Path, value: BaseModel) -> CaseBundleFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump_json(indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return _bundle_file_for(path, root)


def _copy_evidence(root: Path, source: Path, destination: Path) -> CaseBundleFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256_file(source) != _sha256_file(destination):
        raise CaseBundleError(f"Copied evidence did not match source: {source}.")
    return _bundle_file_for(destination, root)


def _bundle_file_for(path: Path, root: Path) -> CaseBundleFile:
    return CaseBundleFile(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _verify_file(root: Path, file: CaseBundleFile) -> Path:
    path = (root / file.path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CaseBundleError(f"Bundle path escapes root: {file.path}.") from exc
    if not path.is_file():
        raise CaseBundleError(f"Bundle file is missing: {file.path}.")
    if path.stat().st_size != file.size_bytes:
        raise CaseBundleError(f"Bundle file size differs: {file.path}.")
    if _sha256_file(path) != file.sha256:
        raise CaseBundleError(f"Bundle file checksum differs: {file.path}.")
    return path


def _read_json_verified(root: Path, file: CaseBundleFile) -> str:
    return _read_json_file(_verify_file(root, file))


def _read_json_file(path: Path) -> str:
    if not path.is_file():
        raise CaseBundleError(f"Bundle file is missing: {path.name}.")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CaseBundleError(f"Could not read bundle JSON {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CaseBundleError(f"Bundle JSON is not UTF-8: {path.name}.") from exc


def _parse_json[ModelT: BaseModel](
    payload: str, model: type[ModelT], label: str
) -> ModelT:
    try:
        return model.model_validate_json(payload)
    except ValidationError as exc:
        raise CaseBundleError(f"Invalid {label} in case bundle.") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CaseBundleError(f"Could not read bundle file {path}: {exc}") from exc
    return digest.hexdigest()


__all__ = ["CaseBundleService", "LoadedCaseBundle"]
