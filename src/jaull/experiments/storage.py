"""Filesystem store for immutable experiment records."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from jaull.domain.experiments import ExperimentRecord
from jaull.experiments.errors import (
    ExperimentRecordNotFoundError,
    ExperimentStoreError,
    InvalidExperimentIdError,
)
from jaull.paths import user_data_dir

_EXPERIMENT_SUFFIX = ".json"
SCHEMA_VERSION = 1
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _default_experiments_dir() -> Path:
    return user_data_dir("experiments")


class ExperimentStore:
    """Persist one ``ExperimentRecord`` per JSON file.

    This is deliberately not a benchmark database: there is no aggregation,
    matrix runner, index schema or calibration logic. The directory of JSON
    files is the first durable representation.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _default_experiments_dir()).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, experiment_id: str) -> Path:
        self._validate_experiment_id(experiment_id)
        candidate = (self._root / f"{experiment_id}{_EXPERIMENT_SUFFIX}").resolve()
        root_resolved = self._root.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise InvalidExperimentIdError(
                f"Refusing path outside experiment store root: {candidate}."
            ) from exc
        return candidate

    def save(self, record: ExperimentRecord) -> Path:
        path = self.path_for(record.identity.experiment_id)
        if path.is_file():
            existing = self.load(record.identity.experiment_id)
            if existing != record:
                raise ExperimentStoreError(
                    "Experiment id already exists with different record content: "
                    f"{record.identity.experiment_id}."
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_envelope(record).model_dump_json(indent=2)
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ExperimentStoreError(f"Could not save experiment {path}: {exc}") from exc
        return path

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self.path_for(experiment_id)
        if not path.is_file():
            raise ExperimentRecordNotFoundError(
                f"Experiment record not found: {experiment_id}."
            )
        try:
            envelope = _ExperimentRecordEnvelope.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if envelope.schema_version != SCHEMA_VERSION:
                raise ExperimentStoreError(
                    "Unsupported experiment schema_version "
                    f"{envelope.schema_version}; expected {SCHEMA_VERSION}."
                )
            return envelope.experiment
        except OSError as exc:
            raise ExperimentStoreError(f"Could not read experiment {path}: {exc}") from exc
        except ValidationError as exc:
            return _load_legacy_record(path, exc)

    def exists(self, experiment_id: str) -> bool:
        return self.path_for(experiment_id).is_file()

    def list_ids(self) -> list[str]:
        if not self._root.is_dir():
            return []
        ids = [
            path.name[: -len(_EXPERIMENT_SUFFIX)]
            for path in self._root.glob(f"*{_EXPERIMENT_SUFFIX}")
            if path.is_file()
        ]
        return sorted(ids)

    @staticmethod
    def _validate_experiment_id(experiment_id: str) -> None:
        if not experiment_id or not _SAFE_ID_RE.fullmatch(experiment_id):
            raise InvalidExperimentIdError(
                f"Invalid experiment_id {experiment_id!r}: expected a safe filename id."
            )


class _ExperimentRecordEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    experiment: ExperimentRecord


def _to_envelope(record: ExperimentRecord) -> _ExperimentRecordEnvelope:
    return _ExperimentRecordEnvelope(
        schema_version=SCHEMA_VERSION,
        experiment=record,
    )


def _load_legacy_record(path: Path, envelope_error: ValidationError) -> ExperimentRecord:
    """Read pre-envelope records created before ``schema_version`` existed."""

    try:
        return ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        if isinstance(exc, OSError):
            raise ExperimentStoreError(f"Could not read experiment {path}: {exc}") from exc
        raise ExperimentStoreError(
            f"Experiment record is invalid JSON/domain data: {path}."
        ) from envelope_error


__all__ = ["SCHEMA_VERSION", "ExperimentStore"]
