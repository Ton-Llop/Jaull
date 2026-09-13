"""Filesystem store for immutable experimental case manifests."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from jaull.cases.errors import (
    CaseManifestNotFoundError,
    CaseStoreError,
    InvalidCaseIdError,
)
from jaull.domain.cases import ExperimentalCaseManifest
from jaull.paths import user_data_dir

_CASE_SUFFIX = ".json"
SCHEMA_VERSION = 1
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _default_cases_dir() -> Path:
    return user_data_dir("cases")


class CaseStore:
    """Persist one ``ExperimentalCaseManifest`` per JSON file.

    Same shape as the experiment and benchmark stores next door: a flat
    directory, no index, immutable once written. Unlike ``ExperimentStore``
    there is no legacy fallback, because no manifest predates the envelope.

    Loading is *parsing only*. Whether the records a manifest points at exist,
    or agree with each other, is a question for
    :class:`jaull.cases.validation.CaseValidationService` — a manifest with a
    broken reference is still a well-formed manifest, and must stay readable so
    the broken reference can be reported.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _default_cases_dir()).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, case_id: str) -> Path:
        self._validate_case_id(case_id)
        candidate = (self._root / f"{case_id}{_CASE_SUFFIX}").resolve()
        root_resolved = self._root.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise InvalidCaseIdError(
                f"Refusing path outside case store root: {candidate}."
            ) from exc
        return candidate

    def save(self, manifest: ExperimentalCaseManifest) -> Path:
        path = self.path_for(manifest.identity.case_id)
        if path.is_file():
            existing = self.load(manifest.identity.case_id)
            if existing != manifest:
                raise CaseStoreError(
                    "Case id already exists with different manifest content: "
                    f"{manifest.identity.case_id}."
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_envelope(manifest).model_dump_json(indent=2)
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
            raise CaseStoreError(f"Could not save case {path}: {exc}") from exc
        return path

    def load(self, case_id: str) -> ExperimentalCaseManifest:
        path = self.path_for(case_id)
        if not path.is_file():
            raise CaseManifestNotFoundError(f"Case manifest not found: {case_id}.")
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CaseStoreError(f"Could not read case {path}: {exc}") from exc
        try:
            envelope = _CaseManifestEnvelope.model_validate_json(payload)
        except ValidationError as exc:
            raise CaseStoreError(
                f"Case manifest is invalid JSON/domain data: {path}."
            ) from exc
        if envelope.schema_version != SCHEMA_VERSION:
            raise CaseStoreError(
                "Unsupported case schema_version "
                f"{envelope.schema_version}; expected {SCHEMA_VERSION}."
            )
        return envelope.case

    def exists(self, case_id: str) -> bool:
        return self.path_for(case_id).is_file()

    def list_ids(self) -> list[str]:
        if not self._root.is_dir():
            return []
        ids = [
            path.name[: -len(_CASE_SUFFIX)]
            for path in self._root.glob(f"*{_CASE_SUFFIX}")
            if path.is_file()
        ]
        return sorted(ids)

    @staticmethod
    def _validate_case_id(case_id: str) -> None:
        if not case_id or not _SAFE_ID_RE.fullmatch(case_id):
            raise InvalidCaseIdError(
                f"Invalid case_id {case_id!r}: expected a safe filename id."
            )


class _CaseManifestEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    case: ExperimentalCaseManifest


def _to_envelope(manifest: ExperimentalCaseManifest) -> _CaseManifestEnvelope:
    return _CaseManifestEnvelope(
        schema_version=SCHEMA_VERSION,
        case=manifest,
    )


__all__ = ["SCHEMA_VERSION", "CaseStore"]
