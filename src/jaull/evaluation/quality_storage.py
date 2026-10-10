"""Filesystem store for immutable quality-evaluation runs.

The identity digest groups a protocol and exact artifact. A repeated run with
different evidence gets a distinct record ID; neither run replaces the other.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from jaull.evaluation.quality_records import (
    QualityEvidence,
    describe_record,
    digest,
    is_reusable_evidence,
    load_record,
    repeat_key,
    save_record,
    validate_identity,
)
from jaull.exceptions import JaullError
from jaull.paths import user_data_dir

logger = logging.getLogger(__name__)

_RECORD_SUFFIX = ".json"
_IDENTITY_RE = re.compile(r"^[0-9a-f]{64}(?:-[0-9a-f]{64})?$")


class QualityStoreError(JaullError):
    """Base class for quality-evidence store failures."""


class QualityRecordNotFoundError(QualityStoreError):
    """The requested quality record does not exist."""


class InvalidQualityIdError(QualityStoreError):
    """The identity digest cannot be mapped safely to a local path."""


def _default_quality_dir() -> Path:
    return user_data_dir("quality")


class QualityEvidenceStore:
    """Persist validated quality records without overwriting prior runs."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _default_quality_dir()).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, record_id: str) -> Path:
        self._validate_identity_id(record_id)
        candidate = (self._root / f"{record_id}{_RECORD_SUFFIX}").resolve()
        try:
            candidate.relative_to(self._root.resolve())
        except ValueError as exc:
            raise InvalidQualityIdError(
                f"Refusing path outside quality store root: {candidate}."
            ) from exc
        return candidate

    def save(self, record: dict[str, Any]) -> Path:
        """Store one completed run, preserving earlier runs of the same identity.

        The caller is expected to have validated with the strongest check it
        has; this re-validates with the shared contract, because a record that
        cannot be read back is not stored evidence.
        """
        identity_sha256 = record["identity_sha256"]
        if identity_sha256 != digest(record["identity"]):
            raise QualityStoreError("Record identity digest does not match its identity.")
        primary = self.path_for(identity_sha256)
        repeated = self.path_for(f"{identity_sha256}-{digest(record)}")
        for path in (primary, repeated):
            try:
                if path.is_file():
                    try:
                        existing = self.load(path.stem)
                    except QualityStoreError:
                        if path == primary:
                            continue
                        raise
                    if existing == record:
                        return path
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        save_record(path, record)
                    except FileExistsError:
                        # A concurrent writer published before us.
                        try:
                            existing = self.load(path.stem)
                        except QualityStoreError:
                            if path == primary:
                                continue
                            raise
                        if existing == record:
                            return path
                    else:
                        return path
            except OSError as exc:
                raise QualityStoreError(f"Could not save quality record {path}: {exc}") from exc
            except ValueError as exc:
                raise QualityStoreError(f"Refusing to store an invalid record: {exc}") from exc
        raise QualityStoreError(
            f"Quality record ID already stored with different content: {repeated.stem}."
        )

    def load(self, record_id: str) -> dict[str, Any]:
        path = self.path_for(record_id)
        if not path.is_file():
            raise QualityRecordNotFoundError(
                f"Quality record not found: {record_id}."
            )
        try:
            record = load_record(path)
        except UnicodeError as exc:
            raise QualityStoreError(f"Quality record is not UTF-8: {path}.") from exc
        except OSError as exc:
            raise QualityStoreError(f"Could not read quality record {path}: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise QualityStoreError(f"Quality record is invalid: {path} ({exc}).") from exc
        identity_sha256 = record_id[:64]
        if record["identity_sha256"] != identity_sha256 or (
            len(record_id) > 64 and digest(record) != record_id[65:]
        ):
            raise QualityStoreError(
                f"Quality record identity does not match requested id {record_id!r}."
            )
        return record

    def exists(self, record_id: str) -> bool:
        return self.path_for(record_id).is_file()

    def list_ids(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(
            path.name[: -len(_RECORD_SUFFIX)]
            for path in self._root.glob(f"*{_RECORD_SUFFIX}")
            if path.is_file() and _IDENTITY_RE.fullmatch(path.name[: -len(_RECORD_SUFFIX)])
        )

    def records(self) -> list[dict[str, Any]]:
        """Every readable record. An unreadable one is skipped, never guessed at."""
        records: list[dict[str, Any]] = []
        for identity_sha256 in self.list_ids():
            try:
                records.append(self.load(identity_sha256))
            except QualityStoreError:
                logger.debug(
                    "Ignoring unreadable local quality evidence %s.",
                    identity_sha256,
                    exc_info=True,
                )
        return records

    def evidence(self) -> list[QualityEvidence]:
        """The display projection of every readable record."""
        return [describe_record(record) for record in self.records()]

    def lookup(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        """Strict: one reusable run for this identity, or no unchosen aggregate."""
        try:
            validate_identity(identity)
            identity_sha256 = digest(identity)
            self.path_for(identity_sha256)
        except (InvalidQualityIdError, ValueError, KeyError, TypeError, AttributeError):
            return None
        reusable: list[dict[str, Any]] = []
        for record_id in self.list_ids():
            if record_id != identity_sha256 and not record_id.startswith(f"{identity_sha256}-"):
                continue
            try:
                record = self.load(record_id)
            except QualityStoreError:
                continue
            if record["identity_sha256"] == identity_sha256 and is_reusable_evidence(record):
                reusable.append(record)
        # Repeats with the same answers are one result; disagreeing ones are none.
        if len({repeat_key(record) for record in reusable}) != 1:
            return None
        # The same representative as the shadow: the lowest record digest.
        result: dict[str, Any] = min(reusable, key=digest)["result"]
        return result

    @staticmethod
    def _validate_identity_id(record_id: str) -> None:
        if not isinstance(record_id, str) or not _IDENTITY_RE.fullmatch(record_id):
            raise InvalidQualityIdError(
                f"Invalid quality record id {record_id!r}: expected an identity digest "
                "with an optional record digest suffix."
            )


__all__ = [
    "InvalidQualityIdError",
    "QualityEvidenceStore",
    "QualityRecordNotFoundError",
    "QualityStoreError",
]
