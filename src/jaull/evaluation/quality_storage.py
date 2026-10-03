"""Filesystem store for immutable quality-evaluation records.

Keyed by ``identity_sha256``, not by the artifact digest and certainly not by
the model name. The placement replay measured that the same artifact under a
different offload split disagrees on every token logprob, so the key has to
cover the runtime as well as the weights — which ``identity_sha256`` already
does, since it digests artifact, suite, dataset, samples, evaluator, runtime
and protocol together.
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
    load_record,
    quality_lookup,
    save_record,
)
from jaull.exceptions import JaullError
from jaull.paths import user_data_dir

logger = logging.getLogger(__name__)

_RECORD_SUFFIX = ".json"
_IDENTITY_RE = re.compile(r"^[0-9a-f]{64}$")


class QualityStoreError(JaullError):
    """Base class for quality-evidence store failures."""


class QualityRecordNotFoundError(QualityStoreError):
    """The requested quality record does not exist."""


class InvalidQualityIdError(QualityStoreError):
    """The identity digest cannot be mapped safely to a local path."""


def _default_quality_dir() -> Path:
    return user_data_dir("quality")


class QualityEvidenceStore:
    """Persist one validated quality record per JSON file."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _default_quality_dir()).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, identity_sha256: str) -> Path:
        self._validate_identity_id(identity_sha256)
        candidate = (self._root / f"{identity_sha256}{_RECORD_SUFFIX}").resolve()
        try:
            candidate.relative_to(self._root.resolve())
        except ValueError as exc:
            raise InvalidQualityIdError(
                f"Refusing path outside quality store root: {candidate}."
            ) from exc
        return candidate

    def save(self, record: dict[str, Any]) -> Path:
        """Store a record under its own identity digest, once.

        The caller is expected to have validated with the strongest check it
        has; this re-validates with the shared contract, because a record that
        cannot be read back is not stored evidence.
        """
        identity_sha256 = record["identity_sha256"]
        if identity_sha256 != digest(record["identity"]):
            raise QualityStoreError("Record identity digest does not match its identity.")
        path = self.path_for(identity_sha256)
        if path.is_file():
            if self.load(identity_sha256) != record:
                raise QualityStoreError(
                    "Quality identity already stored with different record content: "
                    f"{identity_sha256}."
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_record(path, record)
        except FileExistsError:
            # Another writer won the race; the content check above applies.
            if self.load(identity_sha256) != record:
                raise QualityStoreError(
                    f"Quality identity already stored with different content: {identity_sha256}."
                ) from None
        except OSError as exc:
            raise QualityStoreError(f"Could not save quality record {path}: {exc}") from exc
        except ValueError as exc:
            raise QualityStoreError(f"Refusing to store an invalid record: {exc}") from exc
        return path

    def load(self, identity_sha256: str) -> dict[str, Any]:
        path = self.path_for(identity_sha256)
        if not path.is_file():
            raise QualityRecordNotFoundError(
                f"Quality record not found: {identity_sha256}."
            )
        try:
            record = load_record(path)
        except UnicodeError as exc:
            raise QualityStoreError(f"Quality record is not UTF-8: {path}.") from exc
        except OSError as exc:
            raise QualityStoreError(f"Could not read quality record {path}: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise QualityStoreError(f"Quality record is invalid: {path} ({exc}).") from exc
        if record["identity_sha256"] != identity_sha256:
            raise QualityStoreError(
                f"Quality record identity does not match requested id {identity_sha256!r}."
            )
        return record

    def exists(self, identity_sha256: str) -> bool:
        return self.path_for(identity_sha256).is_file()

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
        """Strict: reusable evidence for this exact identity, or nothing."""
        try:
            path = self.path_for(digest(identity))
        except (InvalidQualityIdError, ValueError, TypeError):
            return None
        return quality_lookup(path, identity)

    @staticmethod
    def _validate_identity_id(identity_sha256: str) -> None:
        if not isinstance(identity_sha256, str) or not _IDENTITY_RE.fullmatch(identity_sha256):
            raise InvalidQualityIdError(
                f"Invalid identity digest {identity_sha256!r}: expected 64 lowercase hex chars."
            )


__all__ = [
    "InvalidQualityIdError",
    "QualityEvidenceStore",
    "QualityRecordNotFoundError",
    "QualityStoreError",
]
