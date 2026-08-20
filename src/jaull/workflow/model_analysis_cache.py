"""Persistent cache for expensive repository analysis metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from jaull.domain.candidates import ModelCandidate
from jaull.domain.model import ModelAnalysis
from jaull.paths import user_cache_dir

logger = logging.getLogger(__name__)

ANALYSIS_CACHE_SCHEMA_VERSION = 2
DEFAULT_TTL_SECONDS = 24 * 60 * 60


@dataclass
class ModelAnalysisCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    read_errors: int = 0
    write_errors: int = 0
    unsupported_schema: int = 0
    expired: int = 0


class ModelAnalysisCacheProtocol(Protocol):
    stats: ModelAnalysisCacheStats

    def get(self, candidate: ModelCandidate) -> ModelAnalysis | None: ...
    def put(self, candidate: ModelCandidate, analysis: ModelAnalysis) -> None: ...


class NullModelAnalysisCache:
    """Cache implementation used when persistence is intentionally disabled."""

    def __init__(self) -> None:
        self.stats = ModelAnalysisCacheStats()

    def get(self, candidate: ModelCandidate) -> ModelAnalysis | None:
        del candidate
        self.stats.misses += 1
        return None

    def put(self, candidate: ModelCandidate, analysis: ModelAnalysis) -> None:
        del candidate, analysis


class ModelAnalysisCache:
    """Revision-aware JSON cache for :class:`ModelAnalysis`.

    This is deliberately not an evidence store: entries are disposable, keyed
    by immutable-ish repository metadata from search results, and ignored on
    corruption or unsupported schema.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.root = root or user_cache_dir("model-analysis")
        self.ttl = timedelta(seconds=ttl_seconds)
        self.stats = ModelAnalysisCacheStats()
        self._lock = threading.Lock()

    def get(self, candidate: ModelCandidate) -> ModelAnalysis | None:
        key = _cache_key(candidate)
        path = self._path_for_key(key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._miss()
            return None
        except (OSError, json.JSONDecodeError) as exc:
            self._read_error(path, exc)
            self._miss()
            return None

        if not isinstance(raw, dict):
            self._read_error(path, ValueError("cache entry is not an object"))
            self._miss()
            return None
        if raw.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
            self._unsupported_schema()
            self._miss()
            return None
        if raw.get("repo_id") != candidate.repo_id:
            self._read_error(path, ValueError("repo_id mismatch"))
            self._miss()
            return None
        if raw.get("cache_key") != key:
            self._read_error(path, ValueError("cache key mismatch"))
            self._miss()
            return None
        if _cache_key_kind(candidate) == "ttl" and self._is_expired(raw):
            self._expired()
            self._miss()
            return None

        analysis_payload = raw.get("analysis")
        if not isinstance(analysis_payload, dict):
            self._read_error(path, ValueError("missing analysis payload"))
            self._miss()
            return None
        try:
            analysis = ModelAnalysis.model_validate(analysis_payload)
        except ValueError as exc:
            self._read_error(path, exc)
            self._miss()
            return None
        with self._lock:
            self.stats.hits += 1
        return analysis

    def put(self, candidate: ModelCandidate, analysis: ModelAnalysis) -> None:
        key = _cache_key(candidate)
        path = self._path_for_key(key)
        payload = {
            "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
            "repo_id": candidate.repo_id,
            "cache_key": key,
            "cache_key_kind": _cache_key_kind(candidate),
            "revision_hint": _revision_hint(candidate),
            "cached_at": _now().isoformat(),
            "analysis": analysis.model_dump(mode="json"),
        }
        try:
            self._write_atomic(path, payload)
        except OSError as exc:
            logger.debug("Could not write model analysis cache %s: %s", path, exc)
            with self._lock:
                self.stats.write_errors += 1
            return
        with self._lock:
            self.stats.writes += 1

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _write_atomic(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        tmp = Path(raw_tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError:
            with suppress(OSError):
                tmp.unlink()
            raise

    def _is_expired(self, raw: dict[object, object]) -> bool:
        cached_at = raw.get("cached_at")
        if not isinstance(cached_at, str):
            return True
        try:
            timestamp = datetime.fromisoformat(cached_at)
        except ValueError:
            return True
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return _now() - timestamp > self.ttl

    def _miss(self) -> None:
        with self._lock:
            self.stats.misses += 1

    def _read_error(self, path: Path, exc: Exception) -> None:
        logger.debug("Ignoring model analysis cache entry %s: %s", path, exc)
        with self._lock:
            self.stats.read_errors += 1

    def _unsupported_schema(self) -> None:
        with self._lock:
            self.stats.unsupported_schema += 1

    def _expired(self) -> None:
        with self._lock:
            self.stats.expired += 1


def _cache_key(candidate: ModelCandidate) -> str:
    return f"{candidate.repo_id}\0{_cache_key_kind(candidate)}\0{_revision_hint(candidate)}"


def _cache_key_kind(candidate: ModelCandidate) -> str:
    if candidate.revision_hint:
        return "revision"
    if candidate.last_modified is not None:
        return "last_modified"
    return "ttl"


def _revision_hint(candidate: ModelCandidate) -> str:
    if candidate.revision_hint:
        return candidate.revision_hint
    if candidate.last_modified is not None:
        return candidate.last_modified.isoformat()
    return "unknown"


def _now() -> datetime:
    return datetime.now(tz=UTC)


__all__ = [
    "ANALYSIS_CACHE_SCHEMA_VERSION",
    "DEFAULT_TTL_SECONDS",
    "ModelAnalysisCache",
    "ModelAnalysisCacheProtocol",
    "ModelAnalysisCacheStats",
    "NullModelAnalysisCache",
]
