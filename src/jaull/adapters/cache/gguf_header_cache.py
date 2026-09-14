"""Revision-aware persistent cache for parsed GGUF headers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.model import GgufVariant, ModelAnalysis, ModelFile
from jaull.paths import user_cache_dir
from jaull.ports.cache import GgufHeaderCacheProtocol, GgufHeaderCacheStats

logger = logging.getLogger(__name__)

GGUF_HEADER_CACHE_SCHEMA_VERSION = 1
DEFAULT_GGUF_HEADER_CACHE_TTL_SECONDS = 24 * 60 * 60

# Vocabulary, merges and token types. Megabytes each, never read back.
_UNSTORED_KV_PREFIX = "tokenizer."


class NullGgufHeaderCache:
    """Cache implementation used when persistent headers are disabled."""

    def __init__(self) -> None:
        self.stats = GgufHeaderCacheStats()

    def get(
        self,
        analysis: ModelAnalysis,
        variant: GgufVariant,
    ) -> GgufHeaderMetadata | None:
        del analysis, variant
        self.stats.misses += 1
        return None

    def put(
        self,
        analysis: ModelAnalysis,
        variant: GgufVariant,
        header: GgufHeaderMetadata,
    ) -> None:
        del analysis, variant, header


class GgufHeaderCache:
    """Cache a parsed header for one concrete GGUF artifact.

    A key includes the repository, first shard path, first shard size and the
    repository revision when available. Entries without a revision use the
    same bounded TTL policy as the model-analysis cache. Failed/no-header
    reads are intentionally not cached: another artifact or a later network
    attempt may provide useful metadata.

    The tokenizer entries of ``raw_kv`` are dropped before writing. They are by
    far the largest thing in a GGUF header — ``tokenizer.ggml.merges`` alone was
    6.4 MB in one Llama-3.1 artifact, and a 73-entry cache came to 764 MB — and
    nothing in Jaull reads them: ``raw_kv`` exists so ``_build_header`` can pull
    the typed fields out of it, and no consumer touches it afterwards. Keeping
    the rest costs 0.1 MB for the same 73 entries. A cached header therefore has
    a smaller ``raw_kv`` than a freshly parsed one; that difference is the point,
    not an accident.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        ttl_seconds: int = DEFAULT_GGUF_HEADER_CACHE_TTL_SECONDS,
    ) -> None:
        self.root = root or user_cache_dir("gguf-headers")
        self.ttl = timedelta(seconds=ttl_seconds)
        self.stats = GgufHeaderCacheStats()
        self._lock = threading.Lock()

    def get(
        self,
        analysis: ModelAnalysis,
        variant: GgufVariant,
    ) -> GgufHeaderMetadata | None:
        target = _first_shard(variant)
        if target is None:
            self._miss()
            return None
        key = _cache_key(analysis, target)
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
        if raw.get("schema_version") != GGUF_HEADER_CACHE_SCHEMA_VERSION:
            self._unsupported_schema()
            self._miss()
            return None
        if raw.get("cache_key") != key:
            self._read_error(path, ValueError("cache key mismatch"))
            self._miss()
            return None
        if _cache_key_kind(analysis) == "ttl" and self._is_expired(raw):
            self._expired()
            self._miss()
            return None

        payload = raw.get("header")
        if not isinstance(payload, dict):
            self._read_error(path, ValueError("missing header payload"))
            self._miss()
            return None
        try:
            header = GgufHeaderMetadata.model_validate(payload)
        except ValueError as exc:
            self._read_error(path, exc)
            self._miss()
            return None
        with self._lock:
            self.stats.hits += 1
        return header

    def put(
        self,
        analysis: ModelAnalysis,
        variant: GgufVariant,
        header: GgufHeaderMetadata,
    ) -> None:
        target = _first_shard(variant)
        if target is None:
            return
        key = _cache_key(analysis, target)
        payload = {
            "schema_version": GGUF_HEADER_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "cache_key_kind": _cache_key_kind(analysis),
            "revision_hint": _revision_hint(analysis),
            "repo_id": analysis.repo.repo_id,
            "filename": target.path,
            "size_bytes": target.size_bytes,
            "cached_at": _now().isoformat(),
            "header": _storable_header(header),
        }
        try:
            self._write_atomic(self._path_for_key(key), payload)
        except OSError as exc:
            logger.debug("Could not write GGUF header cache %s: %s", key, exc)
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
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
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
        logger.debug("Ignoring GGUF header cache entry %s: %s", path, exc)
        with self._lock:
            self.stats.read_errors += 1

    def _unsupported_schema(self) -> None:
        with self._lock:
            self.stats.unsupported_schema += 1

    def _expired(self) -> None:
        with self._lock:
            self.stats.expired += 1


def _storable_header(header: GgufHeaderMetadata) -> dict[str, object]:
    """Serialize a header without the tokenizer tables.

    See the note on :class:`GgufHeaderCache`: the vocabulary and merge list are
    almost the whole file and nothing reads them back.
    """

    payload = header.model_dump(mode="json")
    raw_kv = payload.get("raw_kv")
    if isinstance(raw_kv, dict):
        payload["raw_kv"] = {
            key: value
            for key, value in raw_kv.items()
            if not key.startswith(_UNSTORED_KV_PREFIX)
        }
    return payload


def _first_shard(variant: GgufVariant) -> ModelFile | None:
    return min(variant.files, key=lambda file: file.path) if variant.files else None


def _cache_key(analysis: ModelAnalysis, target: ModelFile) -> str:
    return "\0".join(
        (
            analysis.repo.repo_id,
            target.path,
            str(target.size_bytes),
            _cache_key_kind(analysis),
            _revision_hint(analysis),
        )
    )


def _cache_key_kind(analysis: ModelAnalysis) -> str:
    return "last_modified" if analysis.repo.last_modified is not None else "ttl"


def _revision_hint(analysis: ModelAnalysis) -> str:
    return analysis.repo.last_modified.isoformat() if analysis.repo.last_modified else "unknown"


def _now() -> datetime:
    return datetime.now(tz=UTC)


__all__ = [
    "DEFAULT_GGUF_HEADER_CACHE_TTL_SECONDS",
    "GGUF_HEADER_CACHE_SCHEMA_VERSION",
    "GgufHeaderCache",
    "GgufHeaderCacheProtocol",
    "GgufHeaderCacheStats",
    "NullGgufHeaderCache",
]
