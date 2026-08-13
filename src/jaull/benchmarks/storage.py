"""Filesystem store for immutable benchmark records."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from jaull.benchmarks.errors import (
    BenchmarkRecordNotFoundError,
    BenchmarkStoreError,
    InvalidBenchmarkIdError,
)
from jaull.domain.benchmarks import BenchmarkRecord
from jaull.paths import user_data_dir

_BENCHMARK_SUFFIX = ".json"
SCHEMA_VERSION = 1
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _default_benchmarks_dir() -> Path:
    return user_data_dir("benchmarks")


class BenchmarkStore:
    """Persist one ``BenchmarkRecord`` per JSON file."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _default_benchmarks_dir()).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, benchmark_id: str) -> Path:
        self._validate_benchmark_id(benchmark_id)
        candidate = (self._root / f"{benchmark_id}{_BENCHMARK_SUFFIX}").resolve()
        root_resolved = self._root.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise InvalidBenchmarkIdError(
                f"Refusing path outside benchmark store root: {candidate}."
            ) from exc
        return candidate

    def save(self, record: BenchmarkRecord) -> Path:
        path = self.path_for(record.identity.benchmark_id)
        if path.is_file():
            existing = self.load(record.identity.benchmark_id)
            if existing != record:
                raise BenchmarkStoreError(
                    "Benchmark id already exists with different record content: "
                    f"{record.identity.benchmark_id}."
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
            raise BenchmarkStoreError(f"Could not save benchmark {path}: {exc}") from exc
        return path

    def load(self, benchmark_id: str) -> BenchmarkRecord:
        path = self.path_for(benchmark_id)
        if not path.is_file():
            raise BenchmarkRecordNotFoundError(
                f"Benchmark record not found: {benchmark_id}."
            )
        try:
            envelope = _BenchmarkRecordEnvelope.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise BenchmarkStoreError(f"Could not read benchmark {path}: {exc}") from exc
        except ValidationError as exc:
            raise BenchmarkStoreError(
                f"Benchmark record is invalid JSON/domain data: {path}."
            ) from exc
        if envelope.schema_version != SCHEMA_VERSION:
            raise BenchmarkStoreError(
                "Unsupported benchmark schema_version "
                f"{envelope.schema_version}; expected {SCHEMA_VERSION}."
            )
        return envelope.benchmark

    def exists(self, benchmark_id: str) -> bool:
        return self.path_for(benchmark_id).is_file()

    def list_ids(self) -> list[str]:
        if not self._root.is_dir():
            return []
        ids = [
            path.name[: -len(_BENCHMARK_SUFFIX)]
            for path in self._root.glob(f"*{_BENCHMARK_SUFFIX}")
            if path.is_file()
        ]
        return sorted(ids)

    @staticmethod
    def _validate_benchmark_id(benchmark_id: str) -> None:
        if not benchmark_id or not _SAFE_ID_RE.fullmatch(benchmark_id):
            raise InvalidBenchmarkIdError(
                f"Invalid benchmark_id {benchmark_id!r}: expected a safe filename id."
            )


class _BenchmarkRecordEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    benchmark: BenchmarkRecord


def _to_envelope(record: BenchmarkRecord) -> _BenchmarkRecordEnvelope:
    return _BenchmarkRecordEnvelope(
        schema_version=SCHEMA_VERSION,
        benchmark=record,
    )


__all__ = ["SCHEMA_VERSION", "BenchmarkStore"]
