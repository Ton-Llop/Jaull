"""One experimental case: a curated grouping of evidence about one scenario.

An ``ExperimentRecord`` and a ``BenchmarkRecord`` can come from different
processes, minutes or days apart. Nothing in either record points at the other,
so today the relationship is re-derived on every read from the machine
fingerprint, the artifact fields and the runtime flags. That derivation is an
*inference*; a manifest is an *assertion* made by whoever ran the experiment.

The manifest therefore links and never copies. ``HardwareProfile``,
``MemoryEstimate``, ``ExecutionObservation``, ``BenchmarkObservation`` and every
metric already have a canonical authority inside the records themselves, and a
second copy here could only ever drift out of agreement with the first. What the
manifest does keep is derived identity — a fingerprint, an artifact identity —
so a wrong association can be detected rather than silently believed.

"Case" rather than "run" on purpose: the records grouped here are evidence about
the same scenario, not output of the same process.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jaull.domain.artifacts import ArtifactIdentity
from jaull.domain.runtime import RuntimeName

CASE_MANIFEST_SCHEMA_VERSION = 1

# The same shape the experiment and benchmark stores accept, because a case id
# becomes a filename under the same rules.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExperimentalCaseIdentity(BaseModel):
    """What makes two runs the same experimental case.

    Stable facts only. Instantaneous conditions belong to each record and are
    deliberately absent here: available RAM and VRAM, driver and CUDA version,
    and the runtime build all change between two runs of the same case, so
    putting them in the identity would make every case a case of one.

    ``machine_fingerprint`` is the tuple from
    :func:`jaull.evaluation.hardware_fingerprint.machine_fingerprint`, which
    already excludes volatile memory figures.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    created_at: datetime
    label: str | None = None
    machine_fingerprint: tuple[str, ...] = Field(min_length=1)
    artifact: ArtifactIdentity
    runtime: RuntimeName

    @field_validator("case_id")
    @classmethod
    def _id_is_safe(cls, value: str) -> str:
        if not _SAFE_ID_RE.match(value):
            raise ValueError(f"case_id is not a safe identifier: {value!r}")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @classmethod
    def create(
        cls,
        *,
        machine_fingerprint: tuple[str, ...],
        artifact: ArtifactIdentity,
        runtime: RuntimeName,
        label: str | None = None,
    ) -> Self:
        """Mint a new case identity from facts the caller has already derived.

        The fingerprint arrives ready-made rather than being computed here: it
        comes from ``jaull.evaluation``, and the domain does not reach into the
        layers built on top of it. ``jaull.cases.identity_for_experiment`` is the
        seam that does the deriving.
        """
        return cls(
            case_id=f"case-{uuid4()}",
            created_at=datetime.now(UTC),
            label=label,
            machine_fingerprint=machine_fingerprint,
            artifact=artifact,
            runtime=runtime,
        )


class EvidenceFileRole(StrEnum):
    """What a referenced file is, so a reader knows what it would be opening."""

    PREDICTION = "prediction"
    REPORT = "report"
    RUNTIME_LOG = "runtime_log"
    TIMING = "timing"
    OTHER = "other"


class EvidenceFileReference(BaseModel):
    """A pointer to raw evidence the manifest does not interpret.

    No record stores a log path today: experiments discard stdout and stderr
    entirely, and benchmarks inline them as strings. The files produced by
    ``scripts/validate_hardware_fit_against_llama_cpp.py`` live outside the
    record model altogether. This is how they are attached to a case.

    The path is relative and POSIX-shaped so a manifest survives moving between
    machines — the same case is meant to be reproduced on other hardware, and an
    absolute path from the machine that produced it would be a dead reference
    everywhere else. The root it is relative to is supplied by the caller at
    validation time, never stored.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    role: EvidenceFileRole = EvidenceFileRole.OTHER
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    description: str | None = None

    @field_validator("path")
    @classmethod
    def _path_is_relative(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence path must not be empty")
        if "\\" in value:
            raise ValueError(
                f"evidence path must use forward slashes: {value!r}"
            )
        if value.startswith("/"):
            raise ValueError(f"evidence path must be relative: {value!r}")
        # A drive letter is absolute on Windows even without a leading slash.
        if len(value) > 1 and value[1] == ":":
            raise ValueError(f"evidence path must be relative: {value!r}")
        if ".." in value.split("/"):
            raise ValueError(
                f"evidence path must not climb out of its root: {value!r}"
            )
        return value


class ExperimentalCaseManifest(BaseModel):
    """The case itself: identity, the records it groups, and its raw evidence."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = CASE_MANIFEST_SCHEMA_VERSION
    identity: ExperimentalCaseIdentity
    experiment_record_id: str
    benchmark_record_ids: tuple[str, ...] = ()
    evidence_files: tuple[EvidenceFileReference, ...] = ()
    notes: tuple[str, ...] = ()

    @field_validator("experiment_record_id")
    @classmethod
    def _experiment_id_is_safe(cls, value: str) -> str:
        if not _SAFE_ID_RE.match(value):
            raise ValueError(f"experiment_record_id is not a safe identifier: {value!r}")
        return value

    @field_validator("benchmark_record_ids")
    @classmethod
    def _benchmark_ids_are_safe_unique_and_ordered(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        for record_id in value:
            if not _SAFE_ID_RE.match(record_id):
                raise ValueError(
                    f"benchmark_record_ids contains an unsafe identifier: {record_id!r}"
                )
        if len(set(value)) != len(value):
            raise ValueError("benchmark_record_ids must not repeat an id")
        # Sorted on the way in so the same case always serializes identically,
        # whatever order the ids were typed in.
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _supported_schema_version(self) -> Self:
        if self.schema_version != CASE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported ExperimentalCaseManifest schema_version: "
                f"{self.schema_version}"
            )
        return self


class CaseConsistencyStatus(StrEnum):
    """Whether the case holds together, and how confidently.

    ``PARTIAL`` is not a soft failure: it is the honest answer when the records
    agree on everything they can express but something needed for certainty was
    never recorded — most often an artifact digest or the git commit.
    """

    VALID = "valid"
    PARTIAL = "partial"
    INCONSISTENT = "inconsistent"


class CaseCheck(BaseModel):
    """One named question asked of the case, and its answer."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: CaseConsistencyStatus
    detail: str | None = None


class CaseValidationResult(BaseModel):
    """The verdict on a case. It never modifies a record.

    ``reasons`` say why the case is not ``VALID``; ``warnings`` describe
    conditions that legitimately differed between runs — free memory, driver
    version, runtime build — and never lower the status on their own.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    status: CaseConsistencyStatus
    checks: tuple[CaseCheck, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


__all__ = [
    "CASE_MANIFEST_SCHEMA_VERSION",
    "CaseCheck",
    "CaseConsistencyStatus",
    "CaseValidationResult",
    "EvidenceFileReference",
    "EvidenceFileRole",
    "ExperimentalCaseIdentity",
    "ExperimentalCaseManifest",
]
