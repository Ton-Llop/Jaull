"""Does an experimental case hold together?

Offline and read-only. Nothing here detects hardware, queries Hugging Face,
runs a runtime, or rewrites a record. A verdict is a verdict; the evidence it
judges stays exactly as it was written.

Two failure kinds are kept apart on purpose:

* a manifest that will not parse is a store error, raised by ``CaseStore``;
* a manifest that parses but points at a record which is not there is an
  ``INCONSISTENT`` result carrying a concrete reason.

The second must never become an exception, or a mistyped id would crash the
listing instead of explaining itself.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from jaull.domain.artifacts import ArtifactIdentity, artifact_identity_matches
from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.cases import (
    CaseCheck,
    CaseConsistencyStatus,
    CaseValidationResult,
    EvidenceFileReference,
    ExperimentalCaseIdentity,
    ExperimentalCaseManifest,
)
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import runtime_capability_version
from jaull.evaluation.hardware_fingerprint import machine_fingerprint
from jaull.exceptions import JaullError

_READ_CHUNK_BYTES = 1024 * 1024
_LLAMA_CPP_BUILD = re.compile(
    r"(?:version:\s*)?(?P<number>\d+)\s*\((?P<commit>[0-9a-f]+)\)"
    r"|build:\s*(?P<build_commit>[0-9a-f]+)\s*\((?P<build_number>\d+)\)",
    re.IGNORECASE,
)


def identity_for_experiment(
    record: ExperimentRecord,
    *,
    label: str | None = None,
) -> ExperimentalCaseIdentity:
    """Derive a case identity from the experiment that anchors the case.

    The experiment anchors rather than a benchmark because it is the record
    that carries the frozen prediction. Benchmarks are then checked against it.
    """

    return ExperimentalCaseIdentity.create(
        machine_fingerprint=machine_fingerprint(record.hardware),
        artifact=ArtifactIdentity.of(record.artifact),
        runtime=record.runtime.runtime,
        label=label,
    )


@dataclass(frozen=True)
class CaseValidationService:
    """Check a manifest against the records and files it names.

    Both loaders are injected so the service never reaches for a store of its
    own, and so a test can prove that no record was touched beyond the ids the
    manifest actually lists.
    """

    load_experiment: Callable[[str], ExperimentRecord]
    load_benchmark: Callable[[str], BenchmarkRecord]
    evidence_root: Path = field(default_factory=Path)

    def validate(self, manifest: ExperimentalCaseManifest) -> CaseValidationResult:
        checks: list[CaseCheck] = []
        reasons: list[str] = []
        warnings: list[str] = []

        experiment, benchmarks, missing = self._load_records(manifest)
        if missing:
            checks.append(
                CaseCheck(
                    name="records_exist",
                    status=CaseConsistencyStatus.INCONSISTENT,
                    detail="; ".join(missing),
                )
            )
            reasons.extend(missing)
        else:
            checks.append(
                CaseCheck(
                    name="records_exist",
                    status=CaseConsistencyStatus.VALID,
                    detail=f"1 experiment, {len(benchmarks)} benchmark(s)",
                )
            )

        if experiment is None:
            # Without the anchor there is nothing left to compare against. The
            # evidence files are still worth checking: they are the part of a
            # case that does not depend on any record.
            checks.append(self._check_evidence(manifest.evidence_files, reasons))
            return _result(manifest.identity.case_id, checks, reasons, warnings)

        checks.append(
            self._check_machine(manifest.identity, experiment, benchmarks, reasons)
        )
        checks.append(
            self._check_artifact(manifest.identity, experiment, benchmarks, reasons)
        )
        checks.append(self._check_artifact_hash(experiment, benchmarks, reasons))
        checks.append(
            self._check_runtime_family(manifest.identity, experiment, benchmarks, reasons)
        )
        checks.append(self._check_runtime_build(experiment, benchmarks, reasons, warnings))
        checks.append(self._check_conditions(experiment, benchmarks, warnings))
        checks.append(self._check_context(experiment, benchmarks))
        checks.append(self._check_provenance(experiment, benchmarks, reasons))
        checks.append(self._check_evidence(manifest.evidence_files, reasons))

        return _result(manifest.identity.case_id, checks, reasons, warnings)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_records(
        self, manifest: ExperimentalCaseManifest
    ) -> tuple[ExperimentRecord | None, list[BenchmarkRecord], list[str]]:
        missing: list[str] = []
        experiment: ExperimentRecord | None = None
        try:
            experiment = self.load_experiment(manifest.experiment_record_id)
        except JaullError as exc:
            missing.append(
                f"experiment record {manifest.experiment_record_id} "
                f"could not be loaded: {exc}"
            )

        benchmarks: list[BenchmarkRecord] = []
        for benchmark_id in manifest.benchmark_record_ids:
            try:
                benchmarks.append(self.load_benchmark(benchmark_id))
            except JaullError as exc:
                missing.append(
                    f"benchmark record {benchmark_id} could not be loaded: {exc}"
                )
        return experiment, benchmarks, missing

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def _check_machine(
        self,
        identity: ExperimentalCaseIdentity,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        reasons: list[str],
    ) -> CaseCheck:
        expected = tuple(identity.machine_fingerprint)
        mismatched: list[str] = []
        if machine_fingerprint(experiment.hardware) != expected:
            mismatched.append(
                f"experiment {experiment.identity.experiment_id} ran on another machine"
            )
        for benchmark in benchmarks:
            if machine_fingerprint(benchmark.hardware) != expected:
                mismatched.append(
                    f"benchmark {benchmark.identity.benchmark_id} ran on another machine"
                )
        if mismatched:
            reasons.extend(mismatched)
            return CaseCheck(
                name="machine_identity",
                status=CaseConsistencyStatus.INCONSISTENT,
                detail="; ".join(mismatched),
            )
        return CaseCheck(
            name="machine_identity",
            status=CaseConsistencyStatus.VALID,
            detail="verified",
        )

    def _check_artifact(
        self,
        identity: ExperimentalCaseIdentity,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        reasons: list[str],
    ) -> CaseCheck:
        mismatched: list[str] = []
        if ArtifactIdentity.of(experiment.artifact) != identity.artifact:
            mismatched.append(
                f"experiment {experiment.identity.experiment_id} "
                "used a different artifact than the case declares"
            )
        for benchmark in benchmarks:
            if not artifact_identity_matches(experiment.artifact, benchmark.artifact):
                mismatched.append(
                    f"benchmark {benchmark.identity.benchmark_id} "
                    "used a different artifact than the experiment"
                )
        if mismatched:
            reasons.extend(mismatched)
            return CaseCheck(
                name="artifact_identity",
                status=CaseConsistencyStatus.INCONSISTENT,
                detail="; ".join(mismatched),
            )
        return CaseCheck(
            name="artifact_identity",
            status=CaseConsistencyStatus.VALID,
            detail="verified",
        )

    def _check_artifact_hash(
        self,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        reasons: list[str],
    ) -> CaseCheck:
        digests = {experiment.artifact.sha256} | {
            benchmark.artifact.sha256 for benchmark in benchmarks
        }
        known = {digest for digest in digests if digest is not None}
        if len(known) > 1:
            reason = "records declare different artifact digests"
            reasons.append(reason)
            return CaseCheck(
                name="artifact_hash",
                status=CaseConsistencyStatus.INCONSISTENT,
                detail=reason,
            )
        if None in digests:
            # Metadata agreeing is not the same as the bytes being identical.
            # Say so rather than let "verified" mean two different things.
            reason = (
                "artifact digest is unavailable on at least one record; "
                "artifact identity is metadata-only, not cryptographically verified"
            )
            reasons.append(reason)
            return CaseCheck(
                name="artifact_hash",
                status=CaseConsistencyStatus.PARTIAL,
                detail="unavailable",
            )
        return CaseCheck(
            name="artifact_hash",
            status=CaseConsistencyStatus.VALID,
            detail="verified",
        )

    def _check_runtime_family(
        self,
        identity: ExperimentalCaseIdentity,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        reasons: list[str],
    ) -> CaseCheck:
        mismatched: list[str] = []
        if experiment.runtime.runtime is not identity.runtime:
            mismatched.append(
                f"experiment {experiment.identity.experiment_id} "
                f"used runtime {experiment.runtime.runtime.value}, "
                f"not {identity.runtime.value}"
            )
        for benchmark in benchmarks:
            if benchmark.runtime.runtime is not identity.runtime:
                mismatched.append(
                    f"benchmark {benchmark.identity.benchmark_id} "
                    f"used runtime {benchmark.runtime.runtime.value}, "
                    f"not {identity.runtime.value}"
                )
        if mismatched:
            reasons.extend(mismatched)
            return CaseCheck(
                name="runtime_family",
                status=CaseConsistencyStatus.INCONSISTENT,
                detail="; ".join(mismatched),
            )
        return CaseCheck(
            name="runtime_family",
            status=CaseConsistencyStatus.VALID,
            detail=identity.runtime.value,
        )

    def _check_runtime_build(
        self,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        reasons: list[str],
        warnings: list[str],
    ) -> CaseCheck:
        builds: list[str | None] = [
            runtime_capability_version(experiment.preflight.runtime_capability)
        ]
        for benchmark in benchmarks:
            if benchmark.llama_bench_capability is not None:
                builds.append(benchmark.llama_bench_capability.version_text)
            else:
                builds.append(runtime_capability_version(benchmark.runtime_capability))

        known = {build for build in builds if build is not None}
        if not known:
            reason = "no runtime build string was recorded for this case"
            reasons.append(reason)
            return CaseCheck(
                name="runtime_build",
                status=CaseConsistencyStatus.PARTIAL,
                detail="unavailable",
            )
        comparison_keys = {_runtime_build_key(build) for build in known}
        if len(comparison_keys) > 1:
            # A different build is a legitimate experimental condition, and the
            # experiment probes llama-cli while a benchmark probes llama-bench,
            # so these strings can differ without anything being wrong.
            warnings.append(
                "records were produced by different runtime builds: "
                + "; ".join(sorted(known))
            )
        if None in builds:
            reason = "runtime build is unavailable on at least one record"
            reasons.append(reason)
            return CaseCheck(
                name="runtime_build",
                status=CaseConsistencyStatus.PARTIAL,
                detail="partially unavailable",
            )
        if len(comparison_keys) > 1:
            return CaseCheck(
                name="runtime_build",
                status=CaseConsistencyStatus.VALID,
                detail="differs between records; see warnings",
            )
        return CaseCheck(
            name="runtime_build",
            status=CaseConsistencyStatus.VALID,
            detail=next(iter(sorted(known))),
        )

    def _check_conditions(
        self,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        warnings: list[str],
    ) -> CaseCheck:
        """Instantaneous conditions. These never lower the status.

        Free memory moves between two runs of the same machine, and a driver
        can be upgraded between them. Neither makes the case wrong; both are
        worth saying out loud, because both can explain a difference in the
        numbers the records hold.
        """

        differences: list[str] = []
        for benchmark in benchmarks:
            differences.extend(
                f"benchmark {benchmark.identity.benchmark_id}: {difference}"
                for difference in _condition_differences(
                    experiment.hardware, benchmark.hardware
                )
            )
        if differences:
            warnings.extend(differences)
            return CaseCheck(
                name="execution_conditions",
                status=CaseConsistencyStatus.VALID,
                detail=f"{len(differences)} difference(s), reported as warnings",
            )
        return CaseCheck(
            name="execution_conditions",
            status=CaseConsistencyStatus.VALID,
            detail="identical",
        )

    def _check_context(
        self,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
    ) -> CaseCheck:
        """Compare declared benchmark scenarios without claiming enforcement.

        llama-bench has no direct context-size flag. New records therefore
        retain the experiment's intended workload context as provenance only;
        the value cannot certify the runtime's allocation behaviour.
        """

        context = experiment.prediction.inference_configuration.context_length
        detail = f"experiment ran at {context} tokens"
        if not benchmarks:
            return CaseCheck(
                name="context",
                status=CaseConsistencyStatus.VALID,
                detail=detail,
            )
        declared = {benchmark.request.context_length for benchmark in benchmarks}
        if declared == {context}:
            return CaseCheck(
                name="context",
                status=CaseConsistencyStatus.VALID,
                detail=(
                    detail
                    + "; benchmark scenario declares the same context "
                    "(not runner-enforced)"
                ),
            )
        if None in declared:
            return CaseCheck(
                name="context",
                status=CaseConsistencyStatus.PARTIAL,
                detail=detail + "; at least one benchmark has no declared context",
            )
        return CaseCheck(
            name="context",
            status=CaseConsistencyStatus.INCONSISTENT,
            detail=detail + "; benchmark declared context differs",
        )

    def _check_provenance(
        self,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
        reasons: list[str],
    ) -> CaseCheck:
        """What the records cannot tell us, stated rather than glossed over."""

        gaps: list[str] = []
        if experiment.environment.git_commit is None or any(
            benchmark.environment.git_commit is None for benchmark in benchmarks
        ):
            gaps.append("the Jaull git commit was not recorded")
        if not experiment.backend_trace.executed_command:
            gaps.append("the experiment did not record the command it executed")
        if experiment.backend_trace.observed_backend is None:
            gaps.append("the backend the experiment actually used was not observed")

        reasons.extend(gaps)
        if not gaps:
            return CaseCheck(
                name="provenance",
                status=CaseConsistencyStatus.VALID,
                detail="complete",
            )
        return CaseCheck(
            name="provenance",
            status=CaseConsistencyStatus.PARTIAL,
            detail="; ".join(gaps),
        )

    def _check_evidence(
        self,
        references: tuple[EvidenceFileReference, ...],
        reasons: list[str],
    ) -> CaseCheck:
        if not references:
            return CaseCheck(
                name="evidence_files",
                status=CaseConsistencyStatus.VALID,
                detail="none referenced",
            )

        problems: list[str] = []
        for reference in references:
            path = self.evidence_root / reference.path
            if not path.is_file():
                problems.append(f"{reference.path} is missing")
                continue
            if (
                reference.size_bytes is not None
                and path.stat().st_size != reference.size_bytes
            ):
                problems.append(f"{reference.path} does not match its recorded size")
                continue
            if reference.sha256 is None:
                continue
            actual = _file_digest(path)
            if actual is None:
                problems.append(f"{reference.path} could not be read")
            elif actual != reference.sha256:
                problems.append(f"{reference.path} does not match its recorded digest")

        if problems:
            reasons.extend(problems)
            return CaseCheck(
                name="evidence_files",
                status=CaseConsistencyStatus.PARTIAL,
                detail="; ".join(problems),
            )
        return CaseCheck(
            name="evidence_files",
            status=CaseConsistencyStatus.VALID,
            detail=f"{len(references)} file(s) present",
        )


def _runtime_build_key(build: str) -> str:
    """Compare llama.cpp build strings across cli and bench output formats."""

    match = _LLAMA_CPP_BUILD.search(build)
    if match is None:
        return build
    number = match.group("number") or match.group("build_number")
    commit = match.group("commit") or match.group("build_commit")
    assert number is not None and commit is not None
    return f"llama.cpp {number} ({commit.casefold()})"


def _condition_differences(left: HardwareProfile, right: HardwareProfile) -> list[str]:
    differences: list[str] = []
    if left.memory.available_bytes != right.memory.available_bytes:
        differences.append(
            "available RAM differed at scan time "
            f"({left.memory.available_bytes} vs {right.memory.available_bytes} bytes)"
        )
    for index, (one, other) in enumerate(zip(left.gpus, right.gpus, strict=False)):
        if one.vram_available_bytes != other.vram_available_bytes:
            differences.append(
                f"available VRAM on GPU {index} differed at scan time "
                f"({one.vram_available_bytes} vs {other.vram_available_bytes} bytes)"
            )
        if one.driver_version != other.driver_version:
            differences.append(
                f"GPU {index} driver differed "
                f"({one.driver_version} vs {other.driver_version})"
            )
        if one.cuda_version != other.cuda_version:
            differences.append(
                f"GPU {index} CUDA version differed "
                f"({one.cuda_version} vs {other.cuda_version})"
            )
    return differences


def _file_digest(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK_BYTES):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _result(
    case_id: str,
    checks: list[CaseCheck],
    reasons: list[str],
    warnings: list[str],
) -> CaseValidationResult:
    statuses = {check.status for check in checks}
    if CaseConsistencyStatus.INCONSISTENT in statuses:
        status = CaseConsistencyStatus.INCONSISTENT
    elif CaseConsistencyStatus.PARTIAL in statuses:
        status = CaseConsistencyStatus.PARTIAL
    else:
        status = CaseConsistencyStatus.VALID
    return CaseValidationResult(
        case_id=case_id,
        status=status,
        checks=tuple(checks),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


__all__ = ["CaseValidationService", "identity_for_experiment"]
