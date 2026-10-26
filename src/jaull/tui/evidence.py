"""How much evidence stands behind an execution plan.

The paths screen used to answer this by scanning ``ExecutionPlan.evidence`` for
the substrings ``"validated"`` and ``"benchmarked"``. That list is built by
``execution_plans.service._plan_evidence``, which only ever emits
``artifact:``, ``identity_match:``, ``quantization:``, ``readiness:`` and
``readiness_reason:`` — so neither substring could ever match, the two filters
were permanently empty, and every plan read "Estimated only" no matter how
many times it had actually been run.

The records exist; nothing was reading them. Validation writes an
``ExperimentRecord`` and benchmarking writes a ``BenchmarkRecord``, both
persisted as JSON under the user data directory. This module loads them and
matches them back to plans.

Quality records are historical results for exact artifact bytes, not evidence
that the current execution plan or evaluation protocol was validated.

It only reads. No metric is recomputed, no methodology is reinterpreted, and a
record that fails to load is skipped rather than guessed at. Because the stores
are flat directories with no index, building the index is O(records) disk I/O
and must run on a worker thread, never on the event loop.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.execution_plans import (
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
    logical_model_repo_key,
)
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeName
from jaull.evaluation.hardware_fingerprint import (
    EVIDENCE_MEMORY_GRANULARITY,
    machine_fingerprint,
)
from jaull.evaluation.quality_records import QualityEvidence, describe_record
from jaull.presentation.plan_labels import is_ready_plan
from jaull.presentation.published_evaluations import published_reference_text
from jaull.presentation.speed import historical_speed_benchmarks
from jaull.recommendation.capability_catalog import CatalogReadResult
from jaull.runtime.device_matching import (
    names_agree,
    runtime_device_name_matches,
    selected_accelerator_matches,
)

_log = logging.getLogger(__name__)


class EvidenceState(StrEnum):
    """What is known about a plan, weakest to strongest.

    The order is the point: a benchmarked plan has been measured, a validated
    plan has been run once, a ready plan has passed preflight, and an estimated
    plan has only been predicted.
    """

    ESTIMATED = "estimated"
    READY = "ready"
    VALIDATED = "validated"
    BENCHMARKED = "benchmarked"


_LABELS: dict[EvidenceState, str] = {
    EvidenceState.ESTIMATED: "Estimated",
    EvidenceState.READY: "Ready",
    EvidenceState.VALIDATED: "Validated",
    EvidenceState.BENCHMARKED: "Benchmarked",
}

# A glyph as well as a colour, so the distinction survives a monochrome
# terminal and does not depend on colour vision.
_GLYPHS: dict[EvidenceState, str] = {
    EvidenceState.ESTIMATED: "·",
    EvidenceState.READY: "○",
    EvidenceState.VALIDATED: "✓",
    EvidenceState.BENCHMARKED: "✓",
}


def state_label(state: EvidenceState) -> str:
    return _LABELS[state]


def state_glyph(state: EvidenceState) -> str:
    return _GLYPHS[state]


def state_class(state: EvidenceState) -> str:
    return f"status-{state.value}"


@dataclass(frozen=True)
class PlanEvidence:
    """Execution evidence for a plan and historical quality for its artifact."""

    state: EvidenceState
    experiment_ids: tuple[str, ...] = ()
    benchmark_ids: tuple[str, ...] = ()
    historical_benchmark_ids: tuple[str, ...] = ()
    quality_records: tuple[QualityEvidence, ...] = ()
    published_summary: str = "No published references loaded."
    published_provenance: str = ""

    @property
    def validated(self) -> bool:
        return bool(self.experiment_ids)

    @property
    def benchmarked(self) -> bool:
        return bool(self.benchmark_ids)

    @property
    def has_benchmark_history(self) -> bool:
        return bool(self.benchmark_ids or self.historical_benchmark_ids)

    def summary(self) -> str:
        """The evidence line: every state earned, weakest first."""
        parts = ["Estimated only" if self.state is EvidenceState.ESTIMATED else "○ Ready"]
        if self.validated:
            parts.append("✓ Validated")
        if self.benchmarked:
            parts.append("✓ Benchmarked")
        elif self.historical_benchmark_ids:
            parts.append("Benchmark history (current plan not verified)")
        return " · ".join(parts)

    def quality_summary(self) -> str:
        if not self.quality_records:
            return ""
        lines = ["Quality · historical artifact results; current execution/protocol not verified"]
        for record in self.quality_records:
            placement = ", ".join(f"{key} {value}" for key, value in record.placement.items())
            lines.append(
                f"{record.suite} · {record.classification} · {record.summary} · "
                f"{record.samples_used}/{record.samples_available} samples · "
                f"ctx {record.context_length or 'unknown'} · {placement or 'placement unknown'}"
            )
        return "\n".join(lines)

    def quality_details(self) -> str:
        return "\n\n".join(
            "\n".join([
                f"Suite: {record.suite}",
                "Placement: " + (
                    ", ".join(f"{key} {value}" for key, value in record.placement.items())
                    or "unknown"
                ),
                f"Artifact SHA256: {record.artifact_sha256}",
                f"Record identity: {record.identity_sha256}",
                f"Dataset: {record.dataset}@{record.dataset_revision}",
                f"Evaluator: {record.evaluator_commit}",
                f"Runtime: {record.runtime_fingerprint}",
                f"Hardware: {record.hardware or 'unknown'}",
                "Evaluated: " + (
                    record.evaluated_at.isoformat() if record.evaluated_at else "unknown"
                ),
                *record.limitations,
            ])
            for record in self.quality_records
        )

    def quality_readout(self) -> str:
        """Readable metrics with their scope, never an aggregate quality score."""
        if not self.quality_records:
            return ""
        sections = [
            "Historical results for this exact artifact; "
            "current execution/protocol not verified."
        ]
        for record in self.quality_records:
            lines = [
                f"Benchmark: {record.dataset}",
                f"Evaluation: {record.classification} evaluation",
                f"Samples: {record.samples_used} of {record.samples_available}",
                f"Evaluation context: {record.context_length or 'unknown'}",
                "",
            ]
            for metric in record.metrics:
                label = {
                    "acc": "Accuracy", "acc_norm": "Length-normalized accuracy",
                }.get(metric.name, metric.name)
                lines.append(
                    f"{label}: {metric.value:.1%} ({metric.correct}/{metric.samples})"
                )
            lines.extend(["", *record.limitations])
            sections.append("\n".join(lines))
        sections.append("These results are not a general capability assessment.")
        return "\n\n".join(sections)


@dataclass(frozen=True)
class EvidenceIndex:
    """Stored records for one logical model, grouped by execution plan.

    Build it once per screen load with :meth:`load`, then query it per plan.
    """

    _experiments: dict[str, list[tuple[str, ExperimentRecord]]] = field(default_factory=dict)
    _benchmarks: dict[str, list[tuple[str, BenchmarkRecord]]] = field(default_factory=dict)
    _quality: dict[str, list[QualityEvidence]] = field(default_factory=dict)
    _catalog: CatalogReadResult | None = None

    @classmethod
    def empty(cls) -> EvidenceIndex:
        return cls()

    @classmethod
    def load(
        cls,
        advisor: AdvisorService,
        identity: ModelIdentity | None = None,
    ) -> EvidenceIndex:
        """Read the evidence stores. Blocking: call from a worker thread.

        With no ``identity`` every record is indexed, which is what a screen
        showing several models wants: the stores are scanned once rather than
        once per model, and the artifact key already carries the repository so
        two models cannot collide. Quality records use the exact content SHA256
        instead, independent of repository labels and the current protocol.

        Evidence enriches a screen; it never gates one. Anything unreadable —
        a missing store, an unwritable data directory, a half-written file —
        degrades to "no evidence" rather than to an error, because a plan the
        user can still run must not disappear because a log could not be read.
        """
        experiments: dict[str, list[tuple[str, ExperimentRecord]]] = {}
        benchmarks: dict[str, list[tuple[str, BenchmarkRecord]]] = {}
        quality: dict[str, list[QualityEvidence]] = {}

        for experiment_id in _safe_ids(advisor.list_experiment_ids):
            record = _safe_load(advisor.load_experiment_record, experiment_id)
            if record is None or not record.observation.success:
                continue
            if identity is not None and not _matches_model(record.artifact, identity):
                continue
            key = _artifact_key(record.artifact, record.runtime.runtime)
            experiments.setdefault(key, []).append((experiment_id, record))

        for benchmark_id in _safe_ids(advisor.list_benchmark_ids):
            benchmark = _safe_load(advisor.load_benchmark_record, benchmark_id)
            if benchmark is None or not benchmark.observation.success:
                continue
            if identity is not None and not _matches_model(benchmark.artifact, identity):
                continue
            key = _artifact_key(benchmark.artifact, benchmark.runtime.runtime)
            benchmarks.setdefault(key, []).append((benchmark_id, benchmark))

        for quality_id in _safe_ids(advisor.list_quality_ids):
            result = _safe_load(
                lambda record_id: describe_record(advisor.load_quality_record(record_id)),
                quality_id,
            )
            if result is not None:
                quality.setdefault(result.artifact_sha256, []).append(result)

        # Older injected facades may expose only the record-store methods.
        catalog_reader = getattr(advisor, "capability_catalog", None)
        catalog = catalog_reader() if catalog_reader is not None else None
        return cls(
            _experiments=experiments, _benchmarks=benchmarks, _quality=quality, _catalog=catalog,
        )

    def for_plan(self, plan: ExecutionPlan) -> PlanEvidence:
        key = _plan_key(plan)
        experiments = tuple(
            experiment_id
            for experiment_id, record in self._experiments.get(key, ())
            if _experiment_matches_plan(record, plan)
        )
        benchmarks = tuple(
            benchmark_id
            for benchmark_id, record in self._benchmarks.get(key, ())
            if _benchmark_matches_plan(record, plan)
        )
        candidates = self._benchmarks.get(key, ())
        metadata_matches = {
            record.identity.benchmark_id
            for record in historical_speed_benchmarks(plan, [record for _, record in candidates])
        }
        history = tuple(
            record_id for record_id, record in candidates
            if plan.backend_selection is not None
            and record.requested_backend is plan.backend_selection.selected_backend
            and (
                _same_machine_and_artifact(record, plan)
                or record.identity.benchmark_id in metadata_matches
            )
        )
        if benchmarks:
            state = EvidenceState.BENCHMARKED
        elif experiments:
            state = EvidenceState.VALIDATED
        elif is_ready_plan(plan):
            state = EvidenceState.READY
        else:
            state = EvidenceState.ESTIMATED
        artifact = plan.artifact
        quality = (
            self.quality_for_sha(artifact.sha256)
            if artifact.sha256 is not None
            and artifact.format is ArtifactVariantFormat.GGUF
            and artifact.file_count == 1
            and plan.runtime_family is RuntimeName.LLAMA_CPP
            else ()
        )
        published, provenance = published_reference_text(plan.model_identity, self._catalog)
        return PlanEvidence(
            state=state,
            experiment_ids=experiments,
            benchmark_ids=benchmarks,
            historical_benchmark_ids=history,
            quality_records=quality,
            published_summary=published,
            published_provenance=provenance,
        )

    def quality_for_sha(self, sha256: str | None) -> tuple[QualityEvidence, ...]:
        """Historical content matches, not a current execution/protocol match."""
        return tuple(self._quality.get(sha256, ())) if sha256 is not None else ()

    def saved_experiments_for_plan(self, plan: ExecutionPlan) -> tuple[ExperimentRecord, ...]:
        """Every saved validation of this exact artifact on this machine, newest first.

        History, like :meth:`saved_benchmarks_for_plan`; the "Validated" tick
        keeps the stricter :meth:`for_plan` match.
        """
        records = (
            record
            for _, record in self._experiments.get(_plan_key(plan), ())
            if _same_machine_and_artifact(record, plan)
        )
        return tuple(sorted(records, key=lambda item: item.identity.created_at, reverse=True))

    def saved_benchmarks_for_plan(self, plan: ExecutionPlan) -> tuple[BenchmarkRecord, ...]:
        """Every saved probe of this exact artifact on this machine, newest first.

        History, not evidence for the plan: a probe may have used another backend,
        context or flags, and each record states its own configuration. The
        "Benchmarked" tick keeps the stricter :meth:`for_plan` match.
        """
        records = (
            record
            for _, record in self._benchmarks.get(_plan_key(plan), ())
            if _same_machine_and_artifact(record, plan)
        )
        return tuple(sorted(records, key=lambda item: item.identity.created_at, reverse=True))

    def benchmark_records(self) -> tuple[BenchmarkRecord, ...]:
        """Already loaded records; no further I/O for Results speed readouts."""
        return tuple(record for group in self._benchmarks.values() for _, record in group)


def _safe_ids(list_ids: Callable[[], list[str]]) -> list[str]:
    """A missing or unreadable store means no evidence, not a crashed screen."""
    try:
        return list(list_ids())
    except Exception as error:
        _log.debug("evidence store unreadable: %s", error)
        return []


def _safe_load[Record](load: Callable[[str], Record], record_id: str) -> Record | None:
    try:
        return load(record_id)
    except Exception as error:
        # A single corrupt or half-written file must not blank the whole
        # screen; the plan simply reports less evidence than it has.
        _log.debug("skipping unreadable record %s: %s", record_id, error)
        return None


def _matches_model(artifact: ModelArtifact, identity: ModelIdentity) -> bool:
    """Same test the advisor already applies when collecting benchmarks."""
    if (
        identity.canonical_repo_id
        and logical_model_repo_key(artifact.repo_id)
        == logical_model_repo_key(identity.canonical_repo_id)
    ):
        return True
    model_name = identity.model_name.casefold()
    if not model_name:
        return False
    return any(
        model_name in value.casefold()
        for value in (artifact.repo_id, artifact.filename)
    )


def _artifact_key(artifact: ModelArtifact, runtime: RuntimeName) -> str:
    """Identify the concrete thing that was executed.

    Repository, runtime and artifact together — a llama.cpp GGUF run is not
    evidence for a Transformers safetensors plan of the same model, and a
    Q4_K_M run is not evidence for Q5_K_M.
    """
    variant = (artifact.quantization or artifact.filename or artifact.format).casefold()
    return f"{artifact.repo_id.casefold()}|{runtime.value}|{artifact.format.casefold()}|{variant}"


def _plan_key(plan: ExecutionPlan) -> str:
    return _artifact_key(plan.artifact.to_model_artifact(), plan.runtime_family)


def _same_machine_and_artifact(
    record: ExperimentRecord | BenchmarkRecord, plan: ExecutionPlan
) -> bool:
    """The same bytes, run by the same runtime, on the same machine.

    Backend, context length and runtime flags are left out on purpose: that is
    what separates two probes of the same artifact from each other.
    """
    if plan.hardware is None:
        return False
    if machine_fingerprint(
        record.hardware, memory_granularity=EVIDENCE_MEMORY_GRANULARITY,
    ) != machine_fingerprint(plan.hardware, memory_granularity=EVIDENCE_MEMORY_GRANULARITY):
        return False
    if record.runtime.runtime is not plan.runtime_family:
        return False
    artifact = plan.artifact
    measured = record.artifact
    if artifact.sha256 is not None:
        if measured.sha256 != artifact.sha256:
            return False
    elif not (
        artifact.revision is not None
        and re.fullmatch(r"[0-9a-f]{40}", artifact.revision)
        and measured.revision == artifact.revision
        and measured.filename == artifact.filename
    ):
        # A mutable branch name or an unpinned discovery result cannot identify
        # the exact bytes of an earlier run.
        return False
    return True


def _experiment_matches_plan(record: ExperimentRecord, plan: ExecutionPlan) -> bool:
    """Validated on this machine, with this artifact, through this backend.

    Context length, launch flags and whether the plan carries a memory estimate
    do not decide it: a variant found by discovery has no estimate, and the saved
    record states its own configuration.
    """
    same_backend = (
        plan.backend_selection is not None
        and _same_machine_and_artifact(record, plan)
        and record.preflight.execution_readiness.selection.selected_backend
        is plan.backend_selection.selected_backend
        and record.backend_trace.observed_backend in {
            None, plan.backend_selection.selected_backend,
        }
    )
    if not same_backend or plan.backend_selection is None:
        return False
    if plan.backend_selection.selected_backend is ComputeBackend.CPU:
        return True
    expected = plan.backend_selection.selected_accelerator
    actual = record.preflight.execution_readiness.selection.selected_accelerator
    if not selected_accelerator_matches(expected, actual):
        return False
    assert expected is not None and plan.hardware is not None
    return bool(expected.uuid or expected.pci_bus_id) or sum(
        names_agree(item.name, expected.name) for item in plan.hardware.accelerators
    ) == 1


def _benchmark_matches_plan(record: BenchmarkRecord, plan: ExecutionPlan) -> bool:
    """Benchmarked on this machine, with this artifact, through this backend."""
    same_backend = (
        plan.backend_selection is not None
        and _same_machine_and_artifact(record, plan)
        and record.request.backend is plan.backend_selection.selected_backend
    )
    if not same_backend or plan.backend_selection is None:
        return False
    if plan.backend_selection.selected_backend is ComputeBackend.CPU:
        return True
    accelerator = plan.backend_selection.selected_accelerator
    return accelerator is not None and runtime_device_name_matches(
        "\n".join((record.observation.raw_stdout, record.observation.raw_stderr)),
        record.requested_device or "", accelerator.name,
    )


__all__ = [
    "EvidenceIndex",
    "EvidenceState",
    "PlanEvidence",
    "state_class",
    "state_glyph",
    "state_label",
]
