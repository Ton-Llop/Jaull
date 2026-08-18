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

It only reads. No metric is recomputed, no methodology is reinterpreted, and a
record that fails to load is skipped rather than guessed at. Because the stores
are flat directories with no index, building the index is O(records) disk I/O
and must run on a worker thread, never on the event loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import (
    ExecutionPlan,
    ModelIdentity,
    logical_model_repo_key,
)
from jaull.domain.runtime import RuntimeName
from jaull.presentation.plan_labels import is_ready_plan

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
    """Every stored record that belongs to one plan."""

    state: EvidenceState
    experiment_ids: tuple[str, ...] = ()
    benchmark_ids: tuple[str, ...] = ()

    @property
    def validated(self) -> bool:
        return bool(self.experiment_ids)

    @property
    def benchmarked(self) -> bool:
        return bool(self.benchmark_ids)

    def summary(self) -> str:
        """The evidence line: every state earned, weakest first."""
        if self.state is EvidenceState.ESTIMATED:
            return "Estimated only"
        parts = ["○ Ready"]
        if self.validated:
            parts.append("✓ Validated")
        if self.benchmarked:
            parts.append("✓ Benchmarked")
        return " · ".join(parts)


@dataclass(frozen=True)
class EvidenceIndex:
    """Stored records for one logical model, grouped by execution plan.

    Build it once per screen load with :meth:`load`, then query it per plan.
    """

    _experiments: dict[str, list[str]] = field(default_factory=dict)
    _benchmarks: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> EvidenceIndex:
        return cls()

    @classmethod
    def load(
        cls,
        advisor: AdvisorService,
        identity: ModelIdentity | None = None,
    ) -> EvidenceIndex:
        """Read both stores. Blocking: call from a worker thread.

        With no ``identity`` every record is indexed, which is what a screen
        showing several models wants: the stores are scanned once rather than
        once per model, and the artifact key already carries the repository so
        two models cannot collide.

        Evidence enriches a screen; it never gates one. Anything unreadable —
        a missing store, an unwritable data directory, a half-written file —
        degrades to "no evidence" rather than to an error, because a plan the
        user can still run must not disappear because a log could not be read.
        """
        experiments: dict[str, list[str]] = {}
        benchmarks: dict[str, list[str]] = {}

        for experiment_id in _safe_ids(advisor.list_experiment_ids):
            record = _safe_load(advisor.load_experiment_record, experiment_id)
            if record is None or not record.observation.success:
                continue
            if identity is not None and not _matches_model(record.artifact, identity):
                continue
            key = _artifact_key(record.artifact, record.runtime.runtime)
            experiments.setdefault(key, []).append(experiment_id)

        for benchmark_id in _safe_ids(advisor.list_benchmark_ids):
            benchmark = _safe_load(advisor.load_benchmark_record, benchmark_id)
            if benchmark is None or not benchmark.observation.success:
                continue
            if identity is not None and not _matches_model(benchmark.artifact, identity):
                continue
            key = _artifact_key(benchmark.artifact, benchmark.runtime.runtime)
            benchmarks.setdefault(key, []).append(benchmark_id)

        return cls(_experiments=experiments, _benchmarks=benchmarks)

    def for_plan(self, plan: ExecutionPlan) -> PlanEvidence:
        key = _plan_key(plan)
        experiments = tuple(self._experiments.get(key, ()))
        benchmarks = tuple(self._benchmarks.get(key, ()))
        if benchmarks:
            state = EvidenceState.BENCHMARKED
        elif experiments:
            state = EvidenceState.VALIDATED
        elif is_ready_plan(plan):
            state = EvidenceState.READY
        else:
            state = EvidenceState.ESTIMATED
        return PlanEvidence(
            state=state,
            experiment_ids=experiments,
            benchmark_ids=benchmarks,
        )


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
    artifact = plan.artifact
    variant = (
        artifact.quantization or artifact.filename or artifact.format.value
    ).casefold()
    return (
        f"{artifact.repo_id.casefold()}|{plan.runtime_family.value}|"
        f"{artifact.format.value.casefold()}|{variant}"
    )


__all__ = [
    "EvidenceIndex",
    "EvidenceState",
    "PlanEvidence",
    "state_class",
    "state_glyph",
    "state_label",
]
