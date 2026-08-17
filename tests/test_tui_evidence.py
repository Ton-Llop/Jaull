"""How much evidence stands behind an execution plan.

These cover the resolver that replaced the substring scan of
``ExecutionPlan.evidence``. That scan looked for ``"validated"`` and
``"benchmarked"``, which ``_plan_evidence`` never emits, so both states were
permanently unreachable and the two filters were permanently empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
)
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.tui.evidence import EvidenceIndex, EvidenceState
from tests.test_experiment_store import _sample_record


class _FakeStoreAdvisor:
    """Only the four record-store methods the resolver actually calls."""

    def __init__(
        self,
        experiments: dict[str, Any] | None = None,
        benchmarks: dict[str, Any] | None = None,
        *,
        broken: bool = False,
    ) -> None:
        self._experiments = experiments or {}
        self._benchmarks = benchmarks or {}
        self._broken = broken

    def list_experiment_ids(self) -> list[str]:
        if self._broken:
            raise OSError("data directory unreadable")
        return list(self._experiments)

    def load_experiment_record(self, experiment_id: str) -> Any:
        return self._experiments[experiment_id]

    def list_benchmark_ids(self) -> list[str]:
        if self._broken:
            raise OSError("data directory unreadable")
        return list(self._benchmarks)

    def load_benchmark_record(self, benchmark_id: str) -> Any:
        return self._benchmarks[benchmark_id]


def _plan(
    *,
    repo_id: str = "owner/repo",
    quantization: str | None = "Q4_K_M",
    runtime: RuntimeName = RuntimeName.LLAMA_CPP,
    ready: bool = True,
) -> ExecutionPlan:
    identity = ModelIdentity(model_name=repo_id.split("/")[-1])
    variant = ArtifactVariant(
        model_identity=identity,
        repo_id=repo_id,
        revision="main",
        format=ArtifactVariantFormat.GGUF,
        filename="model.gguf",
        quantization=quantization,
        compatible_runtimes=[runtime],
    )
    recommendation = RuntimeRecommendation(
        runtime=runtime,
        confidence=EstimationConfidence.HIGH,
    )
    # A real readiness object rather than a hand-built one: it carries a
    # selection and a runtime capability, and the resolver only reads `.status`.
    readiness = _sample_record().preflight.execution_readiness if ready else None
    return ExecutionPlan(
        plan_id=f"{repo_id}:{quantization}",
        model_identity=identity,
        artifact=variant,
        runtime=recommendation,
        runtime_family=runtime,
        execution_readiness=readiness,
    )


def _experiment(
    *,
    repo_id: str = "owner/repo",
    quantization: str = "Q4_K_M",
    success: bool = True,
) -> Any:
    record = _sample_record()
    artifact = record.artifact.model_copy(
        update={"repo_id": repo_id, "quantization": quantization}
    )
    observation = record.observation.model_copy(update={"success": success})
    return record.model_copy(update={"artifact": artifact, "observation": observation})


def test_plan_with_no_records_is_ready_not_validated() -> None:
    index = EvidenceIndex.load(_FakeStoreAdvisor(), ModelIdentity(model_name="repo"))

    evidence = index.for_plan(_plan())

    assert evidence.state is EvidenceState.READY
    assert evidence.validated is False
    assert evidence.benchmarked is False


def test_plan_without_readiness_is_only_estimated() -> None:
    index = EvidenceIndex.empty()

    evidence = index.for_plan(_plan(ready=False))

    assert evidence.state is EvidenceState.ESTIMATED
    assert evidence.summary() == "Estimated only"


def test_a_stored_experiment_makes_the_plan_validated() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment()})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))
    evidence = index.for_plan(_plan())

    assert evidence.state is EvidenceState.VALIDATED
    assert evidence.experiment_ids == ("exp-1",)
    assert "Validated" in evidence.summary()


def test_a_failed_experiment_does_not_count_as_validated() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment(success=False)})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan()).state is EvidenceState.READY


def test_evidence_does_not_cross_quantizations() -> None:
    """A Q4_K_M run says nothing about Q5_K_M: different artifact, different run."""
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment(quantization="Q4_K_M")})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(quantization="Q4_K_M")).validated is True
    assert index.for_plan(_plan(quantization="Q5_K_M")).validated is False


def test_evidence_does_not_cross_runtimes() -> None:
    """A llama.cpp GGUF run is not evidence for a Transformers plan."""
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment()})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(runtime=RuntimeName.LLAMA_CPP)).validated is True
    assert index.for_plan(_plan(runtime=RuntimeName.TRANSFORMERS)).validated is False


def test_evidence_does_not_cross_repositories() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment(repo_id="owner/repo")})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(repo_id="other/repo")).validated is False


def test_loading_without_an_identity_indexes_every_model() -> None:
    """The results screen scans the stores once for several models at a time."""
    advisor = _FakeStoreAdvisor(
        experiments={
            "exp-1": _experiment(repo_id="owner/repo"),
            "exp-2": _experiment(repo_id="other/second"),
        }
    )

    index = EvidenceIndex.load(advisor)

    assert index.for_plan(_plan(repo_id="owner/repo")).validated is True
    assert index.for_plan(_plan(repo_id="other/second")).validated is True


def test_an_unreadable_store_degrades_to_no_evidence() -> None:
    """Evidence enriches a screen; a plan must not vanish because a log failed."""
    index = EvidenceIndex.load(_FakeStoreAdvisor(broken=True), ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan()).state is EvidenceState.READY


def test_a_corrupt_record_is_skipped_rather_than_fatal() -> None:
    class _Exploding(_FakeStoreAdvisor):
        def load_experiment_record(self, experiment_id: str) -> Any:
            raise ValueError("truncated json")

    advisor = _Exploding(experiments={"exp-1": None})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan()).validated is False


def test_artifact_without_quantization_falls_back_to_the_filename() -> None:
    record = _sample_record()
    artifact: ModelArtifact = record.artifact.model_copy(
        update={"quantization": None, "filename": "model.gguf"}
    )
    advisor = _FakeStoreAdvisor(
        experiments={"exp-1": record.model_copy(update={"artifact": artifact})}
    )

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(quantization=None)).validated is True


def test_sample_record_artifact_path_is_untouched() -> None:
    """Guards the fixture assumption the tests above are built on."""
    assert _sample_record().artifact.local_path == Path("/models/model.gguf")
