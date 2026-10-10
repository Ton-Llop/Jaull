"""Offline selection/queue regressions; synthetic models, no Hub/Docker/GPU."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jaull.advisor.quality_candidates import (
    QualityCandidate,
    prepare_candidates,
    run_candidates,
)
from jaull.application.recommendation.service import recommend
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import (
    CompatibilityStatus,
    HardwareFitMode,
    HardwareFitResult,
    HardwareMemoryTopology,
)
from jaull.domain.execution_plans import IdentityMatchStatus
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.requirements import RecommendationPriority, UseCase, UserRequirements
from jaull.recommendation.engine_v2 import PlanRankingContext
from jaull.runtime.quality_eval_runner import QualityEvaluationCancelled, QualityEvaluationError
from jaull.workflow.state import RecommendationWorkflowState
from tests._case_fixtures import benchmark_record, experiment_record
from tests._workflow_fixtures import gguf_analysis, hardware, memory_estimate
from tests.test_cli_quality import _request
from tests.test_execution_plans import _gguf_recommendation, _selection
from tests.test_tui_evidence import _plan


class SelectionAdvisor:
    services = SimpleNamespace(capability_analyzer=None)

    def __init__(self, count: int = 8) -> None:
        self.models = []
        self.plans: dict[str, Any] = {}
        self.calls: list[str] = []
        self.fit = HardwareFitMode.GPU_RESIDENT
        self.backend = ComputeBackend.CUDA
        self.cancelled = False
        for index in range(count):
            plan = _plan(repo_id=f"synthetic/Model-{index}", sha256=f"{index + 1:064x}")
            plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
                "filename": "model-Q4_K_M.gguf", "size_bytes": 1024,
            })})
            rec = _gguf_recommendation()
            evaluated = rec.evaluated.model_copy(update={
                "candidate": rec.evaluated.candidate.model_copy(update={
                    "repo_id": plan.artifact.repo_id,
                }),
            })
            rec = rec.model_copy(update={"plan": plan, "evaluated": evaluated})
            self.models.append(rec)
            self.plans[rec.repo_id] = [plan]

    def state(self) -> RecommendationWorkflowState:
        return RecommendationWorkflowState(
            hardware=hardware(),
            requirements=UserRequirements(
                use_case=UseCase.GENERAL_CHAT, priority=RecommendationPriority.QUALITY,
                desired_context=4096, pipeline_tag="text-generation",
            ),
            evaluated_candidates=[rec.evaluated for rec in self.models],
            recommendations=self.models[:5],
        )

    def select_runtime_backend(self, hw: Any) -> Any:
        return _selection(self.backend)

    def _plan_ranking_context(self, hw: Any) -> PlanRankingContext:
        return PlanRankingContext(hardware=hw, backend_selection=self.select_runtime_backend(hw))

    def quality_evidence(self) -> list[Any]:
        return []

    def _stored_quality_records(self) -> list[Any]:
        return []

    def resolve_model_identity(self, rec: Any) -> Any:
        return rec.plan.model_identity

    def execution_plans_for_recommendation(self, rec: Any) -> list[Any]:
        self.calls.append(rec.repo_id)
        return self.plans[rec.repo_id]

    def _artifacts(self) -> Any:
        def resolve(repo: str, **kwargs: Any) -> ModelArtifact:
            plan = self.plans[repo][0]
            return plan.artifact.to_model_artifact().model_copy(update={
                "revision": "f" * 40, "sha256": plan.artifact.sha256,
            })
        return SimpleNamespace(
            resolver=SimpleNamespace(resolve=resolve), is_downloaded=lambda _: False,
        )

    def inspect_model(self, repo: str) -> Any:
        analysis = gguf_analysis(repo, quantizations=("Q4_K_M",))
        variant = analysis.classification.gguf_variants[0]
        exact = variant.model_copy(update={
            "total_bytes": 1024, "files": [variant.files[0].model_copy(update={
                "size_bytes": 1024, "sha256": self.plans[repo][0].artifact.sha256,
            })],
        })
        return analysis.model_copy(update={"classification": analysis.classification.model_copy(
            update={"gguf_variants": [exact]},
        )})

    def estimate_model(self, analysis: Any, hw: Any, cfg: InferenceConfiguration, **kw: Any) -> Any:
        assert cfg.context_length in (2048, 4096) and cfg.batch_size == cfg.concurrent_users == 1
        assert len(analysis.classification.gguf_variants) == 1
        return memory_estimate(analysis, cfg, total_bytes=1024).model_copy(update={
            "hardware_fit": HardwareFitResult(
                mode=self.fit, memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
                weights_bytes=1024, kv_cache_bytes=0, overhead_bytes=0,
                reason="Synthetic full-device fit",
            ),
        })


def _pool(monkeypatch: pytest.MonkeyPatch, advisor: SelectionAdvisor) -> list[Any]:
    observed: list[Any] = []
    def recommend(evaluated: Any, requirements: Any, **kwargs: Any) -> Any:
        observed.append((evaluated, kwargs["limit"], kwargs["plan_context"]))
        return advisor.models
    monkeypatch.setattr("jaull.advisor.quality_candidates.service.recommend", recommend)
    return observed


def test_selection_uses_inspected_pool_beyond_top_five_without_mutating_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor()
    observed = _pool(monkeypatch, advisor)
    for rec in advisor.models[:5]:
        plan = rec.plan
        assert plan is not None
        advisor.plans[rec.repo_id] = [plan.model_copy(update={
            "artifact": plan.artifact.model_copy(update={"file_count": 3}),
        })]
    state = advisor.state()
    before = state.model_dump_json()
    result = prepare_candidates(advisor, state)  # type: ignore[arg-type]
    assert observed[0][1] == 8
    assert result.candidates[0].plan.artifact.repo_id == advisor.models[5].repo_id
    assert len(advisor.calls) == 6 and result.considered == 8
    assert "budget" in result.notices[-1]
    assert all("single-file" in notice for notice in result.notices[:-1])
    assert state.model_dump_json() == before  # Includes ScoreBreakdown/order.


def test_selection_is_bounded_pins_artifacts_and_does_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor()
    _pool(monkeypatch, advisor)
    result = prepare_candidates(advisor, advisor.state())  # type: ignore[arg-type]
    assert len(result.candidates) == len(advisor.calls) == 3
    assert all(candidate.plan.artifact.revision == "f" * 40 for candidate in result.candidates)
    assert all(candidate.plan.artifact.sha256 for candidate in result.candidates)
    assert all(not candidate.downloaded for candidate in result.candidates)


def test_selection_preserves_the_advisors_existing_ranking_evidence_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor()
    state = advisor.state()
    context = replace(advisor._plan_ranking_context(state.hardware),
                      benchmark_records=(benchmark_record(),),
                      experiment_records=(experiment_record(),))
    observed = _pool(monkeypatch, advisor)
    monkeypatch.setattr(advisor, "_plan_ranking_context", lambda _: context)
    result = prepare_candidates(advisor, state)  # type: ignore[arg-type]
    assert result.candidates
    # Forward the complete facade context, not a copy dropping local records/readiness.
    assert observed[0][2] is context
    assert observed[0][2].benchmark_records and observed[0][2].experiment_records


@pytest.mark.parametrize("change", ["offload", "missing_sha", "uncertain", "wrong_model", "cpu"])
def test_unusable_evaluation_candidates_are_explained_not_recommended_away(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    advisor = SelectionAdvisor(1)
    _pool(monkeypatch, advisor)
    plan = advisor.plans[advisor.models[0].repo_id][0]
    if change == "offload":
        advisor.fit = HardwareFitMode.GPU_OFFLOAD
    elif change == "cpu":
        advisor.backend = ComputeBackend.CPU
    elif change == "wrong_model":
        plan = _plan(repo_id="other/wrong", sha256="d" * 64)
    else:
        plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
            "sha256": None if change == "missing_sha" else plan.artifact.sha256,
            "identity_match": IdentityMatchStatus.UNCERTAIN if change == "uncertain"
            else IdentityMatchStatus.CONFIRMED,
        })})
    advisor.plans[advisor.models[0].repo_id] = [plan]
    state = advisor.state()
    before = state.model_dump_json()
    result = prepare_candidates(advisor, state)  # type: ignore[arg-type]
    assert not result.candidates and result.notices
    assert state.model_dump_json() == before


def test_selection_cancellation_stops_before_the_next_metadata_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor()
    _pool(monkeypatch, advisor)
    cancelled = False
    def progress(text: str) -> None:
        nonlocal cancelled
        if text.startswith("Inspecting evaluation paths"):
            cancelled = True
    with pytest.raises(QualityEvaluationCancelled):
        prepare_candidates(advisor, advisor.state(), is_cancelled=lambda: cancelled,
                           on_progress=progress)  # type: ignore[arg-type]
    assert len(advisor.calls) <= 1


def test_bad_variant_does_not_hide_a_valid_path_for_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor(1)
    _pool(monkeypatch, advisor)
    good = advisor.plans[advisor.models[0].repo_id][0]
    bad = good.model_copy(update={
        "plan_id": "000-bad", "artifact": good.artifact.model_copy(update={
            "revision": "old-tag",
        }),
    })
    advisor.plans[advisor.models[0].repo_id] = [bad, good]
    artifacts = advisor._artifacts()
    resolve = artifacts.resolver.resolve
    def fail_old(repo: str, **kwargs: Any) -> Any:
        if kwargs["revision"] == "old-tag":
            raise QualityEvaluationError("Synthetic unresolved revision")
        return resolve(repo, **kwargs)
    artifacts.resolver.resolve = fail_old
    monkeypatch.setattr(advisor, "_artifacts", lambda: artifacts)
    result = prepare_candidates(advisor, advisor.state())  # type: ignore[arg-type]
    assert len(result.candidates) == 1
    assert result.candidates[0].plan.plan_id == good.plan_id


def test_stale_artifact_metadata_is_not_used_to_estimate_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor(1)
    _pool(monkeypatch, advisor)
    analysis = advisor.inspect_model(advisor.models[0].repo_id)
    variant = analysis.classification.gguf_variants[0].model_copy(update={"total_bytes": 999})
    analysis = analysis.model_copy(update={"classification": analysis.classification.model_copy(
        update={"gguf_variants": [variant]},
    )})
    monkeypatch.setattr(advisor, "inspect_model", lambda _: analysis)
    def forbidden_estimate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Stale metadata must not reach the estimator")
    monkeypatch.setattr(advisor, "estimate_model", forbidden_estimate)
    result = prepare_candidates(advisor, advisor.state())  # type: ignore[arg-type]
    assert not result.candidates and "digest/size" in result.notices[0]


def test_evaluation_fit_alone_is_not_fit_for_the_requested_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor(1)
    _pool(monkeypatch, advisor)
    estimate = advisor.estimate_model
    def workload_does_not_fit(analysis: Any, hw: Any, config: Any, **kwargs: Any) -> Any:
        result = estimate(analysis, hw, config, **kwargs)
        if config.context_length == 4096:
            result = result.model_copy(update={"assessment": result.assessment.model_copy(
                update={"status": CompatibilityStatus.INSUFFICIENT},
            )})
        return result
    monkeypatch.setattr(advisor, "estimate_model", workload_does_not_fit)
    result = prepare_candidates(advisor, advisor.state())  # type: ignore[arg-type]
    assert not result.candidates and "search workload" in result.notices[0]


def test_real_recommender_supplies_all_inspected_logical_models_without_score_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor = SelectionAdvisor(8)
    evaluated = []
    for index, rec in enumerate(advisor.models):
        assert rec.evaluated.analysis is not None
        evaluated.append(rec.evaluated.model_copy(update={
            "candidate": rec.evaluated.candidate.model_copy(update={
                "base_model_repo_id": f"synthetic/Base-{index}",
            }),
            "analysis": rec.evaluated.analysis.model_copy(update={
                "repo": rec.evaluated.analysis.repo.model_copy(update={"repo_id": rec.repo_id}),
            }),
        }))
    state = advisor.state().model_copy(update={"evaluated_candidates": evaluated})
    assert state.requirements is not None
    baseline = recommend(evaluated, state.requirements, hardware=state.hardware, limit=5)
    state = state.model_copy(update={"recommendations": baseline})
    before = state.model_dump_json()
    original = advisor.execution_plans_for_recommendation
    def actual_identity_paths(rec: Any) -> Any:
        return [plan.model_copy(update={
            "model_identity": rec.plan.model_identity,
            "artifact": plan.artifact.model_copy(update={
                "model_identity": rec.plan.model_identity,
            }),
        }) for plan in original(rec)]
    monkeypatch.setattr(advisor, "execution_plans_for_recommendation", actual_identity_paths)
    result = prepare_candidates(advisor, state)  # type: ignore[arg-type]
    assert result.considered == 8 and len(result.candidates) == 3
    assert state.model_dump_json() == before
    assert recommend(evaluated, state.requirements, hardware=state.hardware, limit=5) == baseline


class QueueAdvisor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.cancelled = False
        self.fail_at = -1
        self.cancel_at = -1

    def prepare_quality_evaluation_setup(self, request: Any, **kw: Any) -> None:
        if kw["is_cancelled"] and kw["is_cancelled"]():
            raise QualityEvaluationCancelled("Cancelled")

    def run_quality_evaluation_for_plan(self, plan: Any, request: Any, **kw: Any) -> Path:
        assert not kw["allow_download"]
        self.calls.append((plan, request))
        if len(self.calls) == self.fail_at:
            raise QualityEvaluationError("Synthetic artifact failure")
        if len(self.calls) == self.cancel_at:
            self.cancelled = True
            raise QualityEvaluationCancelled("Cleaned up")
        return request.output / "stored.json"


def _candidates() -> tuple[QualityCandidate, ...]:
    return tuple(QualityCandidate(_plan(repo_id=f"org/Model-{i}"), "synthetic", True)
                 for i in range(3))


def test_queue_sequential_outputs_failure_and_prior_success_survive(tmp_path: Path) -> None:
    advisor = QueueAdvisor()
    advisor.fail_at = 2
    request = replace(_request(tmp_path / "input"), output=tmp_path / "batch")
    result = run_candidates(advisor, _candidates(), request)  # type: ignore[arg-type]
    assert len(advisor.calls) == len(result) == 3
    assert result[0].path and result[2].path and result[1].error
    assert [r.output.name for _, r in advisor.calls] == ["model-01", "model-02", "model-03"]
    assert all(r.profile == request.profile for _, r in advisor.calls)
    assert len({r.output for _, r in advisor.calls}) == 3
    with pytest.raises(FileExistsError):
        run_candidates(advisor, _candidates(), request)  # type: ignore[arg-type]
    assert len(advisor.calls) == 3


def test_queue_cancellation_does_not_start_later_candidates(tmp_path: Path) -> None:
    advisor = QueueAdvisor()
    advisor.cancel_at = 2
    request = replace(_request(tmp_path / "input"), output=tmp_path / "batch")
    result = run_candidates(advisor, _candidates(), request,
                            is_cancelled=lambda: advisor.cancelled)  # type: ignore[arg-type]
    assert len(advisor.calls) == len(result) == 2
    assert result[0].path is not None and result[1].error == "Cancelled"


@pytest.mark.parametrize("count", [0, 4])
def test_queue_rejects_unbounded_or_empty_request(tmp_path: Path, count: int) -> None:
    advisor = QueueAdvisor()
    candidates = (_candidates() * 2)[:count]
    with pytest.raises(ValueError, match="one and three"):
        run_candidates(advisor, candidates, _request(tmp_path))  # type: ignore[arg-type]
    assert not advisor.calls
