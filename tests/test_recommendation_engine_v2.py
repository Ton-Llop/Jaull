from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jaull.advisor.service import AdvisorService
from jaull.benchmarks.storage import BenchmarkStore
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkIdentity,
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRecord,
    BenchmarkRequest,
)
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.comparison import (
    CompatibilityComparison,
    CompatibilityOutcome,
    MetricComparison,
    MetricComparisonAvailability,
    PredictionComparison,
)
from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution import ExecutionObservation
from jaull.domain.experiments import ExperimentRecord, ExperimentWorkload
from jaull.domain.hardware import ComputeBackend
from jaull.domain.recommendation import (
    AssessmentLevel,
    ExternalEvaluationEvidence,
    HardConstraintCode,
)
from jaull.domain.requirements import (
    CommercialUse,
    RecommendationPriority,
    UseCase,
)
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBinaryStatus,
    LlamaCppRuntimeCapability,
    PyTorchRuntimeCapability,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.experiments.storage import ExperimentStore
from jaull.presentation.plan_labels import plan_backend
from jaull.recommendation import scoring
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    assess_plan,
    generate_execution_plans,
    rank_execution_plans,
)
from jaull.workflow import ranking
from jaull.workflow.container import ServiceContainer
from jaull.workflow.requirements import build_requirements
from tests._workflow_fixtures import (
    GIB,
    candidate,
    gguf_analysis,
    hardware,
    memory_estimate,
    size_driven_estimator,
    transformers_analysis,
)


def test_same_model_can_produce_multiple_execution_plans() -> None:
    evaluated = _evaluated_gguf(priority=RecommendationPriority.QUALITY)
    plans = generate_execution_plans(
        evaluated,
        _requirements(RecommendationPriority.QUALITY),
        estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
    )

    assert len(plans) > 1
    assert {plan.artifact.quantization for plan in plans} >= {"Q4_K_M", "Q5_K_M"}
    assert all(plan.runtime_family is RuntimeName.LLAMA_CPP for plan in plans)


def test_recommendation_result_operates_on_plan_not_only_repo() -> None:
    evaluated = _evaluated_gguf()
    recs = ranking.recommend(
        [evaluated],
        _requirements(),
        hardware=hardware(),
    )

    assert recs[0].plan is not None
    assert recs[0].plan_assessment is not None
    assert recs[0].plan.artifact.quantization == evaluated.selected_configuration.quantization


def test_cuda_backend_selection_flows_to_recommendation_plan() -> None:
    selection = _selection(ComputeBackend.CUDA)
    readiness = _readiness(
        RuntimeName.LLAMA_CPP,
        selection,
        ExecutionReadinessStatus.READY,
    )

    recs = ranking.recommend(
        [_evaluated_gguf()],
        _requirements(),
        hardware=hardware(),
        plan_context=PlanRankingContext(
            hardware=hardware(),
            backend_selection=selection,
            readiness_by_runtime={RuntimeName.LLAMA_CPP: readiness},
        ),
    )

    assert recs[0].plan is not None
    assert recs[0].plan.backend_selection is selection
    assert recs[0].plan.execution_readiness is readiness
    assert plan_backend(recs[0].plan) == "CUDA"


def test_hard_memory_failure_rejects_plan() -> None:
    evaluated = _evaluated_gguf(status=CompatibilityStatus.INSUFFICIENT)
    plan = generate_execution_plans(evaluated, _requirements())[0]

    assessment = assess_plan(evaluated, plan, _requirements())

    assert HardConstraintCode.MEMORY_INSUFFICIENT in {
        constraint.code for constraint in assessment.hard_constraints
    }


def test_missing_runtime_rejects_or_marks_not_ready() -> None:
    evaluated = _evaluated_gguf()
    selection = _selection()
    readiness = _readiness(RuntimeName.LLAMA_CPP, selection, ExecutionReadinessStatus.NOT_READY)
    plans = generate_execution_plans(
        evaluated,
        _requirements(),
        context=PlanRankingContext(
            hardware=hardware(),
            backend_selection=selection,
            readiness_by_runtime={RuntimeName.LLAMA_CPP: readiness},
        ),
    )

    assessment = assess_plan(
        evaluated,
        plans[0],
        _requirements(),
        context=PlanRankingContext(readiness_by_runtime={RuntimeName.LLAMA_CPP: readiness}),
    )

    assert assessment.executability is AssessmentLevel.BLOCKED
    assert HardConstraintCode.RUNTIME_NOT_READY in {
        constraint.code for constraint in assessment.hard_constraints
    }


def test_license_hard_constraint() -> None:
    evaluated = _evaluated_gguf(license_value="cc-by-nc-4.0")
    plan = generate_execution_plans(evaluated, _requirements(commercial=CommercialUse.YES))[0]

    assessment = assess_plan(evaluated, plan, _requirements(commercial=CommercialUse.YES))

    assert HardConstraintCode.LICENSE_INCOMPATIBLE in {
        constraint.code for constraint in assessment.hard_constraints
    }


def test_language_hard_constraint_when_metadata_explicitly_conflicts() -> None:
    evaluated = _evaluated_gguf(languages=["fr"])
    req = _requirements(languages=["Spanish"])
    plan = generate_execution_plans(evaluated, req)[0]

    assessment = assess_plan(evaluated, plan, req)

    assert assessment.language_fit is AssessmentLevel.BLOCKED
    assert HardConstraintCode.LANGUAGE_INCOMPATIBLE in {
        constraint.code for constraint in assessment.hard_constraints
    }


def test_use_case_suitability_and_missing_metadata_confidence() -> None:
    strong = _evaluated_gguf(repo_id="org/Coder-7B-GGUF", tags=["code", "instruct"])
    thin = _evaluated_gguf(
        repo_id="org/Unknown-7B-GGUF",
        tags=[],
        languages=[],
        metadata_confidence=EstimationConfidence.LOW,
    )

    strong_assessment = assess_plan(
        strong,
        generate_execution_plans(strong, _requirements(use_case=UseCase.CODING))[0],
        _requirements(use_case=UseCase.CODING),
    )
    thin_assessment = assess_plan(
        thin,
        generate_execution_plans(thin, _requirements(use_case=UseCase.CODING))[0],
        _requirements(use_case=UseCase.CODING),
    )

    assert strong_assessment.suitability in {AssessmentLevel.STRONG, AssessmentLevel.ADEQUATE}
    assert thin_assessment.rejected is False
    assert thin_assessment.confidence in {
        EstimationConfidence.LOW,
        EstimationConfidence.MEDIUM,
    }


def test_local_validated_evidence_is_recognized() -> None:
    evaluated = _evaluated_gguf()
    req = _requirements()
    plan = generate_execution_plans(
        evaluated,
        req,
        context=PlanRankingContext(hardware=hardware()),
    )[0]
    record = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)

    assessment = assess_plan(
        evaluated,
        plan,
        req,
        context=PlanRankingContext(hardware=hardware(), experiment_records=[record]),
    )

    assert assessment.local_experiment_id == record.identity.experiment_id
    assert "local:validated" in assessment.reasons


def test_local_benchmark_requires_compatible_fingerprint_and_machine() -> None:
    evaluated = _evaluated_gguf()
    req = _requirements()
    plan = generate_execution_plans(
        evaluated,
        req,
        context=PlanRankingContext(hardware=hardware()),
    )[0]
    matching = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=hardware(),
        tps=55.0,
    )
    other_machine = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=hardware(name="Other GPU"),
        tps=99.0,
    )

    assessment = assess_plan(
        evaluated,
        plan,
        req,
        context=PlanRankingContext(
            hardware=hardware(),
            benchmark_records=[other_machine, matching],
        ),
    )

    assert assessment.local_benchmark_id == matching.identity.benchmark_id
    assert assessment.measured_generation_tokens_per_second == 55.0


def test_local_evidence_matches_when_only_available_memory_changes() -> None:
    scanned = hardware()
    recorded = scanned.model_copy(
        update={
            "memory": scanned.memory.model_copy(
                update={"available_bytes": scanned.memory.available_bytes // 2}
            )
        }
    )
    evaluated = _evaluated_gguf()
    req = _requirements()
    plan = generate_execution_plans(
        evaluated,
        req,
        context=PlanRankingContext(hardware=scanned),
    )[0]
    benchmark = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=recorded,
        tps=55.0,
    )
    experiment = _experiment_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
    ).model_copy(update={"hardware": recorded})

    assessment = assess_plan(
        evaluated,
        plan,
        req,
        context=PlanRankingContext(
            hardware=scanned,
            benchmark_records=[benchmark],
            experiment_records=[experiment],
        ),
    )

    assert assessment.local_benchmark_id == benchmark.identity.benchmark_id
    assert assessment.local_experiment_id == experiment.identity.experiment_id


def test_advisor_plan_context_loads_local_evidence(tmp_path: Path) -> None:
    evaluated = _evaluated_gguf()
    req = _requirements()
    plan = generate_execution_plans(
        evaluated,
        req,
        context=PlanRankingContext(hardware=hardware()),
    )[0]
    experiment = _experiment_record(plan.artifact.to_model_artifact(), plan.runtime)
    benchmark = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=hardware(),
        tps=63.0,
    )
    experiment_store = ExperimentStore(root=tmp_path / "experiments")
    benchmark_store = BenchmarkStore(root=tmp_path / "benchmarks")
    experiment_store.save(experiment)
    benchmark_store.save(benchmark)
    advisor = AdvisorService(
        services=_services(),
        experiment_store=experiment_store,
        benchmark_store=benchmark_store,
    )

    context = advisor._plan_ranking_context(hardware())
    assessment = assess_plan(evaluated, plan, req, context=context)

    assert assessment.local_experiment_id == experiment.identity.experiment_id
    assert assessment.local_benchmark_id == benchmark.identity.benchmark_id
    assert assessment.measured_generation_tokens_per_second == 63.0


def test_advisor_plan_context_includes_backend_selection_and_readiness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selection = _selection(ComputeBackend.CUDA)
    llama_ready = _readiness(
        RuntimeName.LLAMA_CPP,
        selection,
        ExecutionReadinessStatus.READY,
    )
    torch_ready = _readiness(
        RuntimeName.TRANSFORMERS,
        selection,
        ExecutionReadinessStatus.READY,
    )

    def select_backend(self, observed_hardware):
        return selection

    def llama_readiness(self, **kwargs):
        assert kwargs["selection"] is selection
        return llama_ready

    def pytorch_readiness(self, **kwargs):
        assert kwargs["selection"] is selection
        return torch_ready

    monkeypatch.setattr(AdvisorService, "select_runtime_backend", select_backend)
    monkeypatch.setattr(AdvisorService, "evaluate_execution_readiness", llama_readiness)
    monkeypatch.setattr(
        AdvisorService,
        "evaluate_pytorch_execution_readiness",
        pytorch_readiness,
    )
    advisor = AdvisorService(
        services=_services(),
        experiment_store=ExperimentStore(root=tmp_path / "experiments"),
        benchmark_store=BenchmarkStore(root=tmp_path / "benchmarks"),
    )

    context = advisor._plan_ranking_context(hardware())

    assert context.backend_selection is selection
    assert context.readiness_by_runtime == {
        RuntimeName.LLAMA_CPP: llama_ready,
        RuntimeName.TRANSFORMERS: torch_ready,
    }


def test_measured_memory_is_preferred_over_estimate_when_compatible() -> None:
    evaluated = _evaluated_gguf()
    req = _requirements()
    plan = generate_execution_plans(
        evaluated,
        req,
        context=PlanRankingContext(hardware=hardware()),
    )[0]
    record = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=hardware(),
        tps=40.0,
        peak_ram_bytes=2600,
    )

    assessment = assess_plan(
        evaluated,
        plan,
        req,
        context=PlanRankingContext(hardware=hardware(), benchmark_records=[record]),
    )

    assert assessment.estimated_memory_bytes != assessment.measured_memory_bytes
    assert assessment.measured_memory_bytes == 2600
    assert "memory:measured" in assessment.reasons


def test_q4_and_q5_are_different_plans_and_priority_changes_policy() -> None:
    evaluated = _evaluated_gguf()
    quality = rank_execution_plans(
        [evaluated],
        _requirements(RecommendationPriority.QUALITY),
        estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
    )
    speed = rank_execution_plans(
        [evaluated],
        _requirements(RecommendationPriority.SPEED),
        estimate_fn=size_driven_estimator(vram_budget=24 * GIB),
    )

    assert {item.plan.artifact.quantization for item in quality} >= {"Q4_K_M", "Q5_K_M"}
    assert quality[0].plan.artifact.quantization != speed[0].plan.artifact.quantization
    assert quality[0].plan.artifact.quantization == "Q6_K"
    assert speed[0].plan.artifact.quantization == "Q4_K_M"


def test_balanced_does_not_reward_extra_headroom_over_capability() -> None:
    lightweight = _evaluated_gguf(
        repo_id="org/Smol-135M-Instruct-GGUF",
        total_bytes=1 * GIB,
    )
    balanced = _evaluated_gguf(
        repo_id="org/Useful-1.5B-Instruct-GGUF",
        total_bytes=3 * GIB,
    )
    larger = _evaluated_gguf(
        repo_id="org/Useful-3B-Instruct-GGUF",
        total_bytes=5 * GIB,
    )

    ranked = rank_execution_plans(
        [lightweight, balanced, larger],
        _requirements(RecommendationPriority.BALANCED, use_case=UseCase.GENERAL_CHAT),
    )

    assert ranked[0].evaluated.repo_id != lightweight.repo_id
    assert ranked[0].assessment.capability in {
        AssessmentLevel.STRONG,
        AssessmentLevel.ADEQUATE,
    }
    lightweight_assessment = next(
        item.assessment for item in ranked if item.evaluated.repo_id == lightweight.repo_id
    )
    assert lightweight_assessment.feasibility is AssessmentLevel.STRONG
    assert "capability:lightweight_model" in lightweight_assessment.trade_offs


def test_speed_can_still_promote_lightweight_models_without_benchmark_evidence() -> None:
    lightweight = _evaluated_gguf(
        repo_id="org/Smol-135M-Instruct-GGUF",
        total_bytes=1 * GIB,
    )
    balanced = _evaluated_gguf(
        repo_id="org/Useful-1.5B-Instruct-GGUF",
        total_bytes=3 * GIB,
    )

    ranked = rank_execution_plans(
        [balanced, lightweight],
        _requirements(RecommendationPriority.SPEED, use_case=UseCase.GENERAL_CHAT),
    )

    assert ranked[0].evaluated.repo_id == lightweight.repo_id


def test_instruction_tuned_evidence_improves_assistant_suitability() -> None:
    instruct = _evaluated_gguf(
        repo_id="org/Assistant-1.5B-Instruct-GGUF",
        tags=["text-generation", "instruct", "gguf"],
    )
    base = _evaluated_gguf(
        repo_id="org/Plain-1.5B-Base-GGUF",
        tags=["text-generation", "base", "gguf"],
    )
    req = _requirements(RecommendationPriority.BALANCED, use_case=UseCase.GENERAL_CHAT)

    instruct_assessment = assess_plan(
        instruct,
        generate_execution_plans(instruct, req)[0],
        req,
    )
    base_assessment = assess_plan(
        base,
        generate_execution_plans(base, req)[0],
        req,
    )

    assert instruct_assessment.suitability is AssessmentLevel.ADEQUATE
    assert base_assessment.suitability is AssessmentLevel.WEAK
    assert "capability:instruction_tuned" in instruct_assessment.reasons
    assert "capability:base_model" in base_assessment.trade_offs


def test_general_chat_suitability_accepts_conversational_evidence() -> None:
    req = _requirements(RecommendationPriority.BALANCED, use_case=UseCase.GENERAL_CHAT)
    chat = _assessment_for_tags("org/Model-3B", ["text-generation", "chat"], req)
    assistant = _assessment_for_tags(
        "org/Model-3B",
        ["text-generation", "assistant"],
        req,
    )
    conversational = _assessment_for_tags(
        "org/Model-3B",
        ["text-generation", "conversational"],
        req,
    )
    instruct_only = _assessment_for_tags(
        "org/Model-3B-Instruct",
        ["text-generation", "instruct"],
        req,
    )

    assert chat.suitability is AssessmentLevel.STRONG
    assert assistant.suitability is AssessmentLevel.STRONG
    assert conversational.suitability is AssessmentLevel.STRONG
    assert instruct_only.suitability is AssessmentLevel.ADEQUATE


def test_transformers_and_llama_cpp_plans_can_coexist() -> None:
    plans = rank_execution_plans(
        [_evaluated_gguf(), _evaluated_transformers()],
        _requirements(),
    )

    assert {item.plan.runtime_family for item in plans} >= {
        RuntimeName.LLAMA_CPP,
        RuntimeName.TRANSFORMERS,
    }


def test_runtime_readiness_affects_ranking() -> None:
    selection = _selection()
    ready = _readiness(RuntimeName.LLAMA_CPP, selection, ExecutionReadinessStatus.READY)
    not_ready = _readiness(
        RuntimeName.TRANSFORMERS,
        selection,
        ExecutionReadinessStatus.NOT_READY,
    )
    ranked = rank_execution_plans(
        [
            _evaluated_transformers(repo_id="org/Strong-7B"),
            _evaluated_gguf(repo_id="org/Ready-7B-GGUF"),
        ],
        _requirements(),
        context=PlanRankingContext(
            backend_selection=selection,
            readiness_by_runtime={
                RuntimeName.LLAMA_CPP: ready,
                RuntimeName.TRANSFORMERS: not_ready,
            },
        ),
    )

    assert ranked[0].plan.runtime_family is RuntimeName.LLAMA_CPP
    assert ranked[0].assessment.executability is AssessmentLevel.STRONG


def test_balanced_prefers_runnable_plan_over_more_capable_offloading_plan() -> None:
    ranked = rank_execution_plans(
        [
            _evaluated_gguf(
                repo_id="org/Qwen2.5-7B-Instruct-AWQ",
                status=CompatibilityStatus.OFFLOADING_REQUIRED,
            ),
            _evaluated_gguf(
                repo_id="org/Qwen2.5-1.5B-Instruct-GGUF",
                status=CompatibilityStatus.COMFORTABLE,
            ),
        ],
        _requirements(RecommendationPriority.BALANCED),
        context=_ready_llama_context(),
    )

    assert ranked[0].evaluated.repo_id == "org/Qwen2.5-1.5B-Instruct-GGUF"
    assert ranked[0].assessment.capability is AssessmentLevel.ADEQUATE
    assert ranked[0].assessment.execution_fitness is AssessmentLevel.STRONG
    assert ranked[1].assessment.capability is AssessmentLevel.STRONG
    assert ranked[1].assessment.execution_fitness is AssessmentLevel.WEAK


def test_balanced_can_still_prefer_more_capable_compatible_plan() -> None:
    ranked = rank_execution_plans(
        [
            _evaluated_gguf(
                repo_id="org/Qwen3-4B-Instruct-GGUF",
                status=CompatibilityStatus.COMPATIBLE,
            ),
            _evaluated_gguf(
                repo_id="org/Qwen2.5-1.5B-Instruct-GGUF",
                status=CompatibilityStatus.COMFORTABLE,
            ),
        ],
        _requirements(RecommendationPriority.BALANCED),
        context=_ready_llama_context(),
    )

    assert ranked[0].evaluated.repo_id == "org/Qwen3-4B-Instruct-GGUF"
    assert ranked[0].assessment.capability is AssessmentLevel.STRONG
    assert ranked[0].assessment.execution_fitness is AssessmentLevel.ADEQUATE
    assert ranked[1].assessment.capability is AssessmentLevel.ADEQUATE
    assert ranked[1].assessment.execution_fitness is AssessmentLevel.STRONG


def test_quality_policy_is_unchanged_by_balanced_runnability_gate() -> None:
    ranked = rank_execution_plans(
        [
            _evaluated_gguf(
                repo_id="org/Qwen2.5-7B-Instruct-AWQ",
                status=CompatibilityStatus.OFFLOADING_REQUIRED,
            ),
            _evaluated_gguf(
                repo_id="org/Qwen2.5-1.5B-Instruct-GGUF",
                status=CompatibilityStatus.COMFORTABLE,
            ),
        ],
        _requirements(RecommendationPriority.QUALITY),
        context=_ready_llama_context(),
    )

    assert ranked[0].evaluated.repo_id == "org/Qwen2.5-7B-Instruct-AWQ"
    assert ranked[0].assessment.capability is AssessmentLevel.STRONG
    assert ranked[0].assessment.execution_fitness is AssessmentLevel.WEAK
    assert ranked[1].assessment.execution_fitness is AssessmentLevel.STRONG


def test_non_executable_plan_does_not_appear_in_top3_when_ready_plans_exist() -> None:
    selection = _selection()
    ready = _readiness(RuntimeName.LLAMA_CPP, selection, ExecutionReadinessStatus.READY)
    not_ready = _readiness(
        RuntimeName.TRANSFORMERS,
        selection,
        ExecutionReadinessStatus.NOT_READY,
    )
    evaluated = [
        _evaluated_transformers(repo_id="org/Blocked-7B"),
        _evaluated_gguf(repo_id="org/Ready-A-7B-GGUF"),
        _evaluated_gguf(repo_id="org/Ready-B-7B-GGUF"),
        _evaluated_gguf(repo_id="org/Ready-C-7B-GGUF"),
    ]

    ranked = rank_execution_plans(
        evaluated,
        _requirements(),
        context=PlanRankingContext(
            backend_selection=selection,
            readiness_by_runtime={
                RuntimeName.LLAMA_CPP: ready,
                RuntimeName.TRANSFORMERS: not_ready,
            },
        ),
        limit=3,
    )

    assert all(item.plan.runtime_family is RuntimeName.LLAMA_CPP for item in ranked)


def test_external_evaluation_provenance_is_retained_without_global_quality_score() -> None:
    evidence = ExternalEvaluationEvidence(
        benchmark="HumanEval",
        task="coding",
        value=72.0,
        metric="pass@1",
        source="hf_leaderboard",
        confidence=EstimationConfidence.MEDIUM,
        revision="abc123",
    )
    evaluated = _evaluated_gguf()
    plan = generate_execution_plans(evaluated, _requirements())[0]

    assessment = assess_plan(
        evaluated,
        plan,
        _requirements(),
        context=PlanRankingContext(external_evaluations=[evidence]),
    )

    assert assessment.external_evaluations == [evidence]
    assert "score" not in assessment.model_dump()


def test_incompatible_benchmark_methodologies_are_not_merged() -> None:
    evaluated = _evaluated_gguf()
    req = _requirements()
    plan = generate_execution_plans(
        evaluated,
        req,
        context=PlanRankingContext(hardware=hardware()),
    )[0]
    old = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=hardware(),
        tps=10.0,
        methodology="old_method",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    new = _benchmark_record(
        plan.artifact.to_model_artifact(),
        plan.runtime,
        machine=hardware(),
        tps=20.0,
        methodology="new_method",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assessment = assess_plan(
        evaluated,
        plan,
        req,
        context=PlanRankingContext(hardware=hardware(), benchmark_records=[old, new]),
    )

    assert assessment.local_benchmark_id == new.identity.benchmark_id
    assert assessment.measured_generation_tokens_per_second == 20.0


def test_top3_limit_and_deterministic_reason_codes() -> None:
    ranked = rank_execution_plans(
        [_evaluated_gguf(repo_id=f"org/Model-{index}-7B-GGUF") for index in range(5)],
        _requirements(),
        limit=3,
    )

    assert len(ranked) == 3
    assert ranked[0].assessment.reasons == ranked[0].assessment.reasons
    assert ranked[0].assessment.trade_offs == ranked[0].assessment.trade_offs


def test_engine_does_not_download_or_benchmark_models() -> None:
    evaluated = _evaluated_gguf()
    calls = 0

    def estimate_only(analysis, config):
        nonlocal calls
        calls += 1
        return size_driven_estimator(vram_budget=24 * GIB)(analysis, config)

    rank_execution_plans([evaluated], _requirements(), estimate_fn=estimate_only)

    assert calls == len(evaluated.analysis.classification.gguf_variants)


def _requirements(
    priority: RecommendationPriority = RecommendationPriority.BALANCED,
    *,
    use_case: UseCase = UseCase.CODING,
    languages: list[str] | None = None,
    commercial: CommercialUse = CommercialUse.YES,
):
    return build_requirements(
        _answers(priority, use_case, languages, commercial),
        hardware(),
    )


def _answers(
    priority: RecommendationPriority,
    use_case: UseCase,
    languages: list[str] | None,
    commercial: CommercialUse,
):
    from tests._workflow_fixtures import answers

    return answers(
        use_case=use_case,
        priority=priority,
        languages=languages if languages is not None else ["English"],
        commercial=commercial,
    )


def _evaluated_gguf(
    repo_id: str = "org/Coder-7B-GGUF",
    *,
    priority: RecommendationPriority = RecommendationPriority.BALANCED,
    status: CompatibilityStatus = CompatibilityStatus.COMFORTABLE,
    license_value: str | None = "apache-2.0",
    languages: list[str] | None = None,
    tags: list[str] | None = None,
    metadata_confidence: EstimationConfidence = EstimationConfidence.HIGH,
    total_bytes: int = 5 * GIB,
) -> EvaluatedCandidate:
    req = _requirements(priority)
    analysis = gguf_analysis(repo_id=repo_id)
    estimator = size_driven_estimator(vram_budget=24 * GIB)
    choice = _select(analysis, req, estimator)
    estimate = memory_estimate(
        analysis,
        choice,
        total_bytes=total_bytes,
        status=status,
        confidence=EstimationConfidence.HIGH,
    ).model_copy(
        update={
            "runtime_recommendation": RuntimeRecommendation(
                runtime=RuntimeName.LLAMA_CPP,
                confidence=EstimationConfidence.HIGH,
            )
        }
    )
    candidate_ = candidate(
        repo_id=repo_id,
        license_value=license_value,
        languages=languages if languages is not None else ["en"],
        tags=tags if tags is not None else ["text-generation", "code", "instruct", "gguf"],
    ).model_copy(update={"metadata_confidence": metadata_confidence})
    evaluated = EvaluatedCandidate(
        candidate=candidate_,
        analysis=analysis,
        selected_configuration=choice,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )
    return scoring.score_candidate(evaluated, req, 1.0, capability_signal=0.8)


def _evaluated_transformers(
    repo_id: str = "org/Coder-7B",
    *,
    status: CompatibilityStatus = CompatibilityStatus.COMFORTABLE,
) -> EvaluatedCandidate:
    req = _requirements()
    analysis = transformers_analysis(repo_id=repo_id)
    config = _select(analysis, req, size_driven_estimator(vram_budget=24 * GIB))
    estimate = memory_estimate(
        analysis,
        config,
        total_bytes=8 * GIB,
        status=status,
        confidence=EstimationConfidence.HIGH,
    ).model_copy(
        update={
            "runtime_recommendation": RuntimeRecommendation(
                runtime=RuntimeName.TRANSFORMERS,
                confidence=EstimationConfidence.HIGH,
            )
        }
    )
    evaluated = EvaluatedCandidate(
        candidate=candidate(repo_id=repo_id, tags=["text-generation", "code", "instruct"]),
        analysis=analysis,
        selected_configuration=config,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )
    return scoring.score_candidate(evaluated, req, 1.0, capability_signal=0.8)


def _assessment_for_tags(repo_id: str, tags: list[str], req):
    evaluated = _evaluated_gguf(repo_id=repo_id, tags=tags).model_copy(
        update={"task_match_score": 0.75}
    )
    return assess_plan(evaluated, generate_execution_plans(evaluated, req)[0], req)


def _select(analysis, req, estimator):
    from jaull.estimator.configuration import select_configuration

    choice = select_configuration(analysis, req, estimator)
    assert choice.configuration is not None
    return choice.configuration


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=hardware,
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )


def _selection(backend: ComputeBackend = ComputeBackend.CPU) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=(
            RuntimeBackendSelectionReason.CPU_FALLBACK
            if backend is ComputeBackend.CPU
            else RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE
        ),
    )


def _readiness(
    runtime: RuntimeName,
    selection: RuntimeBackendSelection,
    status: ExecutionReadinessStatus,
) -> ExecutionReadiness:
    capability = (
        LlamaCppRuntimeCapability(
            binary_path="/tmp/llama-cli",
            binary_status=LlamaCppBinaryStatus.AVAILABLE,
        )
        if runtime is RuntimeName.LLAMA_CPP
        else PyTorchRuntimeCapability(
            python_executable="/tmp/python",
            runtime_status=PyTorchRuntimeStatus.AVAILABLE,
        )
    )
    return ExecutionReadiness(
        status=status,
        reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
        selection=selection,
        runtime_capability=capability,
        message=f"{runtime.value} is {status.value}",
    )


def _ready_llama_context() -> PlanRankingContext:
    selection = _selection(ComputeBackend.CPU)
    return PlanRankingContext(
        backend_selection=selection,
        readiness_by_runtime={
            RuntimeName.LLAMA_CPP: _readiness(
                RuntimeName.LLAMA_CPP,
                selection,
                ExecutionReadinessStatus.READY,
            )
        },
    )


def _benchmark_record(
    artifact: ModelArtifact,
    runtime: RuntimeRecommendation,
    *,
    machine,
    tps: float,
    peak_ram_bytes: int | None = None,
    methodology: str = "llama_bench_v1",
    created_at: datetime | None = None,
) -> BenchmarkRecord:
    request = BenchmarkRequest(
        artifact=artifact,
        runtime=runtime,
        backend=ComputeBackend.CPU,
        device="none",
        gpu_layers=BenchmarkGpuLayers.count_layers(0),
    )
    observation = BenchmarkObservation(
        success=True,
        measurements=[
            BenchmarkMeasurement(
                kind=BenchmarkMeasurementKind.GENERATION,
                tokens=64,
                mean_tokens_per_second=tps,
                stddev_tokens_per_second=1.0,
                source_label="tg64",
            )
        ],
        repetitions=3,
        duration_seconds=1.0,
        methodology=methodology,
        peak_ram_bytes=peak_ram_bytes,
    )
    identity = None
    if created_at is not None:
        identity = BenchmarkIdentity.create().model_copy(update={"created_at": created_at})
    return BenchmarkRecord.create(
        hardware=machine,
        request=request,
        observation=observation,
        identity=identity,
    )


def _experiment_record(
    artifact: ModelArtifact,
    runtime: RuntimeRecommendation,
) -> ExperimentRecord:
    selection = _selection()
    readiness = _readiness(runtime.runtime, selection, ExecutionReadinessStatus.READY)
    prediction = memory_estimate(
        gguf_analysis(repo_id=artifact.repo_id),
        _evaluated_gguf().selected_configuration,
        total_bytes=5 * GIB,
    ).model_copy(update={"runtime_recommendation": runtime})
    observation = ExecutionObservation(
        success=True,
        duration_seconds=1.0,
        peak_ram_bytes=2500,
        peak_vram_bytes=None,
        exit_code=0,
    )
    comparison = PredictionComparison(
        ram=MetricComparison(
            predicted_bytes=prediction.total_bytes,
            measured_bytes=2500,
            error_bytes=2500 - (prediction.total_bytes or 0),
            absolute_error_bytes=abs(2500 - (prediction.total_bytes or 0)),
            error_percent=0.0,
            availability=MetricComparisonAvailability.AVAILABLE,
        ),
        vram=MetricComparison(
            availability=MetricComparisonAvailability.MEASUREMENT_UNAVAILABLE,
            unavailable_reason="not measured",
        ),
        compatibility=CompatibilityComparison(
            predicted_status=prediction.assessment.status,
            predicted_runnable=True,
            observed_success=True,
            outcome=CompatibilityOutcome.CORRECT_SUCCESS,
        ),
    )
    return ExperimentRecord.create(
        hardware=hardware(),
        artifact=artifact,
        workload=ExperimentWorkload(prompt="hello"),
        runtime=runtime,
        prediction=prediction,
        runtime_capability=readiness.runtime_capability,
        execution_readiness=readiness,
        observation=observation,
        comparison=comparison,
    )
