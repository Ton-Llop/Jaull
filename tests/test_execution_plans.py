from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jaull.advisor.service import AdvisorService
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
from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate, SearchQuery
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.domain.execution_plans import (
    ArtifactVariantFormat,
    IdentityMatchStatus,
    PlanCompatibilityStatus,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import InferenceConfiguration, WeightPrecision
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.benchmark_comparison import compare_benchmark_records
from jaull.execution_plans import (
    build_execution_plan,
    discover_artifact_variants,
    execution_plan_for_recommendation,
    resolve_model_identity,
    variant_from_recommendation,
)
from jaull.recommendation.models import ModelRecommendation, ScoreBreakdown
from jaull.recommendation.policies import LicenseCategory
from jaull.workflow import policies
from jaull.workflow.container import ServiceContainer
from jaull.workflow.model_analysis_cache import ModelAnalysisCache
from tests._workflow_fixtures import (
    GIB,
    candidate,
    gguf_analysis,
    hardware,
    memory_estimate,
    transformers_analysis,
)


def test_transformers_repo_resolves_model_identity() -> None:
    analysis = transformers_analysis("Qwen/Qwen2.5-0.5B-Instruct")
    identity = resolve_model_identity(
        candidate=candidate("Qwen/Qwen2.5-0.5B-Instruct"),
        analysis=analysis,
    )

    assert identity.model_name == "Qwen2.5-0.5B-Instruct"
    assert identity.family == "llama"
    assert identity.confidence is EstimationConfidence.MEDIUM
    assert identity.evidence


def test_gguf_repo_matches_base_model_metadata() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    derived = ModelCandidate(
        repo_id="someone/Qwen2.5-0.5B-Instruct-GGUF",
        base_model_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    )
    search = _Search([derived])
    inspector = _Inspector(
        {"someone/Qwen2.5-0.5B-Instruct-GGUF": gguf_analysis(derived.repo_id)}
    )

    variants = discover_artifact_variants(
        identity=identity,
        current=variant_from_recommendation(source, identity=identity),
        search_client=search,
        inspect_model=inspector,
    )

    assert search.queries[0].search == "Qwen2.5-0.5B-Instruct GGUF"
    assert inspector.calls == ["someone/Qwen2.5-0.5B-Instruct-GGUF"]
    assert any(
        variant.format is ArtifactVariantFormat.GGUF
        and variant.quantization == "Q4_K_M"
        and variant.identity_match is IdentityMatchStatus.CONFIRMED
        for variant in variants
    )


def test_similar_name_without_evidence_is_not_safe_variant() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    search = _Search([candidate("someone/Qwen2.5-0.5B-Instruct-GGUF")])
    inspector = _Inspector(
        {"someone/Qwen2.5-0.5B-Instruct-GGUF": gguf_analysis("someone/Other-GGUF")}
    )

    variants = discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=search,
        inspect_model=inspector,
    )

    assert variants == []


def test_variant_discovery_does_not_deep_inspect_every_search_result() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    strong = candidate(
        "someone/Qwen2.5-0.5B-Instruct-GGUF",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        tags=["gguf"],
    )
    results = [
        strong,
        *[
            candidate(f"mirror/Other-{index}-GGUF", tags=["gguf"])
            for index in range(19)
        ],
    ]
    inspector = _Inspector({strong.repo_id: gguf_analysis(strong.repo_id)})

    variants = discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=_Search(results),
        inspect_model=inspector,
    )

    assert inspector.calls == [strong.repo_id]
    assert len(inspector.calls) < len(results)
    assert any(variant.quantization == "Q4_K_M" for variant in variants)


def test_structured_base_model_match_gets_priority_over_heuristic() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    heuristic = candidate(
        "mirror/Qwen2.5-0.5B-Instruct-GGUF",
        tags=["gguf"],
    )
    strong = candidate(
        "author/Qwen2.5-0.5B-Instruct-GGUF",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        tags=["gguf"],
    )
    inspector = _Inspector(
        {
            strong.repo_id: gguf_analysis(strong.repo_id),
            heuristic.repo_id: gguf_analysis(heuristic.repo_id),
        }
    )

    discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=_Search([heuristic, strong]),
        inspect_model=inspector,
        include_uncertain=True,
        max_deep_inspection=2,
    )

    assert inspector.calls[0] == strong.repo_id


def test_confirmed_candidate_is_not_lost_after_heuristic_result() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    heuristic = candidate(
        "mirror/Qwen2.5-0.5B-Instruct-GGUF",
        tags=["gguf"],
    )
    strong = candidate(
        "author/Qwen2.5-0.5B-Instruct-GGUF",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        tags=["gguf"],
    )
    inspector = _Inspector(
        {
            strong.repo_id: gguf_analysis(strong.repo_id),
            heuristic.repo_id: gguf_analysis("other/model"),
        }
    )

    variants = discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=_Search([heuristic, strong]),
        inspect_model=inspector,
        max_deep_inspection=1,
    )

    assert inspector.calls == [strong.repo_id]
    assert any(
        variant.identity_match is IdentityMatchStatus.CONFIRMED
        for variant in variants
    )


def test_include_uncertain_allows_name_only_candidate_without_confirming_it() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    heuristic = candidate(
        "mirror/Qwen2.5-0.5B-Instruct-GGUF",
        tags=["gguf"],
    )
    inspector = _Inspector({heuristic.repo_id: gguf_analysis(heuristic.repo_id)})

    variants = discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=_Search([heuristic]),
        inspect_model=inspector,
        include_uncertain=True,
    )

    assert inspector.calls == [heuristic.repo_id]
    assert all(
        variant.identity_match is IdentityMatchStatus.UNCERTAIN
        for variant in variants
    )


def test_variant_discovery_reuses_persistent_analysis_cache(tmp_path: Path) -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    gguf_candidate = candidate(
        "someone/Qwen2.5-0.5B-Instruct-GGUF",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        tags=["gguf"],
    ).model_copy(update={"revision_hint": "rev1"})
    cache = ModelAnalysisCache(root=tmp_path)
    cache.put(gguf_candidate, gguf_analysis(gguf_candidate.repo_id))
    inspector = _Inspector({})

    variants = discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=_Search([gguf_candidate]),
        inspect_model=inspector,
        analysis_cache=cache,
    )

    assert inspector.calls == []
    assert cache.stats.hits == 1
    assert variants


def test_variant_discovery_returns_current_artifact_without_search_results() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    current = variant_from_recommendation(source, identity=identity)

    variants = discover_artifact_variants(
        identity=identity,
        current=current,
        search_client=_Search([]),
        inspect_model=_Inspector({}),
    )

    assert variants == [current]


def test_variant_deep_inspection_budget_is_explicit() -> None:
    source = _transformers_recommendation()
    identity = resolve_model_identity(
        candidate=source.evaluated.candidate,
        analysis=source.evaluated.analysis,
    )
    results = [
        candidate(
            f"mirror/Qwen2.5-0.5B-Instruct-{index}-GGUF",
            tags=["gguf"],
        )
        for index in range(20)
    ]
    inspector = _Inspector(
        {item.repo_id: gguf_analysis(item.repo_id) for item in results}
    )

    discover_artifact_variants(
        identity=identity,
        current=None,
        search_client=_Search(results),
        inspect_model=inspector,
        include_uncertain=True,
    )

    assert len(inspector.calls) == policies.MAX_VARIANT_DEEP_INSPECTION


def test_recommendation_variant_keeps_safetensors_runtime() -> None:
    rec = _transformers_recommendation()
    variant = variant_from_recommendation(rec)

    assert variant.format is ArtifactVariantFormat.SAFETENSORS
    assert variant.compatible_runtimes == [RuntimeName.TRANSFORMERS]
    assert variant.precision == "float32"


def test_gguf_recommendation_variant_lists_all_quantization_metadata() -> None:
    rec = _gguf_recommendation()
    variant = variant_from_recommendation(rec)

    assert variant.format is ArtifactVariantFormat.GGUF
    assert variant.filename == "model-Q4_K_M.gguf"
    assert variant.quantization == "Q4_K_M"
    assert variant.compatible_runtimes == [RuntimeName.LLAMA_CPP]


def test_execution_plan_transformers_cpu_ready() -> None:
    rec = _transformers_recommendation()
    selection = _selection(ComputeBackend.CPU)
    capability = _llama_capability()
    readiness = _readiness(selection, capability, ExecutionReadinessStatus.READY)

    plan = execution_plan_for_recommendation(
        rec,
        hardware=hardware(),
        backend_selection=selection,
        runtime_capability=capability,
        execution_readiness=readiness,
    )

    assert plan.runtime_family is RuntimeName.TRANSFORMERS
    assert plan.compatibility is PlanCompatibilityStatus.COMPATIBLE
    assert plan.execution_readiness is readiness


def test_execution_plan_llama_cpp_not_ready_is_preserved() -> None:
    rec = _gguf_recommendation()
    selection = _selection(ComputeBackend.CPU)
    capability = _llama_capability()
    readiness = _readiness(selection, capability, ExecutionReadinessStatus.NOT_READY)

    plan = execution_plan_for_recommendation(
        rec,
        backend_selection=selection,
        runtime_capability=capability,
        execution_readiness=readiness,
    )

    assert plan.compatibility is PlanCompatibilityStatus.NOT_COMPATIBLE
    assert "Runtime readiness is not_ready." in plan.warnings


def test_build_execution_plan_rejects_runtime_artifact_mismatch() -> None:
    rec = _gguf_recommendation()
    identity = resolve_model_identity(
        candidate=rec.evaluated.candidate,
        analysis=rec.evaluated.analysis,
    )
    variant = variant_from_recommendation(rec, identity=identity)
    runtime = RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        confidence=EstimationConfidence.HIGH,
    )

    plan = build_execution_plan(
        model_identity=identity,
        artifact=variant,
        runtime=runtime,
    )

    assert plan.compatibility is PlanCompatibilityStatus.NOT_COMPATIBLE


def test_benchmark_comparison_warns_for_different_artifacts(tmp_path: Path) -> None:
    identity = resolve_model_identity(
        candidate=candidate("Qwen/Qwen2.5-0.5B-Instruct"),
        analysis=transformers_analysis("Qwen/Qwen2.5-0.5B-Instruct"),
    )
    llama = _benchmark_record(
        tmp_path,
        artifact=ModelArtifact(
            repo_id="someone/Qwen2.5-0.5B-Instruct-GGUF",
            revision="main",
            filename="qwen-q4.gguf",
            format="gguf",
            quantization="Q4_K_M",
        ),
        runtime=_runtime(RuntimeName.LLAMA_CPP),
        tps=40.0,
    )
    torch = _benchmark_record(
        tmp_path,
        artifact=ModelArtifact(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
            filename="Qwen/Qwen2.5-0.5B-Instruct",
            format="safetensors",
        ),
        runtime=_runtime(RuntimeName.TRANSFORMERS, torch_dtype="torch.float32"),
        tps=10.0,
    )

    comparison = compare_benchmark_records(
        model_identity=identity,
        records=[llama, torch],
    )

    assert comparison.overall_score is None
    assert any("Artifact formats differ" in warning for warning in comparison.warnings)
    assert any("methodolog" in warning for warning in comparison.warnings)
    assert comparison.metrics[0].relative_throughput == 0.25
    assert comparison.metrics[0].warning is not None


def test_benchmark_comparison_groups_runs_by_configuration(tmp_path: Path) -> None:
    identity = resolve_model_identity(
        candidate=candidate("Qwen/Qwen2.5-0.5B-Instruct"),
        analysis=transformers_analysis("Qwen/Qwen2.5-0.5B-Instruct"),
    )
    artifact = ModelArtifact(
        repo_id="someone/Qwen2.5-0.5B-Instruct-GGUF",
        revision="main",
        filename="qwen-q4.gguf",
        format="gguf",
        quantization="Q4_K_M",
    )
    older = _benchmark_record(
        tmp_path,
        artifact=artifact,
        runtime=_runtime(RuntimeName.LLAMA_CPP),
        tps=40.0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    latest = _benchmark_record(
        tmp_path,
        artifact=artifact,
        runtime=_runtime(RuntimeName.LLAMA_CPP),
        tps=65.0,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    comparison = compare_benchmark_records(
        model_identity=identity,
        records=[older, latest],
    )

    assert len(comparison.plans) == 1
    assert comparison.plans[0].benchmark_id == latest.identity.benchmark_id
    assert comparison.plans[0].compatible_run_count == 2


def test_benchmark_comparison_omits_older_transformers_methodology(tmp_path: Path) -> None:
    identity = resolve_model_identity(
        candidate=candidate("Qwen/Qwen2.5-0.5B-Instruct"),
        analysis=transformers_analysis("Qwen/Qwen2.5-0.5B-Instruct"),
    )
    artifact = ModelArtifact(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="main",
        filename="model.safetensors",
        format="safetensors",
    )
    legacy = _benchmark_record(
        tmp_path,
        artifact=artifact,
        runtime=_runtime(RuntimeName.TRANSFORMERS),
        tps=12.0,
        methodology="transformers_generate_steady_state_v1",
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
    )
    current = _benchmark_record(
        tmp_path,
        artifact=artifact,
        runtime=_runtime(RuntimeName.TRANSFORMERS),
        tps=14.0,
        methodology="transformers_isolated_inference_v2",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    comparison = compare_benchmark_records(
        model_identity=identity,
        records=[legacy, current],
    )

    assert len(comparison.plans) == 1
    assert comparison.plans[0].methodology == "transformers_isolated_inference_v2"
    assert comparison.plans[0].benchmark_id == current.identity.benchmark_id
    assert any("Older benchmark methodologies" in item for item in comparison.warnings)


def test_advisor_exposes_identity_variant_plan_and_comparison(tmp_path: Path) -> None:
    rec = _transformers_recommendation()
    advisor = AdvisorService(services=_services())
    identity = advisor.resolve_model_identity(rec)
    variant = advisor.artifact_variant_for_recommendation(rec)
    plan = advisor.execution_plan_for_recommendation(rec)
    comparison = advisor.compare_benchmarks(
        model_identity=identity,
        records=[
            _benchmark_record(
                tmp_path,
                artifact=variant.to_model_artifact(),
                runtime=_runtime(RuntimeName.TRANSFORMERS),
                tps=10.0,
            )
        ],
    )

    assert identity.model_name == "Qwen2.5-0.5B-Instruct"
    assert variant.format is ArtifactVariantFormat.SAFETENSORS
    assert plan.artifact == variant
    assert comparison.plans[0].runtime == "transformers"


class _Search:
    def __init__(self, results: list[ModelCandidate]) -> None:
        self.results = results
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery) -> list[ModelCandidate]:
        self.queries.append(query)
        return self.results


class _Inspector:
    def __init__(self, analyses: dict[str, object]) -> None:
        self.analyses = analyses
        self.calls: list[str] = []

    def __call__(self, repo_id: str) -> Any:
        self.calls.append(repo_id)
        return self.analyses[repo_id]


def _transformers_recommendation() -> ModelRecommendation:
    analysis = transformers_analysis("Qwen/Qwen2.5-0.5B-Instruct")
    config = InferenceConfiguration(
        context_length=4096,
        precision=WeightPrecision.FLOAT32,
    )
    estimate = memory_estimate(
        analysis,
        config,
        total_bytes=1 * GIB,
        status=CompatibilityStatus.COMFORTABLE,
    ).model_copy(update={"runtime_recommendation": _runtime(RuntimeName.TRANSFORMERS)})
    evaluated = EvaluatedCandidate(
        candidate=candidate("Qwen/Qwen2.5-0.5B-Instruct"),
        analysis=analysis,
        selected_configuration=config,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )
    return _recommendation(evaluated)


def _gguf_recommendation() -> ModelRecommendation:
    analysis = gguf_analysis("someone/Qwen2.5-0.5B-Instruct-GGUF")
    config = InferenceConfiguration(context_length=4096, quantization="Q4_K_M")
    estimate = memory_estimate(
        analysis,
        config,
        total_bytes=1 * GIB,
        status=CompatibilityStatus.COMFORTABLE,
    ).model_copy(update={"runtime_recommendation": _runtime(RuntimeName.LLAMA_CPP)})
    evaluated = EvaluatedCandidate(
        candidate=candidate(
            "someone/Qwen2.5-0.5B-Instruct-GGUF",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
        ).model_copy(update={"repository_type": RepositoryType.GGUF}),
        analysis=analysis,
        selected_configuration=config,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )
    return _recommendation(evaluated)


def _recommendation(evaluated: EvaluatedCandidate) -> ModelRecommendation:
    return ModelRecommendation(
        rank=1,
        evaluated=evaluated,
        score=ScoreBreakdown(total=0.9),
        status=CompatibilityStatus.COMFORTABLE,
        confidence=EstimationConfidence.HIGH,
        license_category=LicenseCategory.COMMERCIAL_ALLOWED,
        reasons=["fixture"],
    )


def _runtime(
    runtime: RuntimeName,
    *,
    torch_dtype: str | None = None,
) -> RuntimeRecommendation:
    flags = [
        RuntimeFlag(
            name="device_map" if runtime is RuntimeName.TRANSFORMERS else "--n-gpu-layers",
            value="cpu" if runtime is RuntimeName.TRANSFORMERS else "0",
            source=RuntimeFlagSource.HARDWARE,
            explanation="test",
        )
    ]
    if torch_dtype is not None:
        flags.append(
            RuntimeFlag(
                name="torch_dtype",
                value=torch_dtype,
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            )
        )
    return RuntimeRecommendation(
        runtime=runtime,
        flags=flags,
        confidence=EstimationConfidence.HIGH,
    )


def _selection(backend: ComputeBackend) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK,
    )


def _llama_capability() -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_path="/tmp/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CPU,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
            )
        ],
    )


def _readiness(
    selection: RuntimeBackendSelection,
    capability: LlamaCppRuntimeCapability,
    status: ExecutionReadinessStatus,
) -> ExecutionReadiness:
    return ExecutionReadiness(
        status=status,
        reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
        selection=selection,
        runtime_capability=capability,
        message=f"Runtime readiness is {status.value}.",
    )


def _benchmark_record(
    tmp_path: Path,
    *,
    artifact: ModelArtifact,
    runtime: RuntimeRecommendation,
    tps: float,
    methodology: str | None = None,
    created_at: datetime | None = None,
) -> BenchmarkRecord:
    del tmp_path
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
        methodology=methodology
        or (
            "llama_bench_v1"
            if runtime.runtime is RuntimeName.LLAMA_CPP
            else "transformers_isolated_inference_v2"
        ),
    )
    identity = None
    if created_at is not None:
        identity = BenchmarkIdentity.create().model_copy(update={"created_at": created_at})
    return BenchmarkRecord.create(
        hardware=hardware(),
        request=request,
        observation=observation,
        identity=identity,
    )


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=_Search([]),
        detect_hardware=hardware,
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
