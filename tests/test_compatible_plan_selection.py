"""Metadata-only plan selection on bounded, synthetic CPU/CUDA/Vulkan pools."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jaull.advisor.service import AdvisorService
from jaull.application.recommendation import service
from jaull.application.recommendation.alternatives import expand_execution_candidates
from jaull.application.requirements import build_requirements
from jaull.bootstrap.container import ServiceContainer
from jaull.discovery.enrichment import evaluate_candidate
from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
)
from jaull.domain.inference import TargetDevice, WeightPrecision
from jaull.domain.requirements import RecommendationPriority
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    PyTorchRuntimeCapability,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeName,
)
from jaull.domain.runtime_support import available_compute_backends
from jaull.exceptions import JaullError
from jaull.observability.telemetry import PerformanceTelemetry
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    generate_execution_plans,
    rank_execution_plans,
    ranking_criteria,
)
from jaull.recommendation.local_evidence import matching_benchmark
from jaull.reporting.recommendation import report_to_dict, report_to_markdown
from jaull.runtime import service as runtimes
from jaull.workflow.orchestrator import run_workflow
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import (
    GIB,
    FakeHfClient,
    FakeSearchClient,
    answers,
    candidate,
    gguf_analysis,
    hardware,
    size_driven_estimator,
    transformers_analysis,
)
from tests.test_recommendation_engine_v2 import _benchmark_record

BASE = "org/Model-7B-Instruct"
GGUF = "community/Model-7B-Instruct-GGUF"


def _setup(*, ram=8, gpu=None, backend=None, native_bytes=14 * GIB):
    hw = hardware(ram_gib=ram, vram_gib=gpu)
    if backend is not None:
        hw = hw.model_copy(
            update={
                "accelerators": [
                    AcceleratorProfile(
                        name="Test AMD GPU",
                        vendor=AcceleratorVendor.AMD,
                        type=AcceleratorType.INTEGRATED,
                        shared_memory=True,
                        backends=[
                            ComputeBackendInfo(
                                backend=backend,
                                availability=BackendAvailability.AVAILABLE,
                            )
                        ],
                    )
                ]
            }
        )
    req = build_requirements(answers(), hw)
    native = transformers_analysis(BASE).model_copy(update={"total_size_bytes": native_bytes})
    gguf = gguf_analysis(GGUF, quantizations=("Q4_K_M", "Q5_K_M", "Q6_K"))
    analyses = {BASE: native, GGUF: gguf}
    budget = (gpu or ram) * GIB
    estimate_size = size_driven_estimator(vram_budget=budget)

    def estimate(analysis, config):
        result = estimate_size(analysis, config)
        device = (
            TargetDevice.CPU
            if config.target_device is TargetDevice.CPU or gpu is None
            else (TargetDevice.GPU)
        )
        result = result.model_copy(
            update={
                "assessment": result.assessment.model_copy(update={"effective_device": device}),
                "weights": result.weights.model_copy(update={"precision": config.precision}),
            }
        )
        return result.model_copy(
            update={
                "runtime_recommendation": runtimes.recommend(
                    result,
                    hw,
                    analysis.classification.primary_type,
                )
            }
        )

    def evaluated(repo, *, base=None):
        return evaluate_candidate(
            candidate(repo, base_model=base),
            req,
            hw,
            inspect_fn=analyses.__getitem__,
            estimate_fn=estimate,
        )

    return hw, req, analyses, estimate, evaluated


def test_cpu_selects_a_confirmed_viable_gguf_and_preserves_rejected_transformers_paths():
    hw, req, _, estimate, evaluated = _setup()
    original = evaluated(BASE)
    assert original.selected_configuration.precision is WeightPrecision.FLOAT16
    assert original.memory_estimate.assessment.status is CompatibilityStatus.INSUFFICIENT
    recs = service.recommend(
        [original, evaluated(GGUF, base=BASE)],
        req,
        hardware=hw,
        estimate_fn=estimate,
    )
    assert len(recs) == 1
    assert recs[0].plan.runtime_family is RuntimeName.LLAMA_CPP
    assert recs[0].plan.artifact.quantization == "Q5_K_M"  # Balanced ladder head
    assert recs[0].alternative_plans
    assert any(
        plan.runtime_family is RuntimeName.TRANSFORMERS for plan in recs[0].alternative_plans
    )


def test_native_fp16_that_fits_does_not_trigger_gguf_discovery():
    hw, req, _, estimate, evaluated = _setup(native_bytes=2 * GIB)
    search = FakeSearchClient()

    def forbidden_inspection(_):
        pytest.fail("a viable native plan must not trigger alternative inspection")

    result = expand_execution_candidates(
        [evaluated(BASE)],
        candidates=[],
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=forbidden_inspection,
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    recs = service.recommend(result, req, hardware=hw, estimate_fn=estimate)
    assert recs[0].plan.runtime_family is RuntimeName.TRANSFORMERS
    assert recs[0].plan.artifact.precision == "float16"
    assert not search.seen


def test_cpu_without_a_confirmed_conversion_reports_no_viable_plan():
    hw, req, _, estimate, evaluated = _setup()
    assert service.recommend([evaluated(BASE)], req, hardware=hw, estimate_fn=estimate) == []


def test_cuda_retains_a_supported_quantized_transformers_plan():
    hw, req, _, estimate, evaluated = _setup(gpu=8)
    recs = service.recommend([evaluated(BASE)], req, hardware=hw, estimate_fn=estimate)
    assert recs[0].plan.runtime_family is RuntimeName.TRANSFORMERS
    assert recs[0].plan.artifact.precision in {"int8", "int4"}
    assert recs[0].plan.backend_selection.selected_backend is ComputeBackend.CUDA


def test_available_vulkan_allows_gguf_but_not_transformers_quantization():
    hw, req, _, estimate, evaluated = _setup(backend=ComputeBackend.VULKAN)
    recs = service.recommend(
        [evaluated(BASE), evaluated(GGUF, base=BASE)],
        req,
        hardware=hw,
        estimate_fn=estimate,
    )
    assert available_compute_backends(hw) == {ComputeBackend.CPU, ComputeBackend.VULKAN}
    assert recs[0].plan.runtime_family is RuntimeName.LLAMA_CPP
    # Shared RAM is not invented as a second memory pool or as proven GPU offload.
    assert recs[0].plan.memory_prediction.assessment.effective_device is TargetDevice.CPU


@pytest.mark.parametrize("backend", [ComputeBackend.VULKAN, ComputeBackend.HIP])
def test_observed_amd_backend_supports_a_valid_gpu_placement(backend):
    hw, req, analyses, estimate, _ = _setup()
    hw = hw.model_copy(
        update={
            "accelerators": [
                AcceleratorProfile(
                    name="AMD test device",
                    vendor=AcceleratorVendor.AMD,
                    type=AcceleratorType.DEDICATED,
                    dedicated_memory_bytes=8 * GIB,
                    available_memory_bytes=8 * GIB,
                    backends=[
                        ComputeBackendInfo(
                            backend=backend, availability=BackendAvailability.AVAILABLE
                        )
                    ],
                )
            ]
        }
    )

    def placed_estimate(analysis, config):
        result = estimate(analysis, config)
        use_gpu = config.target_device is not TargetDevice.CPU
        result = result.model_copy(
            update={
                "assessment": result.assessment.model_copy(
                    update={
                        "effective_device": TargetDevice.GPU if use_gpu else TargetDevice.CPU,
                    }
                )
            }
        )
        return result.model_copy(
            update={
                "runtime_recommendation": runtimes.recommend(
                    result,
                    hw,
                    analysis.classification.primary_type,
                )
            }
        )

    repos = [BASE, GGUF] if backend is ComputeBackend.VULKAN else [BASE]
    items = [
        evaluate_candidate(
            candidate(repo, base_model=BASE if repo == GGUF else None),
            req,
            hw,
            inspect_fn=analyses.__getitem__,
            estimate_fn=placed_estimate,
        )
        for repo in repos
    ]
    recs = service.recommend(items, req, hardware=hw, estimate_fn=placed_estimate)
    assert recs[0].plan.backend_selection.selected_backend is backend
    assert recs[0].plan.runtime_family is (
        RuntimeName.LLAMA_CPP if backend is ComputeBackend.VULKAN else RuntimeName.TRANSFORMERS
    )


def test_physical_gpu_without_an_available_backend_cannot_authorize_quantization():
    hw, req, _, estimate, evaluated = _setup(gpu=8, backend=ComputeBackend.VULKAN)
    blocked = hw.accelerators[0].model_copy(
        update={
            "backends": [
                ComputeBackendInfo(
                    backend=ComputeBackend.CUDA,
                    availability=BackendAvailability.UNAVAILABLE,
                )
            ]
        }
    )
    hw = hw.model_copy(update={"accelerators": [blocked]})
    plans = generate_execution_plans(
        evaluated(BASE),
        req,
        context=PlanRankingContext(hardware=hw),
        estimate_fn=estimate,
    )
    assert available_compute_backends(hw) == {ComputeBackend.CPU}
    assert all(
        plan.compatibility.value == "not_compatible"
        for plan in plans
        if plan.artifact.precision in {"int4", "int8"}
    )


def test_missing_llama_cpp_does_not_change_the_selected_gguf_configuration():
    hw, req, _, estimate, evaluated = _setup()
    selection = RuntimeBackendSelection(
        selected_backend=ComputeBackend.CPU,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK,
    )
    missing = ExecutionReadiness(
        status=ExecutionReadinessStatus.NOT_READY,
        reason=ExecutionReadinessReason.RUNTIME_MISSING,
        selection=selection,
        runtime_capability=LlamaCppRuntimeCapability(binary_status=LlamaCppBinaryStatus.MISSING),
    )
    items = [evaluated(BASE), evaluated(GGUF, base=BASE)]
    plain = service.recommend(items, req, hardware=hw, estimate_fn=estimate)
    with_missing = service.recommend(
        items,
        req,
        hardware=hw,
        estimate_fn=estimate,
        plan_context=PlanRankingContext(
            hardware=hw,
            backend_selection=selection,
            readiness_by_runtime={RuntimeName.LLAMA_CPP: missing},
        ),
    )
    assert plain[0].plan.plan_id == with_missing[0].plan.plan_id
    assert with_missing[0].plan.execution_readiness.status is ExecutionReadinessStatus.NOT_READY


def test_name_only_conversion_does_not_merge_with_the_native_model():
    hw, req, _, estimate, evaluated = _setup(native_bytes=2 * GIB)
    recs = service.recommend(
        [evaluated(BASE), evaluated("community/Model-7B-Instruct-GGUF")],
        req,
        hardware=hw,
        estimate_fn=estimate,
    )
    assert len(recs) == 2
    assert all(
        not any(
            path.artifact.repo_id != rec.plan.artifact.repo_id for path in rec.alternative_plans
        )
        for rec in recs
    )


def test_search_reuses_inspected_conversion_without_more_inspections_or_execution():
    hw, _, analyses, estimate, _ = _setup()
    inspected = []
    search = FakeSearchClient(default=[candidate(BASE), candidate(GGUF, base_model=BASE)])

    def inspect(repo_id, **kwargs):
        inspected.append(repo_id)
        return analyses[repo_id]

    def estimate_in_workflow(analysis, hardware, inference_cfg, **kwargs):
        return estimate(analysis, inference_cfg)

    services = ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=lambda: hw,
        inspect_model=inspect,
        estimate_memory=estimate_in_workflow,
    )
    result = run_workflow(answers(), hw, services)
    assert result.recommendations[0].plan.runtime_family is RuntimeName.LLAMA_CPP
    assert inspected.count(BASE) == inspected.count(GGUF) == 1
    assert not any(query.label == "artifact-variants" for query in search.seen)


def test_targeted_discovery_confirms_lineage_and_stays_bounded():
    hw, req, analyses, estimate, evaluated = _setup()
    seen = []
    search = FakeSearchClient(default=[candidate(GGUF, base_model=BASE)])

    def inspect(item):
        seen.append(item.repo_id)
        return analyses[item.repo_id]

    expanded = expand_execution_candidates(
        [evaluated(BASE)],
        candidates=[],
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=inspect,
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    assert seen == [GGUF]
    assert len(search.seen) == 1
    recommendations = service.recommend(expanded, req, hardware=hw, estimate_fn=estimate)
    assert recommendations[0].plan.artifact.repo_id == GGUF


@pytest.mark.parametrize(
    "priority,quantization",
    [
        (RecommendationPriority.QUALITY, "Q6_K"),
        (RecommendationPriority.BALANCED, "Q5_K_M"),  # head of its ladder
        (RecommendationPriority.SPEED, "Q4_K_M"),
        (RecommendationPriority.MEMORY, "Q4_K_M"),
    ],
)
def test_multiple_viable_quants_follow_existing_policy(priority, quantization):
    hw, req, _, estimate, evaluated = _setup()
    req = req.model_copy(update={"priority": priority})
    recs = service.recommend(
        [evaluated(BASE), evaluated(GGUF, base=BASE)],
        req,
        hardware=hw,
        estimate_fn=estimate,
    )
    assert recs[0].plan.artifact.quantization == quantization
    assert {plan.artifact.quantization for plan in recs[0].alternative_plans} >= (
        {"Q4_K_M", "Q5_K_M", "Q6_K"} - {quantization}
    )


def test_equally_comfortable_quantizations_follow_the_priority_ladder_not_the_alphabet():
    """Regression found on a B60 box: when every quantization fit equally, the plans fell back
    to alphabetical order and Balanced recommended Q2_K (and a 32B at IQ2_M)."""
    hw, req, _, estimate, evaluated = _setup(ram=64)
    req = req.model_copy(update={"priority": RecommendationPriority.BALANCED})

    recs = service.recommend(
        [evaluated(BASE), evaluated(GGUF, base=BASE)], req, hardware=hw, estimate_fn=estimate,
    )

    assert recs[0].plan.artifact.quantization == "Q5_K_M"  # head of the Balanced ladder


@pytest.mark.parametrize("priority", list(RecommendationPriority))
@pytest.mark.parametrize("commercial", [True, False])
def test_published_criteria_reproduce_order_for_variants_of_one_repository(priority, commercial):
    hw, req, _, estimate, evaluated = _setup(ram=64)
    req = req.model_copy(update={"priority": priority, "commercial_use_required": commercial})
    ranked = rank_execution_plans(
        [evaluated(GGUF, base=BASE)], req,
        context=PlanRankingContext(hardware=hw), estimate_fn=estimate,
    )
    assert len(ranked) > 1
    by_published_criteria = sorted(
        ranked,
        key=lambda item: (
            tuple(c.rank for c in ranking_criteria(
                item, priority, commercial_use_required=commercial,
            )),
            item.evaluated.repo_id,
            item.plan.artifact.quantization or item.plan.artifact.precision or "",
            item.plan.runtime.runtime.value,
        ),
    )
    assert [item.plan.plan_id for item in ranked] == [
        item.plan.plan_id for item in by_published_criteria
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_balanced_ladder_beats_repository_name_for_the_same_confirmed_model(reverse):
    hw, req, analyses, estimate, _ = _setup(ram=64)
    req = req.model_copy(update={"priority": RecommendationPriority.BALANCED})
    source = analyses[GGUF]
    pool = []
    for repo, quant in (("aaa/Model-7B-GGUF", "Q4_K_M"), ("zzz/Model-7B-GGUF", "Q5_K_M")):
        analyses[repo] = source.model_copy(update={
            "repo": source.repo.model_copy(update={"repo_id": repo}),
            "classification": source.classification.model_copy(update={"gguf_variants": [
                variant for variant in source.classification.gguf_variants
                if variant.quantization == quant
            ]}),
        })
        pool.append(evaluate_candidate(
            candidate(repo, base_model=BASE), req, hw,
            inspect_fn=analyses.__getitem__, estimate_fn=estimate,
        ))
    if reverse:
        pool.reverse()
    recs = service.recommend(pool, req, hardware=hw, estimate_fn=estimate)
    assert len(recs) == 1  # confirmed lineage, not matching names
    assert recs[0].plan.artifact.repo_id == "zzz/Model-7B-GGUF"
    assert recs[0].plan.artifact.quantization == "Q5_K_M"
    assert any(plan.artifact.repo_id == "aaa/Model-7B-GGUF" for plan in recs[0].alternative_plans)


def test_uncertain_conversion_cannot_rescue_a_blocked_model():
    hw, req, analyses, estimate, evaluated = _setup()
    search = FakeSearchClient(default=[candidate(GGUF)])
    expanded = expand_execution_candidates(
        [evaluated(BASE)],
        candidates=[],
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=lambda item: analyses[item.repo_id],
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    assert [item.repo_id for item in expanded] == [BASE]
    assert service.recommend(expanded, req, hardware=hw, estimate_fn=estimate) == []


def test_known_uninspected_conversion_is_reused_before_searching():
    hw, req, analyses, estimate, evaluated = _setup()
    search = FakeSearchClient()
    expanded = expand_execution_candidates(
        [evaluated(BASE)],
        candidates=[candidate(GGUF, base_model=BASE)],
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=lambda item: analyses[item.repo_id],
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    assert len(expanded) == 2
    assert not search.seen


def test_explicit_finetune_relation_is_not_a_conversion_of_the_original_model():
    hw, req, analyses, estimate, evaluated = _setup()
    derivative = candidate(GGUF, base_model=BASE, tags=[f"base_model:finetune:{BASE}", "gguf"])
    search = FakeSearchClient(default=[derivative])
    expanded = expand_execution_candidates(
        [evaluated(BASE)],
        candidates=[derivative],
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=lambda item: analyses[item.repo_id],
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    assert [item.repo_id for item in expanded] == [BASE]


def test_failed_variant_estimation_keeps_other_viable_plans_and_a_diagnostic():
    hw, req, _, estimate, evaluated = _setup()

    def fail_one(analysis, config):
        if config.quantization == "Q5_K_M":
            raise JaullError("synthetic missing metadata")
        return estimate(analysis, config)

    recs = service.recommend(
        [evaluated(GGUF, base=BASE)],
        req,
        hardware=hw,
        estimate_fn=fail_one,
    )
    assert recs[0].plan.artifact.quantization == "Q4_K_M"
    unknown = next(
        plan for plan in recs[0].alternative_plans if plan.artifact.quantization == "Q5_K_M"
    )
    assert unknown.memory_prediction is None
    assert any("synthetic missing metadata" in warning for warning in unknown.warnings)


def test_targeted_inspection_budget_is_global_not_per_model():
    hw, req, _, estimate, evaluated = _setup()
    conversions = [candidate(f"community/Variant-{i}-GGUF", base_model=BASE) for i in range(12)]
    inspected = []

    def too_large(item):
        inspected.append(item.repo_id)
        return gguf_analysis(item.repo_id, base_bytes=40 * GIB)

    search = FakeSearchClient()
    expand_execution_candidates(
        [evaluated(BASE)],
        candidates=conversions,
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=too_large,
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    assert len(inspected) == 6
    assert not search.seen


def test_targeted_search_budget_is_global_when_no_conversions_are_found():
    hw, req, _, estimate, evaluated = _setup()
    original = evaluated(BASE)
    items = [
        original.model_copy(
            update={
                "candidate": candidate(f"org/Model-{i}-7B-Instruct"),
                "analysis": original.analysis.model_copy(
                    update={
                        "repo": original.analysis.repo.model_copy(
                            update={
                                "repo_id": f"org/Model-{i}-7B-Instruct",
                            }
                        ),
                    }
                ),
            }
        )
        for i in range(10)
    ]
    search = FakeSearchClient()
    expanded = expand_execution_candidates(
        items,
        candidates=[],
        requirements=req,
        hardware=hw,
        search_client=search,
        inspect_model=lambda _: pytest.fail("no search hits"),
        estimate_fn=estimate,
        check_cancelled=lambda: None,
        telemetry=PerformanceTelemetry(),
    )
    assert len(search.seen) == 3
    assert [item.repo_id for item in expanded] == [item.repo_id for item in items]
    assert all("No technically viable" in " ".join(item.warnings) for item in expanded)


def test_generated_alternative_does_not_inherit_a_different_sha_or_backend():
    hw, req, _, estimate, evaluated = _setup()
    plan = generate_execution_plans(
        evaluated(GGUF, base=BASE),
        req,
        context=PlanRankingContext(hardware=hw),
        estimate_fn=estimate,
    )[0]
    plan = plan.model_copy(
        update={"artifact": plan.artifact.model_copy(update={"sha256": "a" * 64})}
    )
    artifact = plan.artifact.to_model_artifact().model_copy(update={"sha256": "a" * 64})
    record = _benchmark_record(artifact, plan.runtime, machine=hw, tps=12.0)
    assert matching_benchmark(plan, [record]) is record
    different_sha = record.model_copy(
        update={
            "artifact": record.artifact.model_copy(update={"sha256": "b" * 64}),
        }
    )
    assert matching_benchmark(plan, [different_sha]) is None
    different_backend = record.model_copy(update={"requested_backend": ComputeBackend.CUDA})
    assert matching_benchmark(plan, [different_backend]) is None


def test_confirmed_conversion_does_not_bypass_commercial_license_gate():
    hw, req, analyses, estimate, evaluated = _setup()
    restricted = evaluate_candidate(
        candidate(GGUF, base_model=BASE, license_value="cc-by-nc-4.0"),
        req,
        hw,
        inspect_fn=analyses.__getitem__,
        estimate_fn=estimate,
    )
    assert (
        service.recommend(
            [evaluated(BASE), restricted],
            req,
            hardware=hw,
            estimate_fn=estimate,
        )
        == []
    )


def test_selected_plan_drives_tui_helpers_and_json_markdown_export():
    from jaull.application.recommendation.execution_plans import (
        execution_plan_for_recommendation,
        variant_from_recommendation,
    )
    from jaull.presentation.plan_labels import readiness_label
    from jaull.tui.screens.recommendation_results import _recommendation_metadata

    hw, req, _, estimate, evaluated = _setup()
    recs = service.recommend(
        [evaluated(BASE), evaluated(GGUF, base=BASE)],
        req,
        hardware=hw,
        estimate_fn=estimate,
    )
    rec = recs[0]
    # The ladder chooses Q5_K_M and the plan order now agrees with it. It used to end on
    # Q4_K_M only because equal plans were ordered by the quantization's name.
    assert evaluated(GGUF, base=BASE).selected_configuration.quantization == "Q5_K_M"
    assert rec.evaluated.selected_configuration.quantization == "Q5_K_M"
    assert rec.plan.artifact.quantization == "Q5_K_M"
    assert "Q5_K_M" in _recommendation_metadata(rec)
    assert "Q4_K_M" not in _recommendation_metadata(rec)
    assert variant_from_recommendation(rec) is rec.plan.artifact
    assert execution_plan_for_recommendation(rec, hardware=hw).artifact == rec.plan.artifact
    blocked = next(plan for plan in rec.alternative_plans if plan.artifact.precision == "int4")
    assert readiness_label(blocked) == "Unsupported configuration"
    state = RecommendationWorkflowState(
        hardware=hw,
        requirements=req,
        recommendations=recs,
        evaluated_candidates=[rec.evaluated],
    )
    payload = report_to_dict(state)
    assert payload["recommendations"][0]["artifact"]["quantization"] == "Q5_K_M"
    criteria = payload["recommendations"][0]["ranking"]["criteria"]
    assert {"axis": "quantization_preference", "value": "Q5_K_M"} in criteria
    markdown = report_to_markdown(state)
    assert "Q5_K_M" in markdown
    assert "quantization preference: Q5_K_M" in markdown


def test_paths_keeps_the_selected_configuration_and_already_assessed_alternatives():
    hw, req, _, estimate, evaluated = _setup()
    rec = service.recommend(
        [evaluated(BASE), evaluated(GGUF, base=BASE)],
        req,
        hardware=hw,
        estimate_fn=estimate,
    )[0]

    class PathsAdvisor(AdvisorService):
        def discover_artifact_variants(self, **kwargs):
            return [rec.plan.artifact]

        def inspect_llama_cpp_runtime(self, **kwargs):
            return LlamaCppRuntimeCapability(binary_status=LlamaCppBinaryStatus.MISSING)

        def inspect_pytorch_runtime(self):
            return PyTorchRuntimeCapability(runtime_status=PyTorchRuntimeStatus.TORCH_MISSING)

        def _artifacts(self):
            return SimpleNamespace(is_downloaded=lambda _: False)

    advisor = PathsAdvisor(
        services=ServiceContainer(
            hf_client=FakeHfClient(),
            search_client=FakeSearchClient(),
            detect_hardware=lambda: hw,
            inspect_model=lambda *args, **kwargs: pytest.fail("already inspected"),
            estimate_memory=lambda *args, **kwargs: pytest.fail("already estimated"),
        )
    )
    paths = advisor.execution_plans_for_recommendation(rec)
    by_id = {plan.plan_id: plan for plan in paths}
    assert set(by_id) == {rec.plan.plan_id, *(plan.plan_id for plan in rec.alternative_plans)}
    for original in [rec.plan, *rec.alternative_plans]:
        assert by_id[original.plan_id].artifact == original.artifact
        assert by_id[original.plan_id].memory_prediction == original.memory_prediction
        assert by_id[original.plan_id].runtime == original.runtime
    assert all(
        any(text.startswith("Not selected:") for text in by_id[plan.plan_id].warnings)
        for plan in rec.alternative_plans
    )


@pytest.mark.parametrize("runtime_exposes_vulkan", [True, False])
def test_integrated_vulkan_is_an_explicit_path_without_replacing_cpu(
    runtime_exposes_vulkan: bool,
):
    hw, req, _, estimate, evaluated = _setup(backend=ComputeBackend.VULKAN)
    rec = service.recommend(
        [evaluated(GGUF, base=BASE)], req, hardware=hw, estimate_fn=estimate
    )[0]
    assert rec.plan.runtime.flags
    assert any(
        flag.name == "--n-gpu-layers" and flag.value == "0"
        for flag in rec.plan.runtime.flags
    )

    device = LlamaCppRuntimeDevice(
        backend=ComputeBackend.VULKAN,
        runtime_id="Vulkan0",
        name="Test AMD GPU",
    )
    capability = LlamaCppRuntimeCapability(
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
            backend_capabilities=[
                LlamaCppBackendCapability(
                    backend=ComputeBackend.VULKAN,
                    state=(
                        LlamaCppBackendCapabilityState.CONFIRMED
                        if runtime_exposes_vulkan
                        else LlamaCppBackendCapabilityState.NOT_OBSERVED
                    ),
                    devices=[device] if runtime_exposes_vulkan else [],
                    reason=(
                        LlamaCppCapabilityReason.BACKEND_EXPOSED
                        if runtime_exposes_vulkan
                        else LlamaCppCapabilityReason.BACKEND_NOT_OBSERVED
                    ),
                )
            ],
    )
    inspected_selections = []

    class PathsAdvisor(AdvisorService):
        def discover_artifact_variants(self, **kwargs):
            return [rec.plan.artifact]

        def inspect_llama_cpp_runtime(self, **kwargs):
            inspected_selections.append(kwargs.get("selection"))
            return capability

        def _artifacts(self):
            return SimpleNamespace(is_downloaded=lambda _: True)

    advisor = PathsAdvisor(
        services=ServiceContainer(
            hf_client=FakeHfClient(),
            search_client=FakeSearchClient(),
            detect_hardware=lambda: hw,
            inspect_model=lambda *args, **kwargs: pytest.fail("already inspected"),
            estimate_memory=lambda *args, **kwargs: pytest.fail("already estimated"),
        )
    )

    plans = advisor.execution_plans_for_recommendation(rec)
    assert any(
        item is not None and item.selected_backend is ComputeBackend.VULKAN
        for item in inspected_selections
    )
    cpu_plan = next(
        plan for plan in plans
        if plan.backend_selection is not None
        and plan.backend_selection.selected_backend is ComputeBackend.CPU
    )
    vulkan_plan = next(
        plan for plan in plans
        if plan.backend_selection is not None
        and plan.backend_selection.selected_backend is ComputeBackend.VULKAN
    )
    assert cpu_plan.runtime.flags == rec.plan.runtime.flags
    assert vulkan_plan.plan_id != cpu_plan.plan_id
    assert any(
        flag.name == "--n-gpu-layers" and flag.value == "-1"
        for flag in vulkan_plan.runtime.flags
    )
    assert vulkan_plan.execution_readiness is not None
    assert vulkan_plan.execution_readiness.status is (
        ExecutionReadinessStatus.READY
        if runtime_exposes_vulkan
        else ExecutionReadinessStatus.NOT_READY
    )
    assert any("shares system RAM" in warning for warning in vulkan_plan.warnings)
    # Reopening an already selected Vulkan path must not add the same path twice.
    reopened = advisor.execution_plans_for_recommendation(
        rec.model_copy(update={"plan": vulkan_plan})
    )
    assert len({plan.plan_id for plan in reopened}) == len(reopened)
    assert {
        plan.artifact.quantization for plan in reopened
        if plan.backend_selection.selected_backend is ComputeBackend.VULKAN
    } == {plan.artifact.quantization for plan in plans}
