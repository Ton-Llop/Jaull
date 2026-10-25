"""Preparing an explicit path must not silently replace its backend."""

from types import SimpleNamespace

import pytest

from jaull.advisor.service import (
    AdvisorService,
    _selection_for_requested_backend,
    _vulkan_offload_runtime,
)
from jaull.application.execution import ExecutionPlanningError
from jaull.application.recommendation import service
from jaull.bootstrap.container import ServiceContainer
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
)
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
)
from jaull.runtime.backend_selection import select_runtime_backend
from tests.test_compatible_plan_selection import BASE, GGUF, _setup


@pytest.mark.parametrize("requested,available", [
    (ComputeBackend.CPU, True), (ComputeBackend.VULKAN, True), (ComputeBackend.VULKAN, False),
])
def test_prepare_preserves_explicit_backend_or_refuses_it(monkeypatch, requested, available):
    hw, req, analyses, estimate, evaluated = _setup(backend=ComputeBackend.VULKAN)
    rec = service.recommend([evaluated(GGUF, base=BASE)], req, hardware=hw, estimate_fn=estimate)[0]
    nvidia = AcceleratorProfile(
        name="NVIDIA test GPU", vendor=AcceleratorVendor.NVIDIA, type=AcceleratorType.DEDICATED,
        dedicated_memory_bytes=8 * 1024**3, backends=[ComputeBackendInfo(
            backend=ComputeBackend.CUDA, availability=BackendAvailability.AVAILABLE,
        )],
    )
    hybrid = hw.model_copy(update={"accelerators": [nvidia, *hw.accelerators]})
    selection = _selection_for_requested_backend(select_runtime_backend(hybrid), requested)
    runtime = (
        _vulkan_offload_runtime(rec.plan.runtime)
        if requested is ComputeBackend.VULKAN else rec.plan.runtime
    )
    plan = rec.plan.model_copy(update={"backend_selection": selection, "runtime": runtime})
    if not available:
        hybrid = hybrid.model_copy(update={"accelerators": [nvidia]})
    configs = []

    def estimate_memory(analysis, hardware, inference_cfg, **kwargs):
        configs.append(inference_cfg)
        return estimate(analysis, inference_cfg)

    advisor = AdvisorService(services=ServiceContainer(
        hf_client=SimpleNamespace(), search_client=SimpleNamespace(),
        detect_hardware=lambda: hybrid,
        inspect_model=lambda repo_id, **kwargs: analyses[repo_id], estimate_memory=estimate_memory,
    ))
    capability = LlamaCppRuntimeCapability(
        binary_path="/synthetic/llama-cli", binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[LlamaCppBackendCapability(
            backend=item, state=LlamaCppBackendCapabilityState.CONFIRMED,
            reason=LlamaCppCapabilityReason.BACKEND_EXPOSED, devices=[LlamaCppRuntimeDevice(
                backend=item, runtime_id="Vulkan0" if item is ComputeBackend.VULKAN else "CUDA0",
                name="Test AMD GPU" if item is ComputeBackend.VULKAN else nvidia.name,
            )],
        ) for item in (ComputeBackend.CUDA, ComputeBackend.VULKAN)],
    )
    monkeypatch.setattr(
        AdvisorService, "inspect_llama_cpp_runtime", lambda *args, **kwargs: capability,
    )
    monkeypatch.setattr(
        AdvisorService, "_prepare_plan_artifact", lambda *args: plan.artifact.to_model_artifact(),
    )
    if not available:
        with pytest.raises(ExecutionPlanningError, match=r"selected|Selected"):
            advisor.prepare_execution_plan(plan, hardware=hybrid)
        return
    prepared = advisor.prepare_execution_plan(plan, hardware=hybrid).plan
    assert prepared.backend_selection.selected_backend is requested
    assert prepared.execution_readiness is not None
    assert prepared.execution_readiness.status is ExecutionReadinessStatus.READY
    if requested is ComputeBackend.VULKAN:
        assert prepared.backend_selection.selected_accelerator == selection.selected_accelerator
        assert {flag.name: flag.value for flag in prepared.runtime.flags}["--device"] == "Vulkan0"
    else:
        assert configs[0].target_device is TargetDevice.CPU
        assert not any(flag.name == "--device" for flag in prepared.runtime.flags)


@pytest.mark.parametrize("operation", ["run", "benchmark", "validate"])
def test_preparation_errors_are_reported_by_each_execution_worker(monkeypatch, operation):
    from jaull.application.recommendation.execution_plans import execution_plan_for_recommendation
    from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen
    from jaull.tui.screens.recommendation_execution import RecommendationExecutionScreen
    from jaull.tui.screens.recommendation_validation import RecommendationValidationScreen
    from tests.test_tui_recommendation_execution import _recommendation

    rec = _recommendation()
    plan = execution_plan_for_recommendation(rec)
    errors = []

    def fail(*args, **kwargs):
        raise ExecutionPlanningError("Selected device is unavailable; no fallback")

    advisor = SimpleNamespace(prepare_execution_plan=fail)
    kind = {"run": RecommendationExecutionScreen, "benchmark": RecommendationBenchmarkScreen,
            "validate": RecommendationValidationScreen}[operation]
    screen = kind(rec, plan)
    try:
        if operation == "run":
            monkeypatch.setattr(
                screen, "_post_generation_failed", lambda text, *args: errors.append(text),
            )
            screen._generate_worker(advisor, None, "hi", plan.runtime, None, plan)
        else:
            monkeypatch.setattr(screen, "_post_failed", errors.append)
            if operation == "benchmark":
                screen._benchmark_worker(advisor, None, plan.runtime, plan)
            else:
                screen._validation_worker(advisor, None, plan.memory_prediction,
                                          plan.runtime, plan, None, False)
        assert errors == ["Selected device is unavailable; no fallback"]
    finally:
        screen._shutdown_executor()


@pytest.mark.parametrize("changed", ["sha256", "filename", "revision"])
def test_preparation_refuses_a_different_resolved_artifact(changed):
    hw, req, _, estimate, evaluated = _setup()
    plan = service.recommend([evaluated(GGUF, base=BASE)], req, hardware=hw,
                             estimate_fn=estimate)[0].plan
    selected = plan.artifact.model_copy(update={"sha256": "a" * 64, "revision": "c" * 40})
    plan = plan.model_copy(update={"artifact": selected})
    actual = selected.to_model_artifact().model_copy(update={
        changed: {"sha256": "b" * 64, "filename": "another.gguf", "revision": "d" * 40}[changed],
        "is_downloaded": True,
    })
    verifications = []
    advisor = SimpleNamespace(
        resolve_artifact=lambda **kwargs: actual,
        verify_artifact=lambda artifact, **kwargs: verifications.append(kwargs) or artifact,
    )
    with pytest.raises(ExecutionPlanningError, match="artifact"):
        AdvisorService._prepare_plan_artifact(advisor, plan, plan.runtime, lambda _: None)
    assert not verifications


def test_prepared_artifact_is_fully_verified():
    hw, req, _, estimate, evaluated = _setup()
    plan = service.recommend([evaluated(GGUF, base=BASE)], req, hardware=hw,
                             estimate_fn=estimate)[0].plan
    artifact = plan.artifact.to_model_artifact().model_copy(update={"is_downloaded": True})
    calls = []
    advisor = SimpleNamespace(
        resolve_artifact=lambda **kwargs: artifact,
        verify_artifact=lambda artifact, **kwargs: calls.append(kwargs) or artifact,
    )
    assert AdvisorService._prepare_plan_artifact(
        advisor, plan, plan.runtime, lambda _: None,
    ) == artifact
    assert calls == [{"full": True}]


def test_prepared_plan_retains_the_verified_digest_without_mutating_the_search(monkeypatch):
    hw, req, analyses, estimate, evaluated = _setup()
    plan = service.recommend([evaluated(GGUF, base=BASE)], req, hardware=hw,
                             estimate_fn=estimate)[0].plan
    before = plan.model_dump_json()
    artifact = plan.artifact.to_model_artifact().model_copy(update={"sha256": "b" * 64})
    advisor = AdvisorService(services=ServiceContainer(
        hf_client=SimpleNamespace(), search_client=SimpleNamespace(), detect_hardware=lambda: hw,
        inspect_model=lambda repo, **kwargs: analyses[repo],
        estimate_memory=lambda analysis, hardware, inference_cfg, **kwargs:
            estimate(analysis, inference_cfg),
    ))
    monkeypatch.setattr(AdvisorService, "_prepare_plan_artifact", lambda *args: artifact)
    capability = LlamaCppRuntimeCapability(
        binary_path="/synthetic/llama-cli", binary_status=LlamaCppBinaryStatus.AVAILABLE,
    )
    monkeypatch.setattr(
        AdvisorService, "inspect_llama_cpp_runtime", lambda *args, **kwargs: capability,
    )
    prepared = advisor.prepare_execution_plan(plan, hardware=hw)
    assert prepared.plan.artifact.sha256 == prepared.artifact.sha256 == "b" * 64
    assert plan.model_dump_json() == before
