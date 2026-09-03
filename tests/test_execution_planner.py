from __future__ import annotations

import pytest

from jaull.application.execution import (
    ExecutionOverrides,
    ExecutionPlanningError,
    plan_launch,
)
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.runtime import RuntimeFlagSource, RuntimeName, RuntimeRecommendation
from tests._execution_fixtures import qwen_ctx4096_estimate, qwen_hardware


def _flag(rec: RuntimeRecommendation, name: str) -> str | None:
    return next((f.value for f in rec.flags if f.name == name), None)


def _source(rec: RuntimeRecommendation, name: str) -> RuntimeFlagSource | None:
    return next((f.source for f in rec.flags if f.name == name), None)


def test_auto_uses_the_precomputed_launch_policy_number() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=True),
        hardware=qwen_hardware(),
        overrides=ExecutionOverrides(),
    )
    assert _flag(rec, "--n-gpu-layers") == "23"
    assert _flag(rec, "--ctx-size") == "4096"
    assert _source(rec, "--n-gpu-layers") is not RuntimeFlagSource.USER_INPUT


def test_auto_recomputes_when_the_estimate_has_no_recommendation() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=False),
        hardware=qwen_hardware(),
        overrides=ExecutionOverrides(),
    )
    assert _flag(rec, "--n-gpu-layers") == "23"


def test_no_estimate_and_no_layer_override_raises_not_cpu() -> None:
    with pytest.raises(ExecutionPlanningError):
        plan_launch(
            runtime=RuntimeName.LLAMA_CPP,
            estimate=None,
            hardware=None,
            overrides=ExecutionOverrides(),
        )


def test_explicit_zero_is_a_user_override_meaning_cpu_and_needs_no_estimate() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=None,
        hardware=None,
        overrides=ExecutionOverrides(n_gpu_layers=0),
    )
    assert _flag(rec, "--n-gpu-layers") == "0"
    assert _source(rec, "--n-gpu-layers") is RuntimeFlagSource.USER_INPUT
    # ctx is filled at the documented default, but NOT as a user input
    assert _flag(rec, "--ctx-size") == "4096"
    assert _source(rec, "--ctx-size") is RuntimeFlagSource.POLICY


def test_positive_override_is_preserved_exactly_with_explicit_ctx() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=None,
        hardware=None,
        overrides=ExecutionOverrides(context_size=8192, n_gpu_layers=12),
    )
    assert [(f.name, f.value, f.source) for f in rec.flags] == [
        ("--ctx-size", "8192", RuntimeFlagSource.USER_INPUT),
        ("--n-gpu-layers", "12", RuntimeFlagSource.USER_INPUT),
    ]


def test_override_beats_the_automatic_policy_and_keeps_flag_position() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=True),
        hardware=qwen_hardware(),
        overrides=ExecutionOverrides(n_gpu_layers=10),
    )
    assert _flag(rec, "--n-gpu-layers") == "10"  # not 23
    assert _source(rec, "--n-gpu-layers") is RuntimeFlagSource.USER_INPUT
    names = [f.name for f in rec.flags]
    assert (
        names.index("--ctx-size")
        < names.index("--n-gpu-layers")
        < names.index("--batch-size")
    )


def test_ctx_override_upgrades_the_estimate_flag_in_place_to_user_input() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=True),
        hardware=qwen_hardware(),
        overrides=ExecutionOverrides(context_size=4096),  # same value, but user-typed
    )
    assert _flag(rec, "--ctx-size") == "4096"
    assert _source(rec, "--ctx-size") is RuntimeFlagSource.USER_INPUT
    assert next(f.name for f in rec.flags) == "--model"  # position unchanged


def test_omitted_ctx_leaves_the_automatic_flag_untouched() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=True),
        hardware=qwen_hardware(),
        overrides=ExecutionOverrides(),
    )
    assert _source(rec, "--ctx-size") is not RuntimeFlagSource.USER_INPUT


def test_planner_adds_no_new_enum_and_uses_runtime_flag_source() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=None,
        hardware=None,
        overrides=ExecutionOverrides(n_gpu_layers=5),
    )
    assert all(isinstance(f.source, RuntimeFlagSource) for f in rec.flags)


def test_no_estimate_bare_recommendation_is_unknown_confidence() -> None:
    rec = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=None,
        hardware=None,
        overrides=ExecutionOverrides(n_gpu_layers=5),
    )
    assert rec.confidence is EstimationConfidence.UNKNOWN


# ---------------------------------------------------------------------------
# Routing: the TUI's prepare_execution_plan must obtain its runtime from the
# planner, evaluate the policy once, and feed the same resolved runtime into
# both the artifact preparation and the final ExecutionPlan.
# ---------------------------------------------------------------------------
def test_prepare_execution_plan_routes_through_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaull.advisor.service import AdvisorService
    from jaull.domain.artifacts import ModelArtifact
    from jaull.domain.execution_plans import (
        ArtifactVariant,
        ArtifactVariantFormat,
        ExecutionPlan,
        ModelIdentity,
    )
    from jaull.domain.hardware import ComputeBackend
    from jaull.domain.runtime import (
        ExecutionReadiness,
        ExecutionReadinessReason,
        ExecutionReadinessStatus,
        LlamaCppBinaryStatus,
        LlamaCppRuntimeCapability,
        RuntimeBackendSelection,
        RuntimeBackendSelectionReason,
    )
    from jaull.workflow.container import ServiceContainer

    estimate = qwen_ctx4096_estimate(with_runtime_recommendation=True)
    identity = ModelIdentity(model_name="Qwen2.5-7B-Instruct")
    variant = ArtifactVariant(
        model_identity=identity,
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        revision="main",
        format=ArtifactVariantFormat.GGUF,
        filename="qwen.gguf",
        quantization="Q4_K_M",
        compatible_runtimes=[RuntimeName.LLAMA_CPP],
    )
    bare_plan = ExecutionPlan(
        plan_id="p",
        model_identity=identity,
        artifact=variant,
        runtime=RuntimeRecommendation(
            runtime=RuntimeName.LLAMA_CPP, confidence=EstimationConfidence.UNKNOWN
        ),
        runtime_family=RuntimeName.LLAMA_CPP,
    )
    selection = RuntimeBackendSelection(
        selected_backend=ComputeBackend.CPU,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK,
    )
    capability = LlamaCppRuntimeCapability(
        binary_path="/tmp/llama-cli", binary_status=LlamaCppBinaryStatus.AVAILABLE
    )
    readiness = ExecutionReadiness(
        status=ExecutionReadinessStatus.READY,
        reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
        selection=selection,
        runtime_capability=capability,
    )
    ready_artifact = ModelArtifact(
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        revision="main",
        filename="qwen.gguf",
        format="gguf",
        quantization="Q4_K_M",
        size_bytes=4,
        local_path=None,
        sha256="deadbeef",
        is_downloaded=True,
        is_verified=True,
    )

    services = ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=qwen_hardware,
        inspect_model=lambda *args, **kwargs: object(),  # type: ignore[arg-type]
        estimate_memory=lambda **kwargs: estimate,  # type: ignore[arg-type]
    )
    advisor = AdvisorService(services=services)

    launched: list[RuntimeRecommendation] = []
    original_plan_launch = AdvisorService.plan_launch

    def spy_plan_launch(self: AdvisorService, **kwargs: object) -> RuntimeRecommendation:
        rec = original_plan_launch(self, **kwargs)  # type: ignore[arg-type]
        launched.append(rec)
        return rec

    monkeypatch.setattr(AdvisorService, "plan_launch", spy_plan_launch)
    monkeypatch.setattr(
        AdvisorService, "select_runtime_backend", lambda self, hardware=None: selection
    )
    monkeypatch.setattr(
        AdvisorService, "inspect_llama_cpp_runtime", lambda self, **kwargs: capability
    )
    monkeypatch.setattr(
        AdvisorService, "evaluate_execution_readiness", lambda self, **kwargs: readiness
    )
    monkeypatch.setattr(
        AdvisorService,
        "_prepare_plan_artifact",
        lambda self, plan, runtime, report: ready_artifact,
    )

    prepared = advisor.prepare_execution_plan(bare_plan, hardware=qwen_hardware())

    assert len(launched) == 1, "prepare_execution_plan must call plan_launch exactly once"
    assert prepared.plan.runtime == launched[0]
    assert (
        next(f.value for f in prepared.plan.runtime.flags if f.name == "--n-gpu-layers")
        == "23"
    )


def test_cli_run_hands_the_planner_built_runtime_to_the_runner() -> None:
    """Enter through cli/run.py: the executed runtime must be the planner's plan."""
    from jaull.cli.run import RunOptions, run_model
    from tests.test_cli_run import _artifact, _FakeAdvisor

    advisor = _FakeAdvisor(_artifact(is_downloaded=True, is_verified=True))
    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization="Q4_K_M", prompt="Hi"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    _, _, runtime = advisor.runs[0]
    assert runtime is advisor.last_plan.runtime  # not rebuilt inline
    assert next(f.value for f in runtime.flags if f.name == "--n-gpu-layers") == "23"


def test_cli_and_tui_planner_inputs_yield_identical_llama_cpp_flags() -> None:
    """Same estimate + hardware + no overrides via both AdvisorService entrypoints."""
    from jaull.advisor.service import AdvisorService
    from jaull.domain.execution_plans import (
        ArtifactVariant,
        ArtifactVariantFormat,
        ModelIdentity,
    )
    from jaull.workflow.container import ServiceContainer

    estimate = qwen_ctx4096_estimate(with_runtime_recommendation=True)
    services = ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=qwen_hardware,
        inspect_model=lambda *args, **kwargs: object(),  # type: ignore[arg-type]
        estimate_memory=lambda **kwargs: estimate,  # type: ignore[arg-type]
    )
    advisor = AdvisorService(services=services)
    identity = ModelIdentity(model_name="Qwen2.5-7B-Instruct")
    variant = ArtifactVariant(
        model_identity=identity,
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        format=ArtifactVariantFormat.GGUF,
        filename="qwen.gguf",
        quantization="Q4_K_M",
        compatible_runtimes=[RuntimeName.LLAMA_CPP],
    )

    cli_plan = advisor.plan_execution(
        model_identity=identity,
        artifact=variant,
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=qwen_hardware(),
    )
    tui_launch = advisor.plan_launch(
        runtime=RuntimeName.LLAMA_CPP, estimate=estimate, hardware=qwen_hardware()
    )

    assert [(f.name, f.value, f.source) for f in cli_plan.runtime.flags] == [
        (f.name, f.value, f.source) for f in tui_launch.flags
    ]
    assert (
        next(f.value for f in cli_plan.runtime.flags if f.name == "--n-gpu-layers")
        == "23"
    )
