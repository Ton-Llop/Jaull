from dataclasses import replace
from pathlib import Path

import pytest

from jaull.application.execution.planner import ExecutionOverrides, plan_launch
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import RuntimeName
from jaull.runtime.llama_cpp_launch_policy import pick_gpu_layers
from jaull.runtime.llama_cpp_tensor_policy import (
    LlamaCppTensorContext,
    TensorLayerBudget,
    inspect_tensor_context,
    select_tensor_budget,
)
from jaull.runtime.policies import LLAMA_CPP_HEADROOM_BYTES
from tests._execution_fixtures import qwen_ctx4096_estimate, qwen_hardware
from tests._tensor_fixtures import cuda_selection, tensor_file, verified_capability


def local_case(tmp_path: Path, **kwargs: object) -> tuple[MemoryEstimate, LlamaCppTensorContext]:
    artifact = tensor_file(tmp_path, **kwargs)  # type: ignore[arg-type]
    context = inspect_tensor_context(
        artifact,
        qwen_hardware(),
        cuda_selection(),
        verified_capability(),
    )
    estimate = qwen_ctx4096_estimate(with_runtime_recommendation=True)
    estimate = estimate.model_copy(
        update={
            "repository": estimate.repository.model_copy(update={"repo_id": artifact.repo_id}),
            "weights": estimate.weights.model_copy(
                update={
                    "component": estimate.weights.component.model_copy(
                        update={"bytes": artifact.size_bytes}
                    )
                }
            ),
            "kv_cache": estimate.kv_cache.model_copy(
                update={
                    "layers": 3,
                    "component": estimate.kv_cache.component.model_copy(update={"bytes": 60}),
                }
            ),
            "inference_configuration": estimate.inference_configuration.model_copy(
                update={"device_reserve_bytes": 0}
            ),
        }
    )
    return estimate, context


def test_real_suffix_bytes_and_exact_boundary(tmp_path: Path) -> None:
    estimate, context = local_case(tmp_path)
    # Output + norm, last block (9 separately aligned Q4_K tensors) and 1/3 KV.
    required = LLAMA_CPP_HEADROOM_BYTES + 53_760 + 1024 + 9 * 448 + 20
    estimate = estimate.model_copy(
        update={
            "assessment": estimate.assessment.model_copy(update={"available_vram_bytes": required})
        }
    )
    budget = select_tensor_budget(estimate, context)
    assert isinstance(budget, TensorLayerBudget)
    assert budget.requested_units == 2
    assert budget.gpu_blocks == 1
    assert budget.required_bytes == required
    assert budget.host_input_bytes == 36_864
    less = estimate.model_copy(
        update={
            "assessment": estimate.assessment.model_copy(
                update={"available_vram_bytes": required - 1}
            )
        }
    )
    assert pick_gpu_layers(less, tensor_context=context).n_gpu_layers == 1


def test_full_offload_and_no_gpu_budget(tmp_path: Path) -> None:
    estimate, context = local_case(tmp_path)
    assert pick_gpu_layers(estimate, tensor_context=context).n_gpu_layers == 4
    empty = estimate.model_copy(
        update={"assessment": estimate.assessment.model_copy(update={"available_vram_bytes": 0})}
    )
    assert pick_gpu_layers(empty, tensor_context=context).n_gpu_layers == 0
    cpu = estimate.model_copy(
        update={
            "assessment": estimate.assessment.model_copy(
                update={"effective_device": TargetDevice.CPU}
            )
        }
    )
    assert pick_gpu_layers(cpu, tensor_context=context).n_gpu_layers == 0


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"omit": "output.weight"}, "shared weights"),
        ({"omit": "blk.1.ffn_down.weight"}, "Incomplete"),
        ({"metadata": {"general.architecture": "llama"}}, "only dense qwen2"),
    ],
)
def test_unsupported_layout_retains_fallback(tmp_path: Path, kwargs: dict, reason: str) -> None:
    estimate, context = local_case(tmp_path, **kwargs)
    result = pick_gpu_layers(estimate, tensor_context=context)
    assert result.n_gpu_layers == pick_gpu_layers(estimate).n_gpu_layers
    assert any(reason in warning for warning in result.warnings)


def test_unknown_runtime_does_not_inspect_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tensor_file(tmp_path)

    def forbidden(*args: object) -> None:
        raise AssertionError("unsupported runtime must not inspect tensors")

    monkeypatch.setattr("jaull.runtime.llama_cpp_tensor_policy.read_local_tensor_index", forbidden)
    capability = verified_capability().model_copy(update={"version_text": "version: 1 (unknown)"})
    context = inspect_tensor_context(artifact, qwen_hardware(), cuda_selection(), capability)
    assert context.index is None
    assert context.unavailable_reason == "Runtime build is not verified."


def test_planner_refines_cached_recommendation_but_respects_override(tmp_path: Path) -> None:
    estimate, context = local_case(tmp_path)
    prediction = estimate.model_dump_json()
    refined = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=qwen_hardware(),
        tensor_context=context,
    )
    assert next(f.value for f in refined.flags if f.name == "--n-gpu-layers") == "4"
    override = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=qwen_hardware(),
        tensor_context=context,
        overrides=ExecutionOverrides(n_gpu_layers=0),
    )
    assert next(f.value for f in override.flags if f.name == "--n-gpu-layers") == "0"
    assert estimate.model_dump_json() == prediction


def test_unknown_kv_never_enables_tensor_refinement(tmp_path: Path) -> None:
    estimate, context = local_case(tmp_path)
    estimate = estimate.model_copy(
        update={
            "kv_cache": estimate.kv_cache.model_copy(
                update={"component": estimate.kv_cache.component.model_copy(update={"bytes": None})}
            )
        }
    )
    assert isinstance(select_tensor_budget(estimate, context), str)


@pytest.mark.parametrize(
    "version, supported",
    [
        ("build: 689e227db (10357)", True),
        ("version: 10357 (689e227db)", True),
        ("build: aaaaaaaa (123)", False),
        (None, False),
    ],
)
def test_benchmark_build_must_also_be_verified(
    tmp_path: Path,
    version: str | None,
    supported: bool,
) -> None:
    from jaull.domain.benchmarks import LlamaBenchBinaryStatus, LlamaBenchCapability

    capability = LlamaBenchCapability(
        binary_status=LlamaBenchBinaryStatus.AVAILABLE,
        version_text=version,
    )
    context = inspect_tensor_context(
        tensor_file(tmp_path),
        qwen_hardware(),
        cuda_selection(),
        verified_capability(),
        benchmark_capability=capability,
    )
    assert (context.index is not None) is supported
    if not supported:
        assert context.unavailable_reason == "llama-bench build is not verified."


@pytest.mark.parametrize("failure", ["cpu", "multiple_gpus", "no_devices", "missing_build"])
def test_runtime_scope_failures_preserve_fallback(tmp_path: Path, failure: str) -> None:
    estimate, original = local_case(tmp_path)
    assert original.index is not None
    hardware, selection, capability = qwen_hardware(), cuda_selection(), verified_capability()
    if failure == "cpu":
        selection = selection.model_copy(update={"selected_backend": ComputeBackend.CPU})
    elif failure == "multiple_gpus":
        hardware = hardware.model_copy(update={"gpus": hardware.gpus * 2})
    elif failure == "no_devices":
        capability = capability.model_copy(update={"backend_capabilities": []})
    else:
        capability = capability.model_copy(update={"version_text": None})
    context = inspect_tensor_context(original.index.artifact, hardware, selection, capability)
    assert context.index is None
    assert context.unavailable_reason
    result = pick_gpu_layers(estimate, tensor_context=context)
    assert result.n_gpu_layers == pick_gpu_layers(estimate).n_gpu_layers
    assert result.warnings


def test_unclassified_tensor_is_not_silently_ignored(tmp_path: Path) -> None:
    estimate, context = local_case(tmp_path)
    assert context.index is not None
    tensors = context.index.tensors
    unknown = replace(tensors[-1], name="unclassified.weight")
    context = replace(context, index=replace(context.index, tensors=(*tensors[:-1], unknown)))
    assert "Unclassified" in str(select_tensor_budget(estimate, context))


def test_context_mismatch_does_not_refine_using_stale_kv(tmp_path: Path) -> None:
    estimate, context = local_case(tmp_path)
    result = plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=qwen_hardware(),
        tensor_context=context,
        overrides=ExecutionOverrides(context_size=8192),
    )
    assert (
        result.flags
        == plan_launch(
            runtime=RuntimeName.LLAMA_CPP,
            estimate=estimate,
            hardware=qwen_hardware(),
            overrides=ExecutionOverrides(context_size=8192),
        ).flags
    )
    assert any("context override" in warning for warning in result.warnings)


def test_cli_auto_uses_real_facade_refinement_before_runner(tmp_path: Path) -> None:
    from jaull.advisor.service import AdvisorService
    from jaull.cli.run import RunOptions, run_model
    from tests.test_cli_run import _FakeAdvisor

    estimate, context = local_case(tmp_path)
    assert context.index is not None

    class LocalAdvisor(_FakeAdvisor):
        def estimate_model(self, *args: object, **kwargs: object) -> MemoryEstimate:
            return estimate

        def select_runtime_backend(self, *args: object) -> object:
            return cuda_selection()

        def inspect_llama_cpp_runtime(self, **kwargs: object) -> object:
            return verified_capability()

        def plan_launch(self, **kwargs: object) -> object:
            return AdvisorService.plan_launch(self, **kwargs)  # type: ignore[arg-type]

        def plan_execution(self, **kwargs: object) -> object:
            self.last_plan = AdvisorService.plan_execution(self, **kwargs)  # type: ignore[arg-type]
            return self.last_plan

    advisor = LocalAdvisor(context.index.artifact)
    assert (
        run_model(
            "owner/model",
            RunOptions(prompt="hello", quantization=None),
            advisor=advisor,
        )
        == 0
    )
    runtime = advisor.runs[0][2]
    assert runtime == advisor.last_plan.runtime
    assert runtime is not None
    assert next(f.value for f in runtime.flags if f.name == "--n-gpu-layers") == "4"


def test_explicit_override_skips_local_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jaull.advisor.service import AdvisorService
    from jaull.bootstrap.container import ServiceContainer

    estimate, context = local_case(tmp_path)
    assert context.index is not None

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("explicit override must not inspect local tensors/runtime")

    monkeypatch.setattr(AdvisorService, "inspect_llama_cpp_runtime", forbidden)
    monkeypatch.setattr("jaull.runtime.llama_cpp_tensor_policy.inspect_tensor_context", forbidden)
    advisor = AdvisorService(
        services=ServiceContainer(
            hf_client=object(),
            search_client=object(),
            detect_hardware=qwen_hardware,
            inspect_model=forbidden,
            estimate_memory=forbidden,
        )
    )
    result = advisor.plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=qwen_hardware(),
        local_artifact=context.index.artifact,
        overrides=ExecutionOverrides(n_gpu_layers=0),
    )
    assert next(f.value for f in result.flags if f.name == "--n-gpu-layers") == "0"
