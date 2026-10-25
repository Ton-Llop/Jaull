"""benchmark_plan_speed measures exactly the plan: its offload, its device, Fastest's workload."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.benchmarks.errors import BenchmarkConfigurationError
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeFlag,
    RuntimeFlagSource,
)
from tests.test_fastest_v2_applicability import _cuda_case


class _Advisor:
    """Only what benchmark_plan_speed reaches; records the request it builds."""

    def __init__(self, devices: list[str], *, prepared: bool = True) -> None:
        self.devices = devices
        self.prepared = prepared
        self.requests: list[Any] = []
        self.downloads: list[bool] = []

    def _prepare_selected_gguf(self, plan: Any, *, allow_download: bool, report: Any) -> Any:
        self.downloads.append(allow_download)
        if not self.prepared:
            raise ValueError("Selected GGUF is not local. Enable download permission.")
        return plan.artifact.to_model_artifact()

    def inspect_llama_cpp_runtime(self, *, selection: Any) -> LlamaCppRuntimeCapability:
        return LlamaCppRuntimeCapability(
            binary_status=LlamaCppBinaryStatus.AVAILABLE,
            backend_capabilities=[LlamaCppBackendCapability(
                backend=ComputeBackend.CUDA, state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
                devices=[LlamaCppRuntimeDevice(backend=ComputeBackend.CUDA, runtime_id=d)
                         for d in self.devices],
            )],
        )

    def run_benchmark(self, request: Any, *, hardware: Any, backend_selection: Any = None) -> Any:
        self.requests.append(request)
        return SimpleNamespace(record=None, persisted_path=None)


def test_the_request_is_the_plans_own_configuration_with_fastests_workload() -> None:
    plan, _ = _cuda_case()  # Search-style: full offload, no --device flag.
    advisor = _Advisor(["CUDA0"])
    AdvisorService.benchmark_plan_speed(advisor, plan)  # type: ignore[arg-type]
    request, = advisor.requests
    assert request.device == "CUDA0" and request.backend is ComputeBackend.CUDA
    assert request.gpu_layers.full_offload
    assert (request.prefill_sizes, request.generation_sizes, request.repetitions) == (
        (512,), (128,), 5,
    )
    assert advisor.downloads == [False]  # Never fetches without permission.


def test_an_ambiguous_device_is_refused_instead_of_guessed() -> None:
    plan, _ = _cuda_case()
    with pytest.raises(BenchmarkConfigurationError, match="Expected one cuda device"):
        AdvisorService.benchmark_plan_speed(  # type: ignore[arg-type]
            _Advisor(["CUDA0", "CUDA1"]), plan,
        )


def test_a_missing_model_without_permission_runs_nothing() -> None:
    plan, _ = _cuda_case()
    advisor = _Advisor(["CUDA0"], prepared=False)
    with pytest.raises(ValueError, match="not local"):
        AdvisorService.benchmark_plan_speed(advisor, plan)  # type: ignore[arg-type]
    assert advisor.requests == []


def test_an_explicit_speed_device_is_rechecked_before_launch() -> None:
    plan, _ = _cuda_case()
    flag = RuntimeFlag(name="--device", value="CUDA0", source=RuntimeFlagSource.ESTIMATE,
                       explanation="Device selected before renumbering")
    plan = plan.model_copy(update={"runtime": plan.runtime.model_copy(update={
        "flags": [*plan.runtime.flags, flag],
    })})
    advisor = _Advisor(["CUDA1"])
    with pytest.raises(BenchmarkConfigurationError, match="device"):
        AdvisorService.benchmark_plan_speed(advisor, plan)  # type: ignore[arg-type]
    assert not advisor.requests


@pytest.mark.parametrize("renumbered", [False, True])
def test_advisor_checks_the_executed_benchmark_not_the_cli(tmp_path, monkeypatch, renumbered):
    from jaull.domain.benchmarks import BenchmarkGpuLayers, BenchmarkRequest
    from tests._intel_fixtures import B60_NAME, WINDOWS_OS, scan
    from tests._intel_flow import intel_advisor, listing
    from tests.test_llama_cpp_capability import _result

    hw = scan(WINDOWS_OS)
    advisor, scripted = intel_advisor(tmp_path, hw)
    original = scripted.execute

    def execute(request):
        if "--list-devices" in request.command:
            assert "llama-bench" in request.command[0]
            scripted.commands.append(request.command)
            devices = (("Vulkan0", "Integrated GPU"), ("Vulkan1", B60_NAME)) if renumbered else (
                ("Vulkan0", B60_NAME),
            )
            return _result(stdout=listing(devices))
        return original(request)

    monkeypatch.setattr(scripted, "execute", execute)
    from tests.test_benchmarks import _runtime

    request = BenchmarkRequest(
        artifact=advisor.resolve_artifact("org/model"), runtime=_runtime(ngl=-1),
        backend=ComputeBackend.VULKAN, device="Vulkan0", gpu_layers=BenchmarkGpuLayers.full(),
        prefill_sizes=(512,), generation_sizes=(128,),
    )
    if renumbered:
        with pytest.raises(BenchmarkConfigurationError, match="llama-bench"):
            advisor.run_benchmark(request, hardware=hw, persist=False)
    else:
        assert advisor.run_benchmark(request, hardware=hw, persist=False).record.observation.success
    launches = [command for command in scripted.commands if "-dev" in command]
    assert len(launches) == (0 if renumbered else 1)
