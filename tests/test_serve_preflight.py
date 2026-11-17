"""Serving probes the exact server and refuses stale or ambiguous device bindings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaull.advisor.service import AdvisorService
from jaull.application.execution import ExecutionPlanningError
from jaull.application.serving import ContextLimit
from jaull.domain.hardware import AcceleratorVendor, ComputeBackend
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.runtime import RuntimeFlag, RuntimeFlagSource
from jaull.execution.errors import ExecutionTimeoutError
from tests.test_cli_serve import _plan
from tests.test_llama_cpp_capability import _FakeExecutionBackend
from tests.test_serve_facade import _Manager


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    binary = tmp_path / "llama-server"
    binary.write_text("fake server", encoding="utf-8")
    binary.chmod(0o755)
    plan = replace(_plan(), server_binary=binary)
    host = _FakeExecutionBackend({
        (str(binary), "--list-devices"): "Vulkan0: Intel Arc Pro B60\n",
    })
    manager = _Manager(tmp_path)
    monkeypatch.setattr("jaull.execution.host.HostExecutionBackend", lambda: host)
    monkeypatch.setattr(AdvisorService, "_server_manager", lambda self: manager)
    advisor = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]
    return advisor, plan, host, manager


@pytest.mark.parametrize("output", [
    "Vulkan0: Intel UHD Graphics\nVulkan1: Intel Arc Pro B60\n",
    "Vulkan0: Intel UHD Graphics\n",
    "Vulkan0: Intel Arc Pro B60\nVulkan1: Intel Arc Pro B60\n",
    "Vulkan0: (6144 MiB, 6000 MiB free)\n",
    "Vulkan0: llvmpipe\n",
    "CUDA0: NVIDIA GeForce RTX 2060\n",
    "Available devices:\n",
])
def test_changed_or_ambiguous_gpu_never_launches(server, output: str) -> None:
    advisor, plan, host, manager = server
    host.outputs[(str(plan.server_binary), "--list-devices")] = output

    with pytest.raises(ExecutionPlanningError, match="No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started
    assert host.requests[0].command == (str(plan.server_binary), "--list-devices")


def test_a_cpu_plan_cannot_silently_start_with_gpu_flags(server) -> None:
    advisor, plan, _, manager = server
    plan = replace(plan, backend=ComputeBackend.CPU, backend_selection=(
        plan.backend_selection.model_copy(update={"selected_backend": ComputeBackend.CPU,
                                                 "selected_accelerator": None})
    ))

    with pytest.raises(ExecutionPlanningError, match="No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started


@pytest.mark.parametrize(("backend", "runtime_id", "vendor", "expected", "observed"), [
    (ComputeBackend.CUDA, "CUDA0", AcceleratorVendor.NVIDIA,
     "NVIDIA GeForce RTX 2060", "NVIDIA GeForce RTX 4070"),
    (ComputeBackend.CUDA, "CUDA0", AcceleratorVendor.NVIDIA,
     "NVIDIA GeForce RTX 2060", "(6144 MiB, 6000 MiB free)"),
    (ComputeBackend.HIP, "HIP0", AcceleratorVendor.AMD,
     "AMD Radeon RX 7900 XTX", "AMD Radeon RX 7800 XT"),
])
def test_a_reused_gpu_index_does_not_prove_physical_identity(
    server, backend, runtime_id, vendor, expected, observed,
) -> None:
    advisor, plan, host, manager = server
    selection = plan.backend_selection.model_copy(update={
        "selected_backend": backend,
        "selected_accelerator": plan.backend_selection.selected_accelerator.model_copy(
            update={"name": expected, "vendor": vendor}),
    })
    plan = replace(plan, backend=backend, backend_selection=selection,
                   launch=plan.launch.model_copy(update={"flags": [
                       flag.model_copy(update={"value": runtime_id})
                       if flag.name == "--device" else flag for flag in plan.launch.flags
                   ]}))
    host.outputs[(str(plan.server_binary), "--list-devices")] = f"{runtime_id}: {observed}\n"

    with pytest.raises(ExecutionPlanningError, match="No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started


def test_missing_server_never_launches(server) -> None:
    advisor, plan, host, manager = server
    plan.server_binary.unlink()

    with pytest.raises(ExecutionPlanningError, match="No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started and not host.requests


def test_server_probe_timeout_never_launches(server) -> None:
    advisor, plan, host, manager = server
    host.outputs[(str(plan.server_binary), "--list-devices")] = ExecutionTimeoutError("timed out")

    with pytest.raises(ExecutionPlanningError, match="No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started


@pytest.mark.parametrize("during_probe", [False, True])
def test_replacing_the_binary_even_with_a_compatible_build_never_launches(
    server, monkeypatch: pytest.MonkeyPatch, during_probe: bool,
) -> None:
    advisor, plan, host, manager = server
    if during_probe:
        execute = host.execute

        def replace_during_probe(request):
            result = execute(request)
            plan.server_binary.write_text("different server", encoding="utf-8")
            return result

        monkeypatch.setattr(host, "execute", replace_during_probe)
    else:
        plan.server_binary.write_text("different server", encoding="utf-8")

    with pytest.raises(ExecutionPlanningError, match=r"changed.*No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started


def test_planning_probes_the_server_and_retains_the_selected_accelerator(
    server, monkeypatch: pytest.MonkeyPatch,
) -> None:
    advisor, original, host, _ = server
    hardware = SimpleNamespace(memory=SimpleNamespace(total_bytes=32 * 1024**3,
                                                     available_bytes=24 * 1024**3))
    inputs = SimpleNamespace(
        hardware=hardware, selection=original.backend_selection, native=None,
        base=InferenceConfiguration(context_length=1),
        estimate=lambda config: original.sizing.estimate,
        quick_estimate=lambda config: original.sizing.estimate,
    )
    monkeypatch.setattr(AdvisorService, "_server_sizing_inputs", lambda self, *args: inputs)
    monkeypatch.setattr("jaull.application.serving.context_limit", lambda **kwargs: ContextLimit(
        users=1, largest_per_user=8192, native=None))
    monkeypatch.setattr("jaull.application.serving.size_server", lambda **kwargs: original.sizing)
    launches = []

    def launch(self, **kwargs):
        launches.append(kwargs)
        return original.launch

    monkeypatch.setattr(AdvisorService, "plan_launch", launch)
    plan = advisor.plan_server(
        repo_id=original.artifact.repo_id, artifact=original.artifact, users=1,
        context_per_user=8192, llama_server_path=original.server_binary,
    )

    assert plan.backend_selection == original.backend_selection
    assert plan.server_sha256 == original.server_sha256
    assert launches[0]["runtime_capability"].binary_path == str(original.server_binary)
    assert launches[0]["execution_readiness"].selection == original.backend_selection
    assert host.requests[0].command == (str(original.server_binary), "--list-devices")


@pytest.mark.parametrize(("backend", "runtime_id", "vendor", "name"), [
    (ComputeBackend.CPU, None, None, None),
    (ComputeBackend.VULKAN, "Vulkan0", AcceleratorVendor.INTEL, "Intel Arc Pro B60"),
    (ComputeBackend.CUDA, "CUDA0", AcceleratorVendor.NVIDIA, "NVIDIA GeForce RTX 2060"),
    (ComputeBackend.HIP, "HIP0", AcceleratorVendor.AMD, "AMD Radeon RX 7900 XTX"),
])
def test_exact_server_is_reprobed_before_launch(server, backend, runtime_id, vendor, name) -> None:
    advisor, plan, host, manager = server
    accelerator = plan.backend_selection.selected_accelerator
    selection = plan.backend_selection.model_copy(update={
        "selected_backend": backend,
        "selected_accelerator": accelerator.model_copy(update={"name": name, "vendor": vendor})
        if runtime_id else None,
    })
    flags = [("--n-gpu-layers", "-1" if runtime_id else "0")]
    if runtime_id:
        flags.append(("--device", runtime_id))
    plan = replace(plan, backend=backend, backend_selection=selection,
                   launch=plan.launch.model_copy(update={"flags": [
                       RuntimeFlag(name=key, value=value, source=RuntimeFlagSource.POLICY,
                                   explanation="test") for key, value in flags
                   ]}))
    host.outputs[(str(plan.server_binary), "--list-devices")] = (
        f"{runtime_id}: {name}\n" if runtime_id else "Available devices:\n"
    )

    advisor.start_server(plan, opencode=False)

    assert manager.started
    assert manager.argv[0] == str(plan.server_binary)
    assert manager.argv[manager.argv.index("--n-gpu-layers") + 1] == (
        "-1" if runtime_id else "0"
    )
    if runtime_id:
        assert manager.argv[manager.argv.index("--device") + 1] == runtime_id
    else:
        assert "--device" not in manager.argv
    assert [request.command for request in host.requests] == [
        (str(plan.server_binary), "--list-devices"), (str(plan.server_binary), "--version"),
    ]


@pytest.mark.parametrize("flags", [[], [("--device", "Vulkan1")], [
    ("--device", "Vulkan0"), ("--device", "Vulkan1"),
]])
def test_a_gpu_server_needs_one_confirmed_explicit_device(server, flags) -> None:
    advisor, plan, _, manager = server
    flags = [("--n-gpu-layers", "-1"), *flags]
    plan = replace(plan, launch=plan.launch.model_copy(update={"flags": [
        RuntimeFlag(name=key, value=value, source=RuntimeFlagSource.POLICY,
                    explanation="test") for key, value in flags
    ]}))

    with pytest.raises(ExecutionPlanningError, match="No model was launched"):
        advisor.start_server(plan, opencode=False)

    assert not manager.started
