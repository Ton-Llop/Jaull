from __future__ import annotations

from jaull.application.execution import ExecutionOverrides, plan_launch
from jaull.domain.hardware import (
    AcceleratorType,
    AcceleratorVendor,
    ComputeBackend,
)
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
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeName,
    SelectedAcceleratorRef,
)
from jaull.runtime.device_matching import RuntimeDeviceMatchReason, match_runtime_device
from tests._execution_fixtures import qwen_ctx4096_estimate, qwen_hardware

B60 = "Intel(R) Arc(TM) Pro B60 Graphics"
IGPU = "Intel(R) Iris(R) Xe Graphics"


def _device(index: int, name: str | None) -> LlamaCppRuntimeDevice:
    return LlamaCppRuntimeDevice(
        backend=ComputeBackend.VULKAN, runtime_id=f"Vulkan{index}", name=name
    )


def _selection(name: str | None) -> RuntimeBackendSelection:
    accelerator = (
        SelectedAcceleratorRef(
            name=name, vendor=AcceleratorVendor.INTEL, type=AcceleratorType.DEDICATED
        )
        if name is not None
        else None
    )
    return RuntimeBackendSelection(
        selected_backend=ComputeBackend.VULKAN,
        selected_accelerator=accelerator,
        reason=RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE,
    )


def _capability(*devices: LlamaCppRuntimeDevice) -> LlamaCppBackendCapability:
    return LlamaCppBackendCapability(
        backend=ComputeBackend.VULKAN,
        state=LlamaCppBackendCapabilityState.CONFIRMED,
        reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
        devices=list(devices),
    )


def test_single_device_that_is_the_selected_accelerator_is_matched_by_name() -> None:
    match = match_runtime_device(_selection(B60), _capability(_device(0, B60)))

    assert match.device is not None and match.device.runtime_id == "Vulkan0"
    assert match.reason is RuntimeDeviceMatchReason.MATCHED_BY_NAME


def test_selected_accelerator_matching_requires_retained_physical_ids() -> None:
    from jaull.runtime.device_matching import selected_accelerator_matches

    expected = _selection(B60).selected_accelerator.model_copy(update={"uuid": "gpu-a"})
    same = expected.model_copy(update={"detection_sources": ["another probe"]})
    assert selected_accelerator_matches(expected, same)
    assert not selected_accelerator_matches(expected, same.model_copy(update={"uuid": "gpu-b"}))
    assert not selected_accelerator_matches(expected, same.model_copy(update={"uuid": None}))
    assert not selected_accelerator_matches(expected, None)


def test_runtime_numbering_is_not_assumed_to_start_at_the_selected_card() -> None:
    capability = _capability(
        _device(0, "llvmpipe (LLVM 15.0.7, 256 bits)"), _device(1, IGPU), _device(2, B60)
    )

    match = match_runtime_device(_selection(B60), capability)

    assert match.device is not None and match.device.runtime_id == "Vulkan2"


def test_runtime_name_may_carry_a_trailing_annotation() -> None:
    match = match_runtime_device(
        _selection(B60), _capability(_device(0, f"{B60} (BMG G21)"))
    )

    assert match.device is not None


def test_software_renderer_is_never_a_candidate() -> None:
    match = match_runtime_device(
        _selection(None), _capability(_device(0, "llvmpipe (LLVM 15.0.7, 256 bits)"))
    )

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.NO_HARDWARE_DEVICE


def test_a_different_single_device_is_a_mismatch_not_a_fallback() -> None:
    match = match_runtime_device(_selection(B60), _capability(_device(0, IGPU)))

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.NAME_MISMATCH


def test_similar_model_names_do_not_match() -> None:
    selection = _selection("NVIDIA GeForce RTX 2060")
    capability = _capability(_device(0, "NVIDIA GeForce RTX 2060 SUPER"))

    assert match_runtime_device(selection, capability).device is None


def test_two_identical_cards_are_ambiguous() -> None:
    match = match_runtime_device(
        _selection(B60), _capability(_device(0, B60), _device(1, B60))
    )

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.AMBIGUOUS


def test_two_devices_without_a_selected_accelerator_are_ambiguous() -> None:
    match = match_runtime_device(
        _selection(None), _capability(_device(0, B60), _device(1, IGPU))
    )

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.AMBIGUOUS


def test_only_hardware_device_without_a_selected_accelerator_is_not_proof() -> None:
    match = match_runtime_device(_selection(None), _capability(_device(0, B60)))

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.NAME_MISMATCH


def test_unnamed_runtime_device_is_not_proof_of_the_selected_card() -> None:
    match = match_runtime_device(_selection(B60), _capability(_device(0, None)))

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.NAME_MISMATCH


def test_no_devices_means_no_match() -> None:
    match = match_runtime_device(_selection(B60), _capability())

    assert match.device is None
    assert match.reason is RuntimeDeviceMatchReason.NO_HARDWARE_DEVICE


def _readiness(
    selection: RuntimeBackendSelection, capability: LlamaCppBackendCapability
) -> ExecutionReadiness:
    return ExecutionReadiness(
        status=ExecutionReadinessStatus.READY,
        reason=ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED,
        selection=selection,
        runtime_capability=LlamaCppRuntimeCapability(
            binary_path="llama-cli",
            binary_status=LlamaCppBinaryStatus.AVAILABLE,
            backend_capabilities=[capability],
        ),
        selected_backend_capability=capability,
    )


def _launch(selection: RuntimeBackendSelection, capability: LlamaCppBackendCapability):
    return plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=True),
        hardware=qwen_hardware(),
        backend_selection=selection,
        execution_readiness=_readiness(selection, capability),
        overrides=ExecutionOverrides(),
    )


def _device_flag(rec) -> str | None:
    return next((flag.value for flag in rec.flags if flag.name == "--device"), None)


def test_planner_pins_the_matched_device_not_the_first_one() -> None:
    selection = _selection(B60)
    capability = _capability(_device(0, IGPU), _device(1, B60))

    rec = _launch(selection, capability)

    assert _device_flag(rec) == "Vulkan1"
    assert "--device Vulkan1" in (rec.command_preview or "")


def test_planner_does_not_pin_when_the_device_is_ambiguous() -> None:
    selection = _selection(B60)
    capability = _capability(_device(0, B60), _device(1, B60))

    rec = _launch(selection, capability)

    assert _device_flag(rec) is None
    assert "--device" not in (rec.command_preview or "")


def test_planner_does_not_pin_a_software_renderer() -> None:
    selection = _selection(None)
    capability = _capability(_device(0, "llvmpipe (LLVM 15.0.7, 256 bits)"))

    assert _device_flag(_launch(selection, capability)) is None


# ---------------------------------------------------------------------------
# The same rule for every consumer of a runtime device id, not only the planner.
# ---------------------------------------------------------------------------
def _matrix_requests(selection, capability, model):

    from jaull.benchmarks.matrix import BenchmarkMatrixRequest, _build_requests
    from tests.test_benchmarks import _artifact, _hardware, _runtime

    runtime_capability = LlamaCppRuntimeCapability(
        binary_path="llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[capability],
    )
    request = BenchmarkMatrixRequest(
        hardware=_hardware(), artifact=_artifact(model), runtime=_runtime(ngl=-1),
        backend_selection=selection, runtime_capability=runtime_capability, include_cpu=False,
    )
    return _build_requests(request)


def test_benchmark_matrix_uses_the_matched_device(tmp_path) -> None:
    capability = _capability(
        _device(0, "llvmpipe (LLVM 15.0.7, 256 bits)"), _device(1, B60)
    )

    requests, skipped = _matrix_requests(_selection(B60), capability, _model(tmp_path))

    assert [request.device for request in requests] == ["Vulkan1"]
    assert skipped == []


def test_benchmark_matrix_skips_instead_of_guessing_between_identical_cards(tmp_path) -> None:
    capability = _capability(_device(0, B60), _device(1, B60))

    requests, skipped = _matrix_requests(_selection(B60), capability, _model(tmp_path))

    assert requests == []
    assert len(skipped) == 1 and "ambiguous" in skipped[0].reason


def test_tui_benchmark_device_is_the_matched_one() -> None:
    from jaull.tui.screens.recommendation_benchmark import _benchmark_device

    selection = _selection(B60)
    capability = _capability(_device(0, IGPU), _device(1, B60))
    plan = type(
        "Plan",
        (),
        {"execution_readiness": _readiness(selection, capability), "backend_selection": selection},
    )()

    assert _benchmark_device(ComputeBackend.VULKAN, plan) == "Vulkan1"  # type: ignore[arg-type]


def test_tui_benchmark_device_is_unknown_when_ambiguous() -> None:
    from jaull.tui.screens.recommendation_benchmark import _benchmark_device

    selection = _selection(B60)
    capability = _capability(_device(0, B60), _device(1, B60))
    plan = type(
        "Plan",
        (),
        {"execution_readiness": _readiness(selection, capability), "backend_selection": selection},
    )()

    assert _benchmark_device(ComputeBackend.VULKAN, plan) is None  # type: ignore[arg-type]


def test_plan_speed_picks_the_named_device_among_several() -> None:
    from jaull.advisor.service import AdvisorService
    from tests.test_benchmark_plan_speed import _Advisor
    from tests.test_fastest_v2_applicability import _cuda_case

    plan, _ = _cuda_case()
    gpu = plan.backend_selection.selected_accelerator.name

    class Named(_Advisor):
        def inspect_llama_cpp_runtime(self, *, selection):  # type: ignore[no-untyped-def]
            capability = super().inspect_llama_cpp_runtime(selection=selection)
            cuda = capability.backend_capabilities[0]
            named = [
                device.model_copy(update={"name": name})
                for device, name in zip(cuda.devices, ("Another GPU", gpu), strict=True)
            ]
            return capability.model_copy(
                update={"backend_capabilities": [cuda.model_copy(update={"devices": named})]}
            )

    advisor = Named(["CUDA0", "CUDA1"])
    AdvisorService.benchmark_plan_speed(advisor, plan)  # type: ignore[arg-type]

    assert advisor.requests[0].device == "CUDA1"


def _model(tmp_path):  # type: ignore[no-untyped-def]
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    return model


# ---------------------------------------------------------------------------
# Readiness: a Vulkan binary that exposes *a* device is not yet *the* device.
# CUDA and HIP keep the behaviour they were validated with.
# ---------------------------------------------------------------------------
def _runtime_capability(*capabilities: LlamaCppBackendCapability) -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_path="llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=list(capabilities),
    )


def _evaluate(selection: RuntimeBackendSelection, capability: LlamaCppBackendCapability):
    from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness

    return evaluate_execution_readiness(
        selection=selection, runtime_capability=_runtime_capability(capability)
    )


def test_vulkan_binary_exposing_the_selected_card_is_ready() -> None:
    readiness = _evaluate(_selection(B60), _capability(_device(0, B60)))

    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED


def test_vulkan_binary_exposing_only_a_software_renderer_is_not_ready() -> None:
    capability = _capability(_device(0, "llvmpipe (LLVM 15.0.7, 256 bits)"))

    readiness = _evaluate(_selection(B60), capability)

    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_DEVICE_NOT_EXPOSED
    assert "software" in (readiness.message or "")


def test_vulkan_binary_exposing_a_different_card_is_not_ready() -> None:
    readiness = _evaluate(_selection(B60), _capability(_device(0, IGPU)))

    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_DEVICE_NOT_EXPOSED


def test_vulkan_binary_with_two_identical_cards_is_unknown_not_ready_or_guessed() -> None:
    capability = _capability(_device(0, B60), _device(1, B60))

    readiness = _evaluate(_selection(B60), capability)

    assert readiness.status is ExecutionReadinessStatus.UNKNOWN
    assert readiness.reason is ExecutionReadinessReason.SELECTED_DEVICE_NOT_IDENTIFIED


def test_vulkan_selection_without_a_named_accelerator_is_unconfirmed() -> None:
    readiness = _evaluate(_selection(None), _capability(_device(0, B60)))

    assert readiness.status is ExecutionReadinessStatus.NOT_READY


def _cuda_selection(name: str) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=ComputeBackend.CUDA,
        selected_accelerator=SelectedAcceleratorRef(
            name=name, vendor=AcceleratorVendor.NVIDIA, type=AcceleratorType.DEDICATED
        ),
        reason=RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
    )


def _cuda_capability(*names: str) -> LlamaCppBackendCapability:
    return LlamaCppBackendCapability(
        backend=ComputeBackend.CUDA,
        state=LlamaCppBackendCapabilityState.CONFIRMED,
        reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
        devices=[
            LlamaCppRuntimeDevice(
                backend=ComputeBackend.CUDA, runtime_id=f"CUDA{index}", name=name
            )
            for index, name in enumerate(names)
        ],
    )


def test_cuda_keeps_its_validated_behaviour_when_names_differ_slightly() -> None:
    selection = _cuda_selection("NVIDIA GeForce RTX 2060")
    capability = _cuda_capability("NVIDIA GeForce RTX 2060 with Max-Q Design")

    assert _evaluate(selection, capability).status is ExecutionReadinessStatus.READY
    match = match_runtime_device(selection, capability)
    assert match.device is not None and match.device.runtime_id == "CUDA0"


# ---------------------------------------------------------------------------
# The reason for UNKNOWN / NOT_READY must be readable, and must show the facts
# the next person needs: what was selected and what the binary listed.
# ---------------------------------------------------------------------------
def test_ambiguous_message_names_the_selection_and_every_listed_device() -> None:
    capability = _capability(_device(0, B60), _device(1, B60))

    readiness = _evaluate(_selection(B60), capability)

    message = readiness.message or ""
    assert readiness.status is ExecutionReadinessStatus.UNKNOWN
    assert B60 in message
    assert "Vulkan0" in message and "Vulkan1" in message
    assert "no device is selected automatically" in message


def test_mismatch_message_names_the_selection_and_what_was_listed() -> None:
    readiness = _evaluate(_selection(B60), _capability(_device(0, IGPU)))

    message = readiness.message or ""
    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert B60 in message and IGPU in message


def test_software_only_message_says_what_was_listed() -> None:
    capability = _capability(_device(0, "llvmpipe (LLVM 15.0.7, 256 bits)"))

    message = _evaluate(_selection(B60), capability).message or ""

    assert "software" in message and "llvmpipe" in message


def test_unnamed_runtime_devices_are_listed_by_id() -> None:
    capability = _capability(_device(0, None), _device(1, None))

    message = _evaluate(_selection(B60), capability).message or ""

    assert "Vulkan0" in message and "Vulkan1" in message


def test_the_reason_reaches_the_plans_warnings() -> None:
    from types import SimpleNamespace

    from jaull.execution_plans.service import _plan_warnings

    readiness = _evaluate(
        _selection(B60), _capability(_device(0, B60), _device(1, B60))
    )
    artifact = SimpleNamespace(identity_match=None)

    warnings = _plan_warnings(artifact, _launch_runtime(), readiness)  # type: ignore[arg-type]

    assert any("no device is selected automatically" in item for item in warnings)


def _launch_runtime():  # type: ignore[no-untyped-def]
    return plan_launch(
        runtime=RuntimeName.LLAMA_CPP,
        estimate=qwen_ctx4096_estimate(with_runtime_recommendation=True),
        hardware=qwen_hardware(),
        overrides=ExecutionOverrides(),
    )
