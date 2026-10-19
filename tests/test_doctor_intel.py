"""`jaull doctor` on an Intel machine: finding the GPU is not the same as running on it."""

from __future__ import annotations

import json

import pytest

from jaull.diagnostics import service
from jaull.domain.enums import DiagnosticStatus
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    SelectedAcceleratorRef,
)
from jaull.hardware.nvidia import NvidiaProbe
from jaull.hardware.vulkan import VulkanProbe, accelerators_from_summary
from jaull.hardware.windows_adapters import (
    WindowsAdapterProbe,
    parse_windows_video_controllers,
)
from jaull.runtime.backend_selection import select_runtime_backend
from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness
from tests._intel_fixtures import (
    B60_NAME,
    B60_SUMMARY,
    B60_WMI_ROW,
    IGPU_NAME,
    LLVMPIPE_SUMMARY,
    WINDOWS_OS,
    WSL2_OS,
)


def _probes(monkeypatch: pytest.MonkeyPatch, *, vulkan: str, windows: bool, os_label: str) -> None:
    monkeypatch.setattr(
        service, "detect_nvidia_gpus", lambda: NvidiaProbe([], [], [], None, None)
    )
    monkeypatch.setattr(
        service,
        "detect_vulkan_accelerators",
        lambda: VulkanProbe(accelerators=accelerators_from_summary(vulkan), warnings=[]),
    )
    listing = parse_windows_video_controllers(json.dumps(B60_WMI_ROW)) if windows else []
    monkeypatch.setattr(
        service,
        "detect_windows_adapters",
        lambda: WindowsAdapterProbe(accelerators=listing, warnings=[]),
    )
    monkeypatch.setattr(service, "_os_label", lambda: os_label)


def test_windows_listing_and_vulkaninfo_are_one_device_not_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probes(monkeypatch, vulkan=B60_SUMMARY, windows=True, os_label=WINDOWS_OS)

    result = service._check_accelerators()

    assert result.status is DiagnosticStatus.OK
    assert result.detail.count(B60_NAME) == 1
    assert "vulkan: available" in result.detail


def test_an_adapter_only_windows_listed_is_detected_but_vulkan_is_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probes(monkeypatch, vulkan="", windows=True, os_label=WINDOWS_OS)

    result = service._check_accelerators()

    assert result.status is DiagnosticStatus.WARN
    assert B60_NAME in result.detail
    assert "vulkan: not confirmed (adapter listed by Windows)" in result.detail


def test_wsl2_without_a_vulkan_device_keeps_both_the_detail_and_the_caution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probes(monkeypatch, vulkan=LLVMPIPE_SUMMARY, windows=False, os_label=WSL2_OS)

    result = service._check_accelerators()

    assert "vulkan: software renderer only" in result.detail
    assert "Running inside WSL2" in result.detail


def test_the_runtime_profile_used_for_backend_selection_includes_the_windows_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probes(monkeypatch, vulkan=B60_SUMMARY, windows=True, os_label=WINDOWS_OS)

    profile = service._runtime_hardware_profile()

    assert len(profile.accelerators) == 1
    assert select_runtime_backend(profile).selected_backend is ComputeBackend.VULKAN


def _selection() -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=ComputeBackend.VULKAN,
        selected_accelerator=SelectedAcceleratorRef(
            name=B60_NAME,
            vendor=__import__("jaull.domain.hardware", fromlist=["x"]).AcceleratorVendor.INTEL,
            type=__import__("jaull.domain.hardware", fromlist=["x"]).AcceleratorType.DEDICATED,
        ),
        reason=RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE,
    )


def _capability(*names: str) -> LlamaCppRuntimeCapability:
    vulkan = LlamaCppBackendCapability(
        backend=ComputeBackend.VULKAN,
        state=LlamaCppBackendCapabilityState.CONFIRMED,
        reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
        devices=[
            LlamaCppRuntimeDevice(
                backend=ComputeBackend.VULKAN, runtime_id=f"Vulkan{index}", name=name
            )
            for index, name in enumerate(names)
        ],
    )
    return LlamaCppRuntimeCapability(
        binary_path="llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[vulkan],
    )


def test_execution_ready_is_a_separate_question_and_explains_a_mismatch() -> None:
    readiness = evaluate_execution_readiness(
        selection=_selection(), runtime_capability=_capability(IGPU_NAME)
    )

    result = service._check_execution_readiness(readiness)

    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert result.name == "Execution ready"
    assert result.status is DiagnosticStatus.WARN
    assert B60_NAME in result.detail and IGPU_NAME in result.detail


def test_execution_ready_explains_an_unidentified_device() -> None:
    readiness = evaluate_execution_readiness(
        selection=_selection(), runtime_capability=_capability(B60_NAME, B60_NAME)
    )

    result = service._check_execution_readiness(readiness)

    assert readiness.status is ExecutionReadinessStatus.UNKNOWN
    assert result.status is DiagnosticStatus.WARN
    assert "no device is selected automatically" in result.detail


def test_execution_ready_is_ok_only_when_the_selected_card_is_listed() -> None:
    readiness = evaluate_execution_readiness(
        selection=_selection(), runtime_capability=_capability(B60_NAME)
    )

    result = service._check_execution_readiness(readiness)

    assert result.status is DiagnosticStatus.OK
