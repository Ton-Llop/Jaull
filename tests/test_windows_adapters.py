"""Windows adapter identification: what Windows lists, never what the GPU can run.

The JSON mirrors ``Get-CimInstance Win32_VideoController | ConvertTo-Json``;
these fixtures are synthetic, not captured from an Intel Arc Pro B60.
"""

from __future__ import annotations

import json
import subprocess

from jaull.domain.hardware import (
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    BackendAvailabilityReason,
    ComputeBackend,
)
from jaull.hardware.detector import _merge_accelerators
from jaull.hardware.vulkan import accelerators_from_summary
from jaull.hardware.windows_adapters import (
    detect_windows_adapters,
    parse_windows_video_controllers,
)

B60_NAME = "Intel(R) Arc(TM) Pro B60 Graphics"

B60_ROW = {
    "Name": B60_NAME,
    "PNPDeviceID": "PCI\\VEN_8086&DEV_E211&SUBSYS_12345678&REV_00\\4&1234&0&0008",
    "DriverVersion": "32.0.101.6881",
    "AdapterCompatibility": "Intel Corporation",
    # Win32_VideoController.AdapterRAM is a 32-bit field and is deliberately unused.
    "AdapterRAM": 4293918720,
}

BASIC_RENDER = {
    "Name": "Microsoft Basic Render Driver",
    "PNPDeviceID": "ROOT\\BASICRENDER\\0000",
    "DriverVersion": "10.0.26100.1",
    "AdapterCompatibility": "Microsoft",
}

VULKAN_SUMMARY = """
GPU0:
    apiVersion         = 1.4.311
    vendorID           = 0x8086
    deviceID           = 0xe211
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = Intel(R) Arc(TM) Pro B60 Graphics
    driverName         = Intel proprietary Windows driver
"""


def test_single_adapter_json_object_is_accepted() -> None:
    [adapter] = parse_windows_video_controllers(json.dumps(B60_ROW))

    assert adapter.name == B60_NAME
    assert adapter.vendor is AcceleratorVendor.INTEL
    assert (adapter.vendor_id, adapter.device_id) == ("0x8086", "0xe211")
    assert adapter.detection_sources == ["windows-wmi"]


def test_several_adapters_json_array_is_accepted() -> None:
    adapters = parse_windows_video_controllers(json.dumps([B60_ROW, BASIC_RENDER]))

    assert [adapter.name for adapter in adapters] == [B60_NAME]


def test_windows_listing_never_claims_a_backend_or_a_memory_figure() -> None:
    [adapter] = parse_windows_video_controllers(json.dumps(B60_ROW))

    assert adapter.type is AcceleratorType.UNKNOWN
    assert adapter.dedicated_memory_bytes is None
    assert adapter.available_memory_bytes is None
    vulkan = next(item for item in adapter.backends if item.backend is ComputeBackend.VULKAN)
    assert vulkan.availability is BackendAvailability.UNKNOWN
    assert vulkan.reason is BackendAvailabilityReason.NOT_CHECKED
    assert vulkan.driver_version == "32.0.101.6881"
    assert not any(item.availability is BackendAvailability.AVAILABLE for item in adapter.backends)


def test_virtual_and_software_adapters_are_not_accelerators() -> None:
    remote = {**BASIC_RENDER, "Name": "Microsoft Remote Display Adapter"}

    assert parse_windows_video_controllers(json.dumps([BASIC_RENDER, remote])) == []


def test_rows_without_a_name_or_with_bad_ids_degrade_to_partial_information() -> None:
    partial = {"Name": "Some GPU", "PNPDeviceID": None}
    nameless = {"PNPDeviceID": B60_ROW["PNPDeviceID"]}

    [adapter] = parse_windows_video_controllers(json.dumps([partial, nameless]))

    assert adapter.name == "Some GPU"
    assert adapter.vendor is AcceleratorVendor.UNKNOWN
    assert adapter.vendor_id is None and adapter.device_id is None


def test_garbage_and_empty_output_yield_nothing() -> None:
    for text in ("", "   ", "not json", "[]", "null", '"text"', "42"):
        assert parse_windows_video_controllers(text) == []


def test_probe_failures_keep_the_profile_usable() -> None:
    def missing(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("powershell")

    def timed_out(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout)

    def failed(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "boom")

    for runner in (missing, timed_out, failed):
        probe = detect_windows_adapters(runner, system="Windows")
        assert probe.accelerators == []
        assert probe.warnings


def test_probe_is_skipped_outside_windows() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(B60_ROW), "")

    for system in ("Linux", "Darwin"):
        probe = detect_windows_adapters(runner, system=system)
        assert probe.accelerators == [] and probe.warnings == []
    assert calls == []


def test_probe_runs_without_a_shell_and_parses_its_output() -> None:
    seen: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(B60_ROW), "")

    probe = detect_windows_adapters(runner, system="Windows")

    assert [item.name for item in probe.accelerators] == [B60_NAME]
    assert seen and seen[0][0].lower().startswith("powershell")


def test_vulkan_detection_and_windows_listing_merge_into_one_device() -> None:
    vulkan = accelerators_from_summary(VULKAN_SUMMARY)
    windows = parse_windows_video_controllers(json.dumps(B60_ROW))

    [merged] = _merge_accelerators(vulkan, windows)

    assert merged.type is AcceleratorType.DEDICATED
    assert "vulkaninfo" in merged.detection_sources and "windows-wmi" in merged.detection_sources
    vulkan_info = next(item for item in merged.backends if item.backend is ComputeBackend.VULKAN)
    assert vulkan_info.availability is BackendAvailability.AVAILABLE


def test_a_slightly_different_name_still_merges_when_the_pci_ids_identify_one_device() -> None:
    vulkan = accelerators_from_summary(VULKAN_SUMMARY)
    renamed = {**B60_ROW, "Name": "Intel(R) Arc(TM) Pro B60"}
    windows = parse_windows_video_controllers(json.dumps(renamed))

    assert len(_merge_accelerators(vulkan, windows)) == 1


def test_a_windows_only_adapter_is_kept_but_selects_no_backend() -> None:
    windows = parse_windows_video_controllers(json.dumps(B60_ROW))

    merged = _merge_accelerators([], windows)

    assert len(merged) == 1
    assert not any(
        backend.availability is BackendAvailability.AVAILABLE
        for accelerator in merged
        for backend in accelerator.backends
    )


# ---------------------------------------------------------------------------
# End to end through detect_hardware: an Intel GPU never implies Vulkan.
# ---------------------------------------------------------------------------
def _no_vulkaninfo(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
    raise FileNotFoundError("vulkaninfo")


def _windows_lists_b60(
    command: tuple[str, ...], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, json.dumps(B60_ROW), "")


class _NoNvml:
    def nvmlInit(self) -> None:
        raise OSError("no NVIDIA driver")


def test_intel_gpu_listed_by_windows_without_vulkaninfo_selects_no_accelerated_backend(
    monkeypatch,
) -> None:
    from jaull.hardware import windows_adapters
    from jaull.hardware.detector import detect_hardware
    from jaull.runtime.backend_selection import select_runtime_backend

    monkeypatch.setattr(windows_adapters.platform, "system", lambda: "Windows")

    profile = detect_hardware(
        _NoNvml(),
        vulkan_command_runner=_no_vulkaninfo,
        windows_command_runner=_windows_lists_b60,
    )

    assert [item.name for item in profile.accelerators] == [B60_NAME]
    assert any("vulkaninfo" in warning for warning in profile.warnings)
    selection = select_runtime_backend(profile)
    assert selection.selected_backend is ComputeBackend.CPU


def test_vulkaninfo_and_windows_listing_agree_into_one_selectable_device(monkeypatch) -> None:
    from jaull.hardware import windows_adapters
    from jaull.hardware.detector import detect_hardware
    from jaull.runtime.backend_selection import select_runtime_backend

    monkeypatch.setattr(windows_adapters.platform, "system", lambda: "Windows")

    def vulkaninfo(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, VULKAN_SUMMARY, "")

    profile = detect_hardware(
        _NoNvml(),
        vulkan_command_runner=vulkaninfo,
        windows_command_runner=_windows_lists_b60,
    )

    assert len(profile.accelerators) == 1
    selection = select_runtime_backend(profile)
    assert selection.selected_backend is ComputeBackend.VULKAN
