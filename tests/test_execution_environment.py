from __future__ import annotations

import pytest

from jaull.domain.execution_environment import (
    ExecutionEnvironment,
    acceleration_environment_note,
    classify_execution_environment,
)
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
)


def _hardware(
    os_name: str, accelerators: list[AcceleratorProfile] | None = None
) -> HardwareProfile:
    return HardwareProfile(
        os=os_name,
        arch="x86_64",
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=1, available_bytes=1),
        accelerators=accelerators or [],
    )


def _vulkan(*, software: bool, name: str = "device") -> AcceleratorProfile:
    return AcceleratorProfile(
        name=name,
        vendor=AcceleratorVendor.UNKNOWN if software else AcceleratorVendor.INTEL,
        type=AcceleratorType.SOFTWARE if software else AcceleratorType.DEDICATED,
        backends=[
            ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=(
                    BackendAvailability.UNAVAILABLE if software else BackendAvailability.AVAILABLE
                ),
                software_renderer=software,
            )
        ],
    )


@pytest.mark.parametrize(
    ("os_name", "expected"),
    [
        ("Windows 11", ExecutionEnvironment.WINDOWS_NATIVE),
        ("Windows 10", ExecutionEnvironment.WINDOWS_NATIVE),
        ("Linux 5.15.153.1-microsoft-standard-WSL2", ExecutionEnvironment.WSL2),
        ("Linux 6.6.87.2-microsoft-standard-WSL2", ExecutionEnvironment.WSL2),
        ("Linux 4.4.0-19041-Microsoft", ExecutionEnvironment.WSL1),
        ("Linux 6.8.0-45-generic", ExecutionEnvironment.LINUX_NATIVE),
        ("Darwin 23.4.0", ExecutionEnvironment.MACOS),
        ("", ExecutionEnvironment.UNKNOWN),
        ("unknown", ExecutionEnvironment.UNKNOWN),
        ("FreeBSD 14.0", ExecutionEnvironment.UNKNOWN),
    ],
)
def test_environment_is_read_from_the_profile_not_the_gpu(
    os_name: str, expected: ExecutionEnvironment
) -> None:
    assert classify_execution_environment(_hardware(os_name)) is expected


def test_a_wsl_kernel_name_is_not_decided_by_case() -> None:
    hardware = _hardware("Linux 5.15.0-MICROSOFT-STANDARD-wsl2")

    assert classify_execution_environment(hardware) is ExecutionEnvironment.WSL2


def test_wsl2_with_only_a_software_renderer_explains_without_blaming_the_host() -> None:
    hardware = _hardware(
        "Linux 5.15.153.1-microsoft-standard-WSL2",
        [_vulkan(software=True, name="llvmpipe")],
    )

    note = acceleration_environment_note(hardware)

    assert note is not None
    assert "WSL2" in note
    assert "does not show" in note and "Windows" in note


def test_wsl2_with_no_vulkan_device_at_all_gets_the_same_caution() -> None:
    hardware = _hardware("Linux 5.15.153.1-microsoft-standard-WSL2")

    assert acceleration_environment_note(hardware) is not None


def test_wsl2_with_a_real_vulkan_device_needs_no_note() -> None:
    hardware = _hardware(
        "Linux 5.15.153.1-microsoft-standard-WSL2", [_vulkan(software=False)]
    )

    assert acceleration_environment_note(hardware) is None


@pytest.mark.parametrize("os_name", ["Windows 11", "Linux 6.8.0-45-generic", "Darwin 23.4.0"])
def test_other_environments_never_get_the_wsl_note(os_name: str) -> None:
    hardware = _hardware(os_name, [_vulkan(software=True, name="llvmpipe")])

    assert acceleration_environment_note(hardware) is None
