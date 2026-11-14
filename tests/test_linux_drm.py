"""GPUs the Linux kernel lists (DRM sysfs), shown even when Vulkan cannot confirm them.

Reported on Reddit (2026-11-14): an AMD Radeon AI PRO R9700 was not shown at all,
only the CPU. Non-NVIDIA GPUs were found only through vulkaninfo, so a missing
tool or a driver too old for the card hid the GPU. Simulated sysfs trees only:
not validated on a physical AMD card under Linux.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    HardwareProfile,
    planning_accelerator_memory_bytes,
)
from jaull.hardware import linux_drm
from jaull.hardware.detector import combine_accelerators, detect_hardware
from jaull.hardware.linux_drm import detect_drm_adapters
from jaull.hardware.vulkan import (
    VULKANINFO_MISSING_WARNING,
    VulkanProbe,
    accelerators_from_summary,
)

GiB = 1024**3
R9700_VRAM = 32 * GiB

PCI_IDS = """\
# fake pci.ids
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t7551  Navi 48 [Radeon AI PRO R9700]
\t\t1002 7551  Radeon AI PRO R9700
8086  Intel Corporation
\t46a6  Alder Lake-P GT2 [Iris Xe Graphics]
C 03  Display controller
\t00  VGA compatible controller
"""


def _card(root: Path, index: int, *, bus: str, vendor: str, device: str,
          pci_class: str = "0x030000", vram: int | None = None,
          driver: str | None = None) -> None:
    pci = root.parent / "pci" / bus
    pci.mkdir(parents=True)
    for name, value in (("class", pci_class), ("vendor", vendor), ("device", device)):
        (pci / name).write_text(value + "\n")
    if vram is not None:
        (pci / "mem_info_vram_total").write_text(f"{vram}\n")
    if driver is not None:
        drivers = root.parent / "drivers" / driver
        drivers.mkdir(parents=True, exist_ok=True)
        (pci / "driver").symlink_to(drivers)
    card = root / f"card{index}"
    card.mkdir(parents=True)
    (card / "device").symlink_to(pci)
    # A connector of the same card: never a device of its own.
    connector = root / f"card{index}-DP-1"
    connector.mkdir()
    (connector / "device").symlink_to(pci)


@pytest.fixture
def sysfs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "drm"
    root.mkdir()
    _card(root, 0, bus="0000:00:02.0", vendor="0x8086", device="0x46a6", driver="i915")
    _card(root, 1, bus="0000:03:00.0", vendor="0x1002", device="0x7551",
          vram=R9700_VRAM, driver="amdgpu")
    # A non-display PCI function exposed under DRM is ignored.
    _card(root, 2, bus="0000:04:00.0", vendor="0x1002", device="0x1234", pci_class="0x040300")
    ids = tmp_path / "pci.ids"
    ids.write_text(PCI_IDS)
    return root, ids


def _r9700(cards: list[AcceleratorProfile]) -> AcceleratorProfile:
    return next(card for card in cards if card.device_id == "0x7551")


def test_kernel_listed_gpus_are_identified_with_name_bus_and_vram(
    sysfs: tuple[Path, Path],
) -> None:
    root, ids = sysfs
    cards = detect_drm_adapters(system="Linux", drm_root=root, pci_ids_paths=(ids,))

    assert [card.device_id for card in cards] == ["0x46a6", "0x7551"]
    card = _r9700(cards)
    assert card.name == "AMD Navi 48 [Radeon AI PRO R9700]"
    assert card.vendor is AcceleratorVendor.AMD and card.vendor_id == "0x1002"
    assert card.pci_bus_id == "0000:03:00.0"
    assert card.dedicated_memory_bytes == R9700_VRAM
    assert card.type is AcceleratorType.UNKNOWN
    assert card.detection_sources == ["drm"]
    (vulkan,) = card.backends
    assert vulkan.availability is BackendAvailability.UNKNOWN
    assert vulkan.driver_name == "amdgpu"
    assert cards[0].name == "Intel Alder Lake-P GT2 [Iris Xe Graphics]"


def test_without_pci_ids_the_name_carries_the_pci_ids(sysfs: tuple[Path, Path],
                                                     tmp_path: Path) -> None:
    root, _ = sysfs
    cards = detect_drm_adapters(system="Linux", drm_root=root,
                                pci_ids_paths=(tmp_path / "missing.ids",))

    assert _r9700(cards).name == "AMD GPU (PCI 1002:7551)"


def test_only_linux_lists_kernel_gpus(sysfs: tuple[Path, Path]) -> None:
    root, ids = sysfs
    assert detect_drm_adapters(system="Windows", drm_root=root, pci_ids_paths=(ids,)) == []
    assert detect_drm_adapters(system="Linux", drm_root=root / "nope") == []


def _cards(sysfs: tuple[Path, Path]) -> list[AcceleratorProfile]:
    root, ids = sysfs
    return [_r9700(detect_drm_adapters(system="Linux", drm_root=root, pci_ids_paths=(ids,)))]


def test_missing_vulkaninfo_keeps_the_gpu_and_says_what_to_install(
    sysfs: tuple[Path, Path],
) -> None:
    probe = VulkanProbe(accelerators=[], warnings=[VULKANINFO_MISSING_WARNING],
                        tool_missing=True)

    scan = combine_accelerators([], probe, [], _cards(sysfs))

    assert [item.name for item in scan.accelerators] == ["AMD Navi 48 [Radeon AI PRO R9700]"]
    (guidance,) = scan.guidance
    assert "vulkaninfo is not installed" in guidance
    assert "sudo apt install vulkan-tools" in guidance


def test_a_driver_that_hides_the_gpu_from_vulkan_is_named(sysfs: tuple[Path, Path]) -> None:
    llvmpipe = accelerators_from_summary(
        "GPU0:\n    vendorID = 0x10005\n    deviceType = PHYSICAL_DEVICE_TYPE_CPU\n"
        "    deviceName = llvmpipe (LLVM 19.1.7, 256 bits)\n    driverName = llvmpipe\n"
    )
    probe = VulkanProbe(accelerators=llvmpipe, warnings=[], ran=True)

    scan = combine_accelerators([], probe, [], _cards(sysfs))

    assert len(scan.accelerators) == 2
    (guidance,) = scan.guidance
    assert "Vulkan does not expose it (it lists only: llvmpipe" in guidance
    assert "update Mesa and the kernel" in guidance


def test_a_gpu_vulkan_confirmed_is_not_listed_twice(sysfs: tuple[Path, Path]) -> None:
    vulkan = accelerators_from_summary(
        "GPU0:\n    vendorID = 0x1002\n    deviceID = 0x7551\n"
        "    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        "    deviceName = AMD Radeon AI PRO R9700 (RADV GFX1201)\n    driverName = radv\n",
        drm_vram_reader=None,
    )
    probe = VulkanProbe(accelerators=vulkan, warnings=[], ran=True)

    scan = combine_accelerators([], probe, [], _cards(sysfs))

    assert [item.name for item in scan.accelerators] == [
        "AMD Radeon AI PRO R9700 (RADV GFX1201)"
    ]
    assert scan.guidance == []


def test_an_nvml_gpu_without_device_id_is_not_listed_twice() -> None:
    nvml = AcceleratorProfile(name="NVIDIA GeForce RTX 2060", vendor=AcceleratorVendor.NVIDIA,
                              vendor_id="0x10de", pci_bus_id="00000000:01:00.0")
    kernel = AcceleratorProfile(name="NVIDIA TU106 [GeForce RTX 2060]",
                                vendor=AcceleratorVendor.NVIDIA, vendor_id="0x10de",
                                device_id="0x1f08", pci_bus_id="0000:01:00.0",
                                detection_sources=["drm"])

    scan = combine_accelerators([nvml], VulkanProbe([], [], ran=True), [], [kernel])

    assert scan.accelerators == [nvml] and scan.guidance == []


def test_a_kernel_only_gpu_never_enters_planning(sysfs: tuple[Path, Path]) -> None:
    profile = HardwareProfile.model_validate({
        "os": "Linux", "os_version": "6.8", "arch": "x86_64",
        "cpu": {}, "memory": {"total_bytes": 64 * GiB, "available_bytes": 48 * GiB},
        "accelerators": [card.model_dump() for card in _cards(sysfs)],
    })

    assert planning_accelerator_memory_bytes(profile) is None


def _missing_vulkaninfo(
    command: tuple[str, ...], timeout: float,
) -> subprocess.CompletedProcess[str]:
    raise FileNotFoundError(command[0])


def test_scan_shows_the_r9700_and_one_actionable_warning(
    sysfs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, ids = sysfs
    monkeypatch.setattr(linux_drm, "DRM_ROOT", root)
    monkeypatch.setattr(linux_drm, "PCI_IDS_PATHS", (ids,))
    monkeypatch.setattr("jaull.hardware.linux_drm.platform.system", lambda: "Linux")

    profile = detect_hardware(vulkan_command_runner=_missing_vulkaninfo)

    names = [item.name for item in profile.accelerators]
    assert "AMD Navi 48 [Radeon AI PRO R9700]" in names
    assert VULKANINFO_MISSING_WARNING not in profile.warnings
    assert sum("vulkan-tools" in warning for warning in profile.warnings) == 2  # R9700 + iGPU


def test_doctor_names_the_kernel_listed_gpu(
    sysfs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaull.diagnostics import service
    from jaull.hardware.nvidia import NvidiaProbe

    monkeypatch.setattr(service, "detect_nvidia_gpus", lambda: NvidiaProbe([], [], [], None, None))
    monkeypatch.setattr(service, "detect_vulkan_accelerators", lambda: VulkanProbe(
        accelerators=[], warnings=[VULKANINFO_MISSING_WARNING], tool_missing=True))
    monkeypatch.setattr(service, "detect_windows_adapters",
                        lambda: type("P", (), {"accelerators": []})())
    monkeypatch.setattr(service, "detect_drm_adapters", lambda: _cards(sysfs))

    result = service._check_accelerators()

    assert "AMD Navi 48 [Radeon AI PRO R9700]" in result.detail
    assert "vulkan: not confirmed (GPU listed by the Linux kernel)" in result.detail
    assert "vulkan-tools" in result.detail
