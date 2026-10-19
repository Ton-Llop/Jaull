"""Vulkan device memory for non-AMD accelerators (Intel Arc and friends).

All fixtures are synthetic: they follow the layout documented for ``vulkaninfo``
(heap ``size``/``budget``/``usage`` and heap flags) but were NOT captured from a
real Intel Arc Pro B60. They pin Jaull's parsing rules, not driver behaviour.
"""

from __future__ import annotations

import subprocess

from jaull.domain.hardware import AcceleratorType, AcceleratorVendor
from jaull.hardware.vulkan import (
    VulkanHeapMemory,
    detect_vulkan_accelerators,
    parse_vulkan_heap_memory,
)

GIB = 1024**3
MIB = 1024**2

B60_SUMMARY = """
GPU0:
    apiVersion         = 1.4.311
    vendorID           = 0x8086
    deviceID           = 0xe211
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = Intel(R) Arc(TM) Pro B60 Graphics
    driverName         = Intel open-source Mesa driver
"""

INTEL_IGPU_SUMMARY = """
GPU0:
    apiVersion         = 1.3.296
    vendorID           = 0x8086
    deviceID           = 0xa7a0
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = Intel(R) Iris(R) Xe Graphics
    driverName         = Intel open-source Mesa driver
"""

LLVMPIPE_SUMMARY = """
GPU0:
    apiVersion         = 1.3.230
    vendorID           = 0x10005
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe (LLVM 15.0.7, 256 bits)
    driverName         = llvmpipe
"""

# Two device-local heaps, as on a resizable-BAR-less card: a small
# host-visible window and the real VRAM heap.
B60_FULL = """
GPU0:
VkPhysicalDeviceProperties:
---------------------------
\tapiVersion        = 1.4.311
\tvendorID          = 0x8086
\tdeviceID          = 0xe211
\tdeviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName        = Intel(R) Arc(TM) Pro B60 Graphics

VkPhysicalDeviceMemoryProperties:
=================================
\tmemoryHeapCount       = 3
\tmemoryHeaps[0]:
\t\tsize   = 25769803776 (0x600000000) (24.00 GiB)
\t\tbudget = 24696061952 (0x5c0000000) (23.00 GiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags:
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[1]:
\t\tsize   = 268435456 (0x10000000) (256.00 MiB)
\t\tbudget = 268435456 (0x10000000) (256.00 MiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags:
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[2]:
\t\tsize   = 33474412544 (0x7ca3a1000) (31.17 GiB)
\t\tbudget = 30000000000 (0x6fc23ac00) (27.94 GiB)
\t\tusage  = 1048576 (0x00100000) (1.00 MiB)
\t\tflags:
\t\t\tNone
\tmemoryTypeCount       = 4
"""

B60_FULL_NO_BUDGET = B60_FULL.replace(
    "\t\tbudget = 24696061952 (0x5c0000000) (23.00 GiB)\n", ""
).replace("\t\tusage  = 0 (0x00000000) (0.00 B)\n", "", 1)


def _completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("vulkaninfo",), code, stdout, "")


def _runner(summary: str, full: str | Exception, calls: list[tuple[str, ...]] | None = None):
    def run(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        if calls is not None:
            calls.append(command)
        if "--summary" in command:
            return _completed(summary)
        if isinstance(full, Exception):
            raise full
        return _completed(full)

    return run


def _no_drm(vendor: str | None, device: str | None) -> None:
    del vendor, device
    return None


def test_parse_heaps_takes_largest_device_local_heap_with_its_budget() -> None:
    parsed = parse_vulkan_heap_memory(B60_FULL)

    assert len(parsed) == 1
    device = parsed[0]
    assert (device.vendor_id, device.device_id) == ("0x8086", "0xe211")
    assert device.name == "Intel(R) Arc(TM) Pro B60 Graphics"
    assert device.heap == VulkanHeapMemory(
        total_bytes=24 * GIB, budget_bytes=23 * GIB
    )


def test_parse_heaps_without_budget_extension_keeps_capacity_only() -> None:
    device = parse_vulkan_heap_memory(B60_FULL_NO_BUDGET)[0]

    assert device.heap == VulkanHeapMemory(total_bytes=24 * GIB, budget_bytes=None)


def test_parse_heaps_ignores_host_only_heaps() -> None:
    host_only = B60_FULL.replace("MEMORY_HEAP_DEVICE_LOCAL_BIT", "None")

    assert parse_vulkan_heap_memory(host_only)[0].heap is None


def test_parse_heaps_rejects_budget_larger_than_heap() -> None:
    absurd = B60_FULL.replace("24696061952", "99999999999999")

    heap = parse_vulkan_heap_memory(absurd)[0].heap
    assert heap == VulkanHeapMemory(total_bytes=24 * GIB, budget_bytes=None)


def test_parse_heaps_rejects_zero_budget() -> None:
    zero = B60_FULL.replace("24696061952", "0")

    heap = parse_vulkan_heap_memory(zero)[0].heap
    assert heap is not None and heap.budget_bytes is None


def test_parse_heaps_is_empty_for_unrelated_output() -> None:
    assert parse_vulkan_heap_memory("") == []
    assert parse_vulkan_heap_memory("garbage\nGPU0:\n") == []


def test_discrete_intel_gets_capacity_and_budget_from_vulkan_heaps() -> None:
    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, B60_FULL), drm_vram_reader=_no_drm
    )

    [accelerator] = probe.accelerators
    assert accelerator.vendor is AcceleratorVendor.INTEL
    assert accelerator.type is AcceleratorType.DEDICATED
    assert accelerator.shared_memory is False
    assert accelerator.dedicated_memory_bytes == 24 * GIB
    assert accelerator.available_memory_bytes == 23 * GIB
    assert "vulkaninfo" in accelerator.detection_sources
    assert "vulkan-memory-budget" in accelerator.detection_sources


def test_capacity_without_budget_never_becomes_availability() -> None:
    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, B60_FULL_NO_BUDGET), drm_vram_reader=_no_drm
    )

    [accelerator] = probe.accelerators
    assert accelerator.dedicated_memory_bytes == 24 * GIB
    assert accelerator.available_memory_bytes is None
    assert "vulkan-memory-budget" not in accelerator.detection_sources


def test_failed_memory_probe_keeps_device_with_unknown_memory() -> None:
    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, subprocess.TimeoutExpired("vulkaninfo", 1.0)),
        drm_vram_reader=_no_drm,
    )

    [accelerator] = probe.accelerators
    assert accelerator.dedicated_memory_bytes is None
    assert accelerator.available_memory_bytes is None
    assert any("memory" in warning.lower() for warning in probe.warnings)


def test_integrated_and_software_devices_never_start_the_memory_probe() -> None:
    for summary in (INTEL_IGPU_SUMMARY, LLVMPIPE_SUMMARY):
        calls: list[tuple[str, ...]] = []
        probe = detect_vulkan_accelerators(
            _runner(summary, B60_FULL, calls), drm_vram_reader=_no_drm
        )

        assert calls == [("vulkaninfo", "--summary")]
        assert probe.accelerators[0].dedicated_memory_bytes is None


def test_drm_memory_makes_the_extra_probe_unnecessary() -> None:
    from jaull.hardware.vulkan import _DrmVram

    calls: list[tuple[str, ...]] = []
    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, B60_FULL, calls),
        drm_vram_reader=lambda vendor, device: _DrmVram(8 * GIB, 6 * GIB),
    )

    assert calls == [("vulkaninfo", "--summary")]
    assert probe.accelerators[0].dedicated_memory_bytes == 8 * GIB
    assert probe.accelerators[0].available_memory_bytes == 6 * GIB


def test_two_identical_cards_do_not_guess_which_heap_belongs_to_which() -> None:
    summary = B60_SUMMARY + B60_SUMMARY.replace("GPU0:", "GPU1:")
    full = B60_FULL + B60_FULL.replace("GPU0:", "GPU1:")

    probe = detect_vulkan_accelerators(_runner(summary, full), drm_vram_reader=_no_drm)

    assert len(probe.accelerators) == 2
    assert all(item.dedicated_memory_bytes is None for item in probe.accelerators)


def test_summary_device_without_matching_heap_section_stays_unknown() -> None:
    other = B60_FULL.replace("0xe211", "0xbeef")

    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, other), drm_vram_reader=_no_drm
    )

    assert probe.accelerators[0].dedicated_memory_bytes is None


def test_missing_device_type_is_unknown_and_reads_no_memory() -> None:
    summary = B60_SUMMARY.replace(
        "    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n", ""
    )
    calls: list[tuple[str, ...]] = []

    probe = detect_vulkan_accelerators(
        _runner(summary, B60_FULL, calls), drm_vram_reader=_no_drm
    )

    [accelerator] = probe.accelerators
    assert accelerator.type is AcceleratorType.UNKNOWN
    assert accelerator.dedicated_memory_bytes is None
    assert calls == [("vulkaninfo", "--summary")]


def test_nvidia_devices_never_start_the_memory_probe_nvml_owns_their_memory() -> None:
    summary = B60_SUMMARY.replace("0x8086", "0x10de").replace("Intel(R) Arc(TM) Pro B60", "RTX")
    calls: list[tuple[str, ...]] = []

    probe = detect_vulkan_accelerators(
        _runner(summary, B60_FULL, calls), drm_vram_reader=_no_drm
    )

    assert calls == [("vulkaninfo", "--summary")]
    assert probe.accelerators[0].dedicated_memory_bytes is None


def test_budget_is_the_heaps_allowance_not_size_minus_usage() -> None:
    """``heapBudget`` is how much *this process* may allocate; ``heapUsage`` is what
    this process already holds. Neither is global free VRAM, and the budget is not
    reduced by the usage of the process that is merely asking.
    """
    idle = "\t\tusage  = 0 (0x00000000) (0.00 B)\n"
    held = "\t\tusage  = 5368709120 (0x140000000) (5.00 GiB)\n"
    busy = B60_FULL.replace(idle, held, 1)

    heap = parse_vulkan_heap_memory(busy)[0].heap

    assert heap == VulkanHeapMemory(total_bytes=24 * GIB, budget_bytes=23 * GIB)


def test_usage_alone_never_creates_an_availability() -> None:
    no_budget_but_usage = B60_FULL.replace(
        "\t\tbudget = 24696061952 (0x5c0000000) (23.00 GiB)\n", ""
    ).replace("\t\tusage  = 0 (0x00000000) (0.00 B)\n", "\t\tusage  = 1073741824 (1.00 GiB)\n", 1)

    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, no_budget_but_usage), drm_vram_reader=_no_drm
    )

    [accelerator] = probe.accelerators
    assert accelerator.dedicated_memory_bytes == 24 * GIB
    assert accelerator.available_memory_bytes is None


def test_the_budget_source_is_labelled_so_it_is_not_read_as_free_memory() -> None:
    probe = detect_vulkan_accelerators(
        _runner(B60_SUMMARY, B60_FULL), drm_vram_reader=_no_drm
    )

    assert "vulkan-memory-budget" in probe.accelerators[0].detection_sources
    assert any("budget" in warning.lower() and "not a free-vram" in warning.lower()
               for warning in probe.warnings)
