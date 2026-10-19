"""Vulkan observations keep their provenance; Windows-style runtime paths survive.

Log lines are synthetic, shaped like llama.cpp's own report. Nothing here was
produced by a real Intel Arc Pro B60.
"""

from __future__ import annotations

from pathlib import Path

from jaull.domain.execution import RuntimeBufferCategory, RuntimeBufferLocation
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import LlamaCppBackendCapabilityState, LlamaCppBinaryStatus
from jaull.runtime.llama_cpp_capability import inspect_llama_cpp_runtime, parse_llama_cpp_devices
from jaull.runtime.llama_cpp_memory_report import parse_llama_cpp_allocation
from tests.test_llama_cpp_capability import _FakeExecutionBackend

VULKAN_LOAD_LOG = """
load_tensors:      Vulkan0 model buffer size =  4096.00 MiB
load_tensors:  Vulkan_Host model buffer size =   128.00 MiB
load_tensors:   CPU_Mapped model buffer size =   300.00 MiB
llama_context:      Vulkan0 KV buffer size =   512.00 MiB
llama_context:      Vulkan0 compute buffer size =   200.00 MiB
llama_context:  Vulkan_Host output buffer size =     0.50 MiB
"""


def test_vulkan_buffers_are_the_runtimes_report_not_a_measured_peak() -> None:
    allocation = parse_llama_cpp_allocation(VULKAN_LOAD_LOG)

    assert allocation is not None
    assert allocation.device == "Vulkan0"
    # Only the device label counts as device memory; the pinned-host buffer and
    # the mapped CPU buffer must never inflate it.
    mib = 1024 * 1024
    assert allocation.total_device_bytes == (4096 + 512 + 200) * mib
    host = [b for b in allocation.buffers if b.location is RuntimeBufferLocation.HOST]
    assert {b.raw_label.split()[0] for b in host} == {"Vulkan_Host", "CPU_Mapped"}
    assert allocation.device_bytes_for(RuntimeBufferCategory.KV) == 512 * mib


def test_a_buffer_total_has_no_peak_field_to_be_mistaken_for_one() -> None:
    allocation = parse_llama_cpp_allocation(VULKAN_LOAD_LOG)

    assert allocation is not None
    assert not hasattr(allocation, "peak_device_bytes")
    assert not hasattr(allocation, "peak_vram_bytes")


def test_two_vulkan_devices_in_one_report_are_not_attributed_to_either() -> None:
    log = VULKAN_LOAD_LOG + "load_tensors:      Vulkan1 model buffer size =  1024.00 MiB\n"

    allocation = parse_llama_cpp_allocation(log)

    assert allocation is not None and allocation.device is None


def test_device_list_survives_windows_line_endings() -> None:
    output = (
        "ggml_vulkan: Found 1 Vulkan devices:\r\n"
        "Available devices:\r\n"
        "  Vulkan0: Intel(R) Arc(TM) Pro B60 Graphics (24576 MiB, 23000 MiB free)\r\n"
    )

    [device] = parse_llama_cpp_devices(output)

    assert device.runtime_id == "Vulkan0"
    assert device.name == "Intel(R) Arc(TM) Pro B60 Graphics"
    assert device.memory_total_bytes == 24576 * 1024 * 1024


def test_probe_passes_a_path_with_spaces_as_one_argument(tmp_path: Path) -> None:
    folder = tmp_path / "Program Files" / "llama.cpp vulkan"
    folder.mkdir(parents=True)
    binary = folder / "llama-cli.exe"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o755)
    listing = (
        "Available devices:\n"
        "  Vulkan0: Intel(R) Arc(TM) Pro B60 Graphics (24576 MiB, 1 MiB free)\n"
    )
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): listing,
            (str(binary), "--version"): "version: 1 (abc)",
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    assert capability.binary_status is LlamaCppBinaryStatus.AVAILABLE
    assert [request.command[0] for request in backend.requests] == [str(binary)] * 2
    assert all(" " in request.command[0] for request in backend.requests)
    vulkan = next(
        item for item in capability.backend_capabilities if item.backend is ComputeBackend.VULKAN
    )
    assert vulkan.state is LlamaCppBackendCapabilityState.CONFIRMED


def test_probe_that_lists_no_devices_confirms_nothing_for_vulkan(tmp_path: Path) -> None:
    binary = tmp_path / "llama-cli"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o755)
    backend = _FakeExecutionBackend(
        {(str(binary), "--list-devices"): "Available devices:\n"}
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    vulkan = next(
        item for item in capability.backend_capabilities if item.backend is ComputeBackend.VULKAN
    )
    assert vulkan.state is LlamaCppBackendCapabilityState.NOT_OBSERVED
