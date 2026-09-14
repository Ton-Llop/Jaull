"""Parsing llama.cpp's own buffer report into a runtime observation."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaull.domain.execution import RuntimeBufferCategory, RuntimeBufferLocation
from jaull.runtime.llama_cpp_memory_report import parse_llama_cpp_allocation

# Verbatim from validation/qwen2.5-7b-q4km-2060-ctx4096-sweep/llama-ngl18.log.
SWEEP_NGL18 = """
0.00.487.252 I load_tensors: offloaded 18/29 layers to GPU
0.00.487.259 I load_tensors:   CPU_Mapped model buffer size =  2198.74 MiB
0.00.487.260 I load_tensors:        CUDA0 model buffer size =  2706.70 MiB
0.00.500.000 I llama_context:        CUDA0 KV buffer size =    136.00 MiB
0.00.500.001 I llama_context:          CPU KV buffer size =     88.00 MiB
0.00.500.002 I llama_context:      CUDA0 compute buffer size =   183.44 MiB
0.00.500.003 I llama_context:  CUDA_Host compute buffer size =    18.01 MiB
0.00.500.004 I llama_context:   CUDA_Host output buffer size =     0.58 MiB
"""

MiB = 1024 * 1024


def test_host_pinned_buffers_are_not_device_memory() -> None:
    """``CUDA_Host`` is pinned host memory despite the prefix.

    A substring match on "CUDA" would add 18.01 MiB of pinned compute buffer
    and 0.58 MiB of output buffer to the device total.
    """
    allocation = parse_llama_cpp_allocation(SWEEP_NGL18)

    assert allocation is not None
    host = {
        buffer.raw_label
        for buffer in allocation.buffers
        if buffer.location is RuntimeBufferLocation.HOST
    }
    assert host == {
        "CPU_Mapped model",
        "CPU kv",
        "CUDA_Host compute",
        "CUDA_Host output",
    }


def test_total_device_bytes_is_the_sum_of_the_device_buffers() -> None:
    """2706.70 + 136.00 + 183.44 MiB, and nothing inferred on top."""
    allocation = parse_llama_cpp_allocation(SWEEP_NGL18)

    assert allocation is not None
    expected = round((2706.70 + 136.00 + 183.44) * MiB)
    assert allocation.total_device_bytes == pytest.approx(expected, abs=2)
    assert allocation.device == "CUDA0"
    assert allocation.runtime == "llama.cpp"


def test_categories_are_normalised_but_the_raw_label_survives() -> None:
    allocation = parse_llama_cpp_allocation(SWEEP_NGL18)

    assert allocation is not None
    by_category = {
        (buffer.location, buffer.category): buffer
        for buffer in allocation.buffers
    }
    model = by_category[(RuntimeBufferLocation.DEVICE, RuntimeBufferCategory.MODEL)]
    assert model.raw_label == "CUDA0 model"
    assert model.bytes == round(2706.70 * MiB)


def test_device_bytes_for_reports_absence_rather_than_zero() -> None:
    """A category the runtime never printed is unknown, not empty."""
    allocation = parse_llama_cpp_allocation(
        "load_tensors:  CUDA0 model buffer size =  100.00 MiB"
    )

    assert allocation is not None
    assert allocation.device_bytes_for(RuntimeBufferCategory.MODEL) == round(100 * MiB)
    assert allocation.device_bytes_for(RuntimeBufferCategory.KV) is None


def test_a_cpu_only_run_reports_no_device() -> None:
    allocation = parse_llama_cpp_allocation(
        "load_tensors:  CPU_Mapped model buffer size =  4460.45 MiB"
    )

    assert allocation is not None
    assert allocation.device is None
    assert allocation.total_device_bytes == 0


def test_output_without_buffer_lines_yields_no_observation() -> None:
    assert parse_llama_cpp_allocation("hello\nworld\n") is None


def test_the_recorded_sweep_logs_parse(tmp_path: Path) -> None:
    """The real logs in validation/ are the contract this parser serves."""
    del tmp_path
    sweep = Path("validation/qwen2.5-7b-q4km-2060-ctx4096-sweep")
    log = sweep / "llama-ngl18.log"
    if not log.is_file():  # pragma: no cover - evidence not checked out
        pytest.skip("validation evidence not present")

    allocation = parse_llama_cpp_allocation(log.read_text(errors="replace"))

    assert allocation is not None
    assert allocation.device == "CUDA0"
    # The figures the August run recorded, to within rounding.
    assert allocation.device_bytes_for(RuntimeBufferCategory.MODEL) == round(
        2706.70 * MiB
    )
    assert allocation.total_device_bytes == pytest.approx(
        round((2706.70 + 136.00 + 183.44) * MiB), abs=2
    )
