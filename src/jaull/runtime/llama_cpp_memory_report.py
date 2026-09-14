"""Read llama.cpp's own buffer report out of its output.

llama.cpp prints every buffer it allocates while loading a model:

    load_tensors:   CPU_Mapped model buffer size =  2198.74 MiB
    load_tensors:        CUDA0 model buffer size =  2706.70 MiB
    llama_context:      CUDA0 compute buffer size =   183.44 MiB

That is a second observation of device memory, independent of NVML, and the
only one available on a GPU in WDDM mode. It is the runtime's own account of
what it asked for, which is not the same thing as what the driver committed —
callers must keep that distinction, see ``RuntimeReportedAllocation``.

The domain never sees these strings. This module turns them into a normalised
observation and nothing else: no totals beyond summing what was reported, no
inference about buffers llama.cpp did not print.
"""

from __future__ import annotations

import re

from jaull.domain.execution import (
    RuntimeBuffer,
    RuntimeBufferCategory,
    RuntimeBufferLocation,
    RuntimeReportedAllocation,
)
from jaull.domain.runtime import RuntimeName

_MIB = 1024 * 1024

_BUFFER = re.compile(
    r"^(?P<what>.+?)\s+(?P<kind>model|KV|compute|output)\s+buffer size\s*=\s*"
    r"(?P<mib>[\d.]+)\s*MiB",
    re.IGNORECASE,
)

# ``CUDA0`` is device memory; ``CUDA_Host`` is pinned *host* memory despite the
# prefix, and a substring match on "CUDA" would silently inflate the device
# total by every pinned buffer. Match the device suffix exactly.
_DEVICE_LABEL = re.compile(r"^(?:CUDA|ROCm|MUSA|SYCL|Vulkan)\d+$")

_CATEGORIES = {
    "model": RuntimeBufferCategory.MODEL,
    "kv": RuntimeBufferCategory.KV,
    "compute": RuntimeBufferCategory.COMPUTE,
    "output": RuntimeBufferCategory.OUTPUT,
}


def parse_llama_cpp_allocation(
    output: str,
    *,
    runtime_build: str | None = None,
) -> RuntimeReportedAllocation | None:
    """Build an allocation observation, or ``None`` when nothing was reported.

    A buffer label repeats across load phases; the largest value for a label
    wins, matching how the standalone validation harness has read these lines
    since August.
    """

    largest: dict[tuple[str, RuntimeBufferCategory], float] = {}
    for line in output.splitlines():
        match = _BUFFER.search(line)
        if match is None:
            continue
        label = match.group("what").split()[-1]
        category = _CATEGORIES.get(match.group("kind").lower())
        if category is None:
            continue
        key = (label, category)
        largest[key] = max(largest.get(key, 0.0), float(match.group("mib")))

    if not largest:
        return None

    buffers = tuple(
        RuntimeBuffer(
            category=category,
            location=_location(label),
            bytes=round(mib * _MIB),
            raw_label=f"{label} {category.value}",
        )
        for (label, category), mib in sorted(largest.items(), key=lambda item: item[0])
    )
    return RuntimeReportedAllocation(
        runtime=RuntimeName.LLAMA_CPP.value,
        device=_device(largest),
        runtime_build=runtime_build,
        buffers=buffers,
    )


def _location(label: str) -> RuntimeBufferLocation:
    if _DEVICE_LABEL.match(label):
        return RuntimeBufferLocation.DEVICE
    return RuntimeBufferLocation.HOST


def _device(largest: dict[tuple[str, RuntimeBufferCategory], float]) -> str | None:
    devices = sorted(
        {label for label, _ in largest if _DEVICE_LABEL.match(label)}
    )
    if len(devices) != 1:
        # Zero devices means a CPU-only run; more than one means the report
        # describes a split this observation does not model.
        return None
    return devices[0]


__all__ = ["parse_llama_cpp_allocation"]
