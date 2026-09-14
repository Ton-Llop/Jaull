"""Optional local-artifact refinement for a verified llama.cpp implementation.

Placement rules are scoped to 689e227db, src/llama-model.cpp:1324-1349 and
src/models/qwen2.cpp. Stored tensor sizes are not measured CUDA allocations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import LlamaBenchBinaryStatus, LlamaBenchCapability
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.gguf import GgufTensorIndex
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.runtime import (
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
)
from jaull.exceptions import GgufHeaderIncompleteError, GgufHeaderInvalidError
from jaull.metadata.gguf_reader import read_local_tensor_index
from jaull.runtime.policies import LLAMA_CPP_HEADROOM_BYTES

VERIFIED_BUILD = "689e227db"


@dataclass(frozen=True)
class LlamaCppTensorContext:
    index: GgufTensorIndex | None = None
    unavailable_reason: str | None = None


def inspect_tensor_context(
    artifact: ModelArtifact,
    hardware: HardwareProfile,
    selection: RuntimeBackendSelection,
    capability: LlamaCppRuntimeCapability,
    *,
    benchmark_capability: LlamaBenchCapability | None = None,
) -> LlamaCppTensorContext:
    """Inspect locally only after the runtime scope has been established."""
    if capability.binary_status is not LlamaCppBinaryStatus.AVAILABLE or not re.search(
        rf"^version:\s+\d+\s+\({VERIFIED_BUILD}\)\s*$",
        capability.version_text or "",
        re.MULTILINE,
    ):
        return LlamaCppTensorContext(unavailable_reason="Runtime build is not verified.")
    if benchmark_capability is not None and (
        benchmark_capability.binary_status is not LlamaBenchBinaryStatus.AVAILABLE
        or not re.search(
            rf"^(?:version:\s+\d+\s+\({VERIFIED_BUILD}\)|"
            rf"build:\s+{VERIFIED_BUILD}\s+\(\d+\))\s*$",
            benchmark_capability.version_text or "",
            re.MULTILINE,
        )
    ):
        return LlamaCppTensorContext(unavailable_reason="llama-bench build is not verified.")
    cuda = [c for c in capability.backend_capabilities if c.backend is ComputeBackend.CUDA]
    if (
        selection.selected_backend is not ComputeBackend.CUDA
        or len(hardware.gpus) != 1
        or any(a.shared_memory for a in hardware.accelerators)
        or len(cuda) != 1
        or cuda[0].state is not LlamaCppBackendCapabilityState.CONFIRMED
        or len(cuda[0].devices) != 1
        or sum(len(c.devices) for c in capability.backend_capabilities) != 1
    ):
        return LlamaCppTensorContext(unavailable_reason="Requires one confirmed discrete CUDA GPU.")
    try:
        return LlamaCppTensorContext(index=read_local_tensor_index(artifact))
    except (OSError, GgufHeaderIncompleteError, GgufHeaderInvalidError) as exc:
        return LlamaCppTensorContext(
            unavailable_reason=f"Local tensor inspection unavailable: {exc}",
        )


@dataclass(frozen=True)
class TensorLayerBudget:
    requested_units: int
    gpu_blocks: int
    total_blocks: int
    gpu_block_bytes: int
    gpu_output_bytes: int
    host_input_bytes: int
    gpu_kv_bytes: int
    reserve_bytes: int
    headroom_bytes: int
    available_vram_bytes: int

    @property
    def required_bytes(self) -> int:
        return (
            self.gpu_block_bytes
            + self.gpu_output_bytes
            + self.gpu_kv_bytes
            + self.reserve_bytes
            + self.headroom_bytes
        )


def select_tensor_budget(
    estimate: MemoryEstimate,
    context: LlamaCppTensorContext,
) -> TensorLayerBudget | str:
    """Largest supported suffix placement within the existing launch budget.

    Positive runtime units include the output layer in this build. For u units,
    max(u-1, 0) repeating blocks live on GPU, up to the complete block count.
    This mapping belongs only to this backend and never consumes HFA placement.
    """
    index = context.index
    if index is None:
        return context.unavailable_reason or "Local tensor descriptors unavailable."
    total = index.block_count
    if index.architecture != "qwen2" or total is None:
        return "Local tensor placement supports only dense qwen2 with a block count."
    if total > len(index.tensors):
        return "Block count exceeds the tensor descriptor count."
    if (
        estimate.kv_cache.layers != total
        or estimate.kv_cache.component.bytes is None
        or estimate.kv_cache.component.bytes < 0
        or estimate.weights.component.bytes != index.file_size_bytes
        or estimate.repository.repo_id != index.artifact.repo_id
        or (
            estimate.weights.gguf_variant is not None
            and index.artifact.quantization is not None
            and estimate.weights.gguf_variant != index.artifact.quantization
        )
    ):
        return "Tensor artifact and estimate inputs do not agree or KV is unknown."
    by_name = {t.name: t for t in index.tensors}
    if not {"token_embd.weight", "output.weight", "output_norm.weight"} <= by_name.keys():
        return "Separate input/output tensors are required; shared weights are unsupported."
    if by_name["output.weight"].dimensions != by_name["token_embd.weight"].dimensions:
        return "Input/output tensor shapes disagree."
    if (
        len(by_name["output.weight"].dimensions) != 2
        or by_name["output_norm.weight"].dimensions != by_name["output.weight"].dimensions[:1]
    ):
        return "Unsupported input/output tensor dimensions."
    blocks = [0] * total
    block_names: list[set[str]] = [set() for _ in range(total)]
    output_bytes = 0
    input_bytes = 0
    required = {
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_down.weight",
        "ffn_up.weight",
    }
    optional = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
    for tensor in index.tensors:
        # Include per-tensor alignment in the budget, without calling it payload.
        size = (tensor.size_bytes + index.alignment - 1) // index.alignment * index.alignment
        if tensor.name == "token_embd.weight":
            input_bytes += size
        elif tensor.name in {"output.weight", "output_norm.weight"}:
            output_bytes += size
        else:
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
            if match is None or int(match[1]) >= total or match[2] not in required | optional:
                return f"Unclassified tensor: {tensor.name}."
            block = int(match[1])
            blocks[block] += size
            block_names[block].add(match[2])
    if any(not required <= names for names in block_names):
        return "Incomplete repeating-block tensor layout."
    vram = estimate.assessment.available_vram_bytes
    if vram is None:
        return "VRAM budget unavailable."
    kv = estimate.kv_cache.component.bytes
    for n in range(total, -1, -1):
        budget = TensorLayerBudget(
            requested_units=n + 1,
            gpu_blocks=n,
            total_blocks=total,
            gpu_block_bytes=sum(blocks[total - n :]),
            gpu_output_bytes=output_bytes,
            host_input_bytes=input_bytes,
            gpu_kv_bytes=(kv * n + total - 1) // total,
            reserve_bytes=estimate.inference_configuration.device_reserve_bytes,
            headroom_bytes=LLAMA_CPP_HEADROOM_BYTES,
            available_vram_bytes=vram,
        )
        if budget.required_bytes <= vram:
            return budget
    return TensorLayerBudget(
        requested_units=0,
        gpu_blocks=0,
        total_blocks=total,
        gpu_block_bytes=0,
        gpu_output_bytes=0,
        host_input_bytes=input_bytes,
        gpu_kv_bytes=0,
        reserve_bytes=0,
        headroom_bytes=0,
        available_vram_bytes=vram,
    )
