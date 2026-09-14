"""Small mixed-quantization GGUF files with independently specified byte sizes."""

import struct
from pathlib import Path

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
)
from tests._gguf_fixtures import _pack_string, build_header


def tensor_file(
    tmp_path: Path,
    *,
    metadata: dict[str, object] | None = None,
    omit: str | None = None,
) -> ModelArtifact:
    kv: dict[str, object] = {"general.architecture": "qwen2", "qwen2.block_count": 3}
    kv.update(metadata or {})
    # name, dimensions, GGML type, independently computed stored byte count
    tensors = [
        ("token_embd.weight", (256, 256), 12, 256 * 144),
        ("output.weight", (256, 256), 14, 256 * 210),
        ("output_norm.weight", (256,), 0, 256 * 4),
    ]
    for i in range(3):
        for name in (
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_down",
            "ffn_up",
        ):
            # Unequal blocks ensure policy uses the actual suffix, not an average.
            tensors.append((f"blk.{i}.{name}.weight", (256, i + 1), 12, 144 * (i + 1)))
    tensors = [t for t in tensors if t[0] != omit]
    header = bytearray(build_header(kv, tensor_count=len(tensors)))
    offset = 0
    for name, dims, type_id, size in tensors:
        header += _pack_string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", type_id, offset)
        offset += (size + 31) // 32 * 32
    header += b"\0" * (-len(header) % 32)
    path = tmp_path / "model.gguf"
    with path.open("wb") as stream:
        stream.write(header)
        stream.truncate(len(header) + offset)
    return ModelArtifact(
        repo_id="owner/model",
        revision="frozen",
        filename=path.name,
        format="gguf",
        local_path=path,
        size_bytes=path.stat().st_size,
        is_downloaded=True,
        is_verified=True,
    )


def cuda_selection() -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=ComputeBackend.CUDA,
        reason=RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
    )


def verified_capability() -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text="version: 10357 (689e227db)\nbuilt with GNU 13.3.0 for Linux x86_64",
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CUDA,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
                devices=[LlamaCppRuntimeDevice(backend=ComputeBackend.CUDA, runtime_id="CUDA0")],
            )
        ],
    )
