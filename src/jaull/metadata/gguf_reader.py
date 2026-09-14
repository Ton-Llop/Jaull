"""GGUF v2/v3 metadata and optional local tensor-descriptor reader.

The remote path reads only the leading key/value block. The local path also
reads tensor descriptors, never their payload. If a metadata buffer
is truncated mid-string or mid-value the parser raises
:class:`GgufHeaderIncompleteError` so the caller can enlarge the range and
retry with a bigger prefix.

GGUF spec reference:
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from math import prod
from os import fstat
from typing import Any, BinaryIO

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.gguf import GgufTensor, GgufTensorIndex
from jaull.exceptions import (
    GgufHeaderIncompleteError,
    GgufHeaderInvalidError,
)
from jaull.metadata.policies import MAX_HEADER_DOWNLOAD_BYTES

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = frozenset({2, 3})

# GGUF value type codes as defined in the spec.
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12

_FIXED_FORMATS: dict[int, tuple[str, int]] = {
    _T_UINT8: ("<B", 1),
    _T_INT8: ("<b", 1),
    _T_UINT16: ("<H", 2),
    _T_INT16: ("<h", 2),
    _T_UINT32: ("<I", 4),
    _T_INT32: ("<i", 4),
    _T_FLOAT32: ("<f", 4),
    _T_BOOL: ("<?", 1),
    _T_UINT64: ("<Q", 8),
    _T_INT64: ("<q", 8),
    _T_FLOAT64: ("<d", 8),
}


class _Cursor:
    __slots__ = ("data", "offset")

    def __init__(self, data: bytes | BinaryIO) -> None:
        self.data = data
        self.offset = 0

    def _read(self, n: int) -> bytes:
        if not isinstance(self.data, bytes):
            if self.offset + n > MAX_HEADER_DOWNLOAD_BYTES:
                raise GgufHeaderInvalidError("Local tensor header exceeds the read budget.")
            raw = self.data.read(n)
        else:
            raw = self.data[self.offset : self.offset + n]
        if len(raw) != n:
            raise GgufHeaderIncompleteError(
                f"Incomplete GGUF field at offset {self.offset}; need {n} bytes."
            )
        self.offset += n
        return raw

    def read_u32(self) -> int:
        value: int = struct.unpack("<I", self._read(4))[0]
        return value

    def read_u64(self) -> int:
        value: int = struct.unpack("<Q", self._read(8))[0]
        return value

    def read_string(self) -> str:
        length = self.read_u64()
        if length > 1_000_000:  # sanity, spec allows huge but keys/values are small
            raise GgufHeaderInvalidError(f"String length {length} looks malformed.")
        raw = self._read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufHeaderInvalidError(f"Non-UTF-8 GGUF string: {exc}") from exc

    def read_fixed(self, type_id: int) -> Any:
        fmt, size = _FIXED_FORMATS[type_id]
        value = struct.unpack(fmt, self._read(size))[0]
        return value


def parse_header(data: bytes) -> GgufHeaderMetadata | None:
    """Parse GGUF v2/v3 KV metadata from ``data``.

    Returns:
        - ``GgufHeaderMetadata`` if parsing completed.
        - ``None`` if the magic bytes are not GGUF (caller decides what to do).

    Raises:
        GgufHeaderIncompleteError: buffer ended mid-value; caller should retry
            with a larger range.
        GgufHeaderInvalidError: structural inconsistency (unknown type, bad
            version, decoding failure).
    """
    if len(data) < 4:
        raise GgufHeaderIncompleteError("Not enough bytes for magic.")
    if data[:4] != _MAGIC:
        return None

    cursor = _Cursor(data)
    cursor.offset = 4  # skip magic
    _, raw_kv = _read_metadata(cursor)
    return _build_header(raw_kv)


def _read_metadata(cursor: _Cursor) -> tuple[int, dict[str, object]]:
    version = cursor.read_u32()
    if version not in _SUPPORTED_VERSIONS:
        raise GgufHeaderInvalidError(f"Unsupported GGUF version: {version}")

    # Counts are u64 in both supported versions.
    tensor_count = cursor.read_u64()
    metadata_kv_count = cursor.read_u64()
    if metadata_kv_count > 100_000:
        raise GgufHeaderInvalidError(
            f"Metadata KV count {metadata_kv_count} is implausible."
        )

    raw_kv: dict[str, object] = {}
    for _ in range(metadata_kv_count):
        key = cursor.read_string()
        raw_kv[key] = _read_value(cursor)

    return tensor_count, raw_kv


# GGML stored block layouts: ggml/src/ggml-common.h at llama.cpp 689e227db.
# (elements per quantization block, stored bytes including scales).
# Q8_1 is intentionally excluded: not an on-disk weight format supported here.
_TENSOR_BLOCK_LAYOUTS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
}


def read_local_tensor_index(artifact: ModelArtifact) -> GgufTensorIndex:
    """Read only metadata/descriptors from one verified local GGUF v2/v3 file.

    Offsets and payload lengths are validated against fstat without reading
    payload bytes. The existing digest is carried through, never recomputed.
    Unsupported or inconsistent files raise an explicit parsing error.
    """
    if (
        artifact.format.lower() != "gguf" or artifact.local_path is None
        or not artifact.is_downloaded or not artifact.is_verified
    ):
        raise GgufHeaderInvalidError("A verified local GGUF artifact is required.")
    with artifact.local_path.open("rb", buffering=0) as stream:
        before = fstat(stream.fileno())
        if artifact.size_bytes is not None and artifact.size_bytes != before.st_size:
            raise GgufHeaderInvalidError("Local GGUF size differs from artifact identity.")
        cursor = _Cursor(stream)
        if cursor._read(4) != _MAGIC:
            raise GgufHeaderInvalidError("Local file is not GGUF.")
        count, metadata = _read_metadata(cursor)
        if not 0 < count <= 100_000:
            raise GgufHeaderInvalidError("Invalid GGUF tensor count.")
        if metadata.get("split.count", 1) != 1 or metadata.get("split.no", 0) != 0:
            raise GgufHeaderInvalidError("Multipart GGUF tensor inspection is unsupported.")
        alignment = metadata.get("general.alignment", 32)
        if (
            type(alignment) is not int or alignment <= 0
            or alignment > 4096 or alignment & (alignment - 1)
        ):
            raise GgufHeaderInvalidError("Invalid GGUF alignment.")
        tensors: list[GgufTensor] = []
        names: set[str] = set()
        for _ in range(count):
            name = cursor.read_string()
            if not name or name in names:
                raise GgufHeaderInvalidError("Empty or duplicate tensor name.")
            names.add(name)
            dimensions_count = cursor.read_u32()
            if not 1 <= dimensions_count <= 4:
                raise GgufHeaderInvalidError("Invalid tensor dimension count.")
            dimensions = tuple(cursor.read_u64() for _ in range(dimensions_count))
            type_id = cursor.read_u32()
            offset = cursor.read_u64()
            if any(d <= 0 for d in dimensions) or offset % alignment:
                raise GgufHeaderInvalidError("Invalid tensor dimensions or offset alignment.")
            layout = _TENSOR_BLOCK_LAYOUTS.get(type_id)
            if layout is None:
                raise GgufHeaderInvalidError(f"Unsupported GGML tensor type {type_id}.")
            elements, size = layout
            if dimensions[0] % elements:
                raise GgufHeaderInvalidError("Tensor row is not quantization-block aligned.")
            tensors.append(GgufTensor(name, dimensions, type_id, offset,
                                      prod(dimensions) // elements * size))
        data_offset = (cursor.offset + alignment - 1) // alignment * alignment
        previous_end = 0
        for tensor in sorted(tensors, key=lambda t: t.offset):
            if tensor.offset < previous_end:
                raise GgufHeaderInvalidError("Overlapping tensor payloads.")
            previous_end = tensor.offset + tensor.size_bytes
            if data_offset + previous_end > before.st_size:
                raise GgufHeaderInvalidError("Tensor payload extends beyond the file.")
        after = fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise GgufHeaderInvalidError("GGUF changed during tensor inspection.")
    header = _build_header(metadata)
    blocks = metadata.get(f"{header.architecture}.block_count")
    return GgufTensorIndex(
        artifact=artifact, architecture=header.architecture,
        block_count=blocks if type(blocks) is int and blocks > 0 else None,
        alignment=alignment, data_offset=data_offset, file_size_bytes=before.st_size,
        tensors=tuple(tensors),
    )


def _read_value(cursor: _Cursor) -> object:
    type_id = cursor.read_u32()
    if type_id == _T_STRING:
        return cursor.read_string()
    if type_id == _T_ARRAY:
        return _read_array(cursor)
    if type_id in _FIXED_FORMATS:
        return cursor.read_fixed(type_id)
    raise GgufHeaderInvalidError(f"Unknown GGUF value type: {type_id}")


def _read_array(cursor: _Cursor, depth: int = 0) -> list[object]:
    if depth > 32:
        raise GgufHeaderInvalidError("GGUF array nesting exceeds the parser limit.")
    inner_type = cursor.read_u32()
    count = cursor.read_u64()
    if count > 10_000_000:
        raise GgufHeaderInvalidError(f"Array length {count} is implausible.")
    result: list[object] = []
    for _ in range(count):
        if inner_type == _T_STRING:
            result.append(cursor.read_string())
        elif inner_type == _T_ARRAY:
            # Nested arrays are legal per spec but very rare; support them.
            result.append(_read_array(cursor, depth + 1))
        elif inner_type in _FIXED_FORMATS:
            result.append(cursor.read_fixed(inner_type))
        else:
            raise GgufHeaderInvalidError(
                f"Unknown GGUF array element type: {inner_type}"
            )
    return result


def _build_header(raw_kv: dict[str, object]) -> GgufHeaderMetadata:
    architecture = _str_or_none(raw_kv.get("general.architecture"))
    prefix = f"{architecture}." if architecture else ""

    def arch_key(name: str) -> object:
        return raw_kv.get(f"{prefix}{name}") if prefix else None

    return GgufHeaderMetadata(
        architecture=architecture,
        name=_str_or_none(raw_kv.get("general.name")),
        quantization_version=_int_or_none(raw_kv.get("general.quantization_version")),
        file_type=_int_or_none(raw_kv.get("general.file_type")),
        context_length=_int_or_none(arch_key("context_length")),
        embedding_length=_int_or_none(arch_key("embedding_length")),
        block_count=_int_or_none(arch_key("block_count")),
        head_count=_int_or_none(arch_key("attention.head_count")),
        head_count_kv=_int_or_none(arch_key("attention.head_count_kv")),
        rope_dim=_int_or_none(arch_key("rope.dimension_count")),
        source_repository=_str_or_none(
            raw_kv.get("general.source.huggingface.repository")
        ),
        raw_kv=raw_kv,
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


__all__ = ["parse_header"]
