"""Minimal GGUF v2/v3 metadata-table parser.

Reads only the leading key/value block of a GGUF file. Never touches tensor
data. Designed to be fed incrementally by :mod:`range_reader`: if the buffer
is truncated mid-string or mid-value the parser raises
:class:`GgufHeaderIncompleteError` so the caller can enlarge the range and
retry with a bigger prefix.

GGUF spec reference:
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from typing import Any

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.exceptions import (
    GgufHeaderIncompleteError,
    GgufHeaderInvalidError,
)

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

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def _need(self, n: int) -> None:
        if self.offset + n > len(self.data):
            raise GgufHeaderIncompleteError(
                f"Need {self.offset + n} bytes, have {len(self.data)}."
            )

    def read_u32(self) -> int:
        self._need(4)
        value: int = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_u64(self) -> int:
        self._need(8)
        value: int = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return value

    def read_string(self) -> str:
        length = self.read_u64()
        if length > 1_000_000:  # sanity, spec allows huge but keys/values are small
            raise GgufHeaderInvalidError(f"String length {length} looks malformed.")
        self._need(length)
        raw = self.data[self.offset : self.offset + length]
        self.offset += length
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufHeaderInvalidError(f"Non-UTF-8 GGUF string: {exc}") from exc

    def read_fixed(self, type_id: int) -> Any:
        fmt, size = _FIXED_FORMATS[type_id]
        self._need(size)
        value = struct.unpack_from(fmt, self.data, self.offset)[0]
        self.offset += size
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
    version = cursor.read_u32()
    if version not in _SUPPORTED_VERSIONS:
        raise GgufHeaderInvalidError(f"Unsupported GGUF version: {version}")

    # v2 uses u32 for counts; v3 uses u64. In practice both are read as u64 in v3
    # while v2 keeps them as u64 too (spec clarifies: from v2 onward, u64).
    cursor.read_u64()               # tensor_count — not needed for metadata
    metadata_kv_count = cursor.read_u64()
    if metadata_kv_count > 100_000:
        raise GgufHeaderInvalidError(
            f"Metadata KV count {metadata_kv_count} is implausible."
        )

    raw_kv: dict[str, object] = {}
    for _ in range(metadata_kv_count):
        key = cursor.read_string()
        raw_kv[key] = _read_value(cursor)

    return _build_header(raw_kv)


def _read_value(cursor: _Cursor) -> object:
    type_id = cursor.read_u32()
    if type_id == _T_STRING:
        return cursor.read_string()
    if type_id == _T_ARRAY:
        return _read_array(cursor)
    if type_id in _FIXED_FORMATS:
        return cursor.read_fixed(type_id)
    raise GgufHeaderInvalidError(f"Unknown GGUF value type: {type_id}")


def _read_array(cursor: _Cursor) -> list[object]:
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
            result.append(_read_array(cursor))
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
