"""Helpers to build in-memory GGUF headers for tests without shipping binary fixtures."""

from __future__ import annotations

import struct

_MAGIC = b"GGUF"
_VERSION = 3

# GGUF type codes (mirror gguf_reader.py private constants)
_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10


def build_header(kv: dict[str, object], version: int = _VERSION, tensor_count: int = 0) -> bytes:
    """Return a GGUF v2/v3-compatible header containing the given KV dict.

    Supports string, uint32, uint64 and list-of-string values — everything the
    real headers we care about use.
    """
    buf = bytearray()
    buf += _MAGIC
    buf += struct.pack("<I", version)
    buf += struct.pack("<Q", tensor_count)
    buf += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        buf += _pack_string(key)
        buf += _pack_value(value)
    return bytes(buf)


def _pack_string(text: str) -> bytes:
    payload = text.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def _pack_value(value: object) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", _T_STRING) + _pack_string(value)
    if isinstance(value, bool):
        raise TypeError("Use uint32/uint64 rather than bool in fixtures for clarity.")
    if isinstance(value, int):
        if value < 0 or value > 2**32 - 1:
            return struct.pack("<I", _T_UINT64) + struct.pack("<Q", value)
        return struct.pack("<I", _T_UINT32) + struct.pack("<I", value)
    if isinstance(value, list):
        if not value:
            return struct.pack("<II", _T_ARRAY, _T_STRING) + struct.pack("<Q", 0)
        if all(isinstance(v, str) for v in value):
            body = b"".join(_pack_string(v) for v in value)  # type: ignore[arg-type]
            return (
                struct.pack("<I", _T_ARRAY)
                + struct.pack("<I", _T_STRING)
                + struct.pack("<Q", len(value))
                + body
            )
        if all(isinstance(v, int) for v in value):
            body = b"".join(struct.pack("<I", v) for v in value)  # type: ignore[arg-type]
            return (
                struct.pack("<I", _T_ARRAY)
                + struct.pack("<I", _T_UINT32)
                + struct.pack("<Q", len(value))
                + body
            )
    raise TypeError(f"Unsupported test-fixture value: {value!r} ({type(value)})")
