import struct
from pathlib import Path

import pytest

from jaull.exceptions import GgufHeaderIncompleteError, GgufHeaderInvalidError
from jaull.metadata.gguf_reader import _Cursor, read_local_tensor_index
from tests._gguf_fixtures import _pack_string
from tests._tensor_fixtures import tensor_file


def test_reads_mixed_tensor_types_without_reading_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tensor_file(tmp_path)
    endpoints = []
    original = _Cursor._read

    def track(cursor: _Cursor, size: int) -> bytes:
        data = original(cursor, size)
        endpoints.append(cursor.offset)
        return data

    monkeypatch.setattr(_Cursor, "_read", track)
    index = read_local_tensor_index(artifact)
    assert index.architecture == "qwen2"
    assert index.block_count == 3
    assert index.artifact is artifact
    tensors = {t.name: t for t in index.tensors}
    assert tensors["token_embd.weight"].size_bytes == 36_864
    assert tensors["output.weight"].size_bytes == 53_760
    assert index.data_offset < index.file_size_bytes
    assert max(endpoints) <= index.data_offset


@pytest.mark.parametrize(
    "relative_offset, packed, message",
    [
        (0, struct.pack("<I", 5), "dimension count"),
        (4, struct.pack("<Q", 0), "dimensions"),
        (20, struct.pack("<I", 999), "Unsupported GGML"),
        (24, struct.pack("<Q", 1), "alignment"),
        (24, struct.pack("<Q", 0), "Overlapping"),
    ],
)
def test_malformed_descriptors_fail_closed(
    tmp_path: Path,
    relative_offset: int,
    packed: bytes,
    message: str,
) -> None:
    artifact = tensor_file(tmp_path)
    assert artifact.local_path is not None
    data = bytearray(artifact.local_path.read_bytes())
    name = _pack_string("output.weight")
    start = data.index(name) + len(name) + relative_offset
    data[start : start + len(packed)] = packed
    artifact.local_path.write_bytes(data)
    with pytest.raises(GgufHeaderInvalidError, match=message):
        read_local_tensor_index(artifact)


@pytest.mark.parametrize(
    "metadata",
    [
        {"split.count": 2},
        {"general.alignment": 3},
        {"general.alignment": 0},
    ],
)
def test_invalid_or_multipart_metadata_is_rejected(tmp_path: Path, metadata: dict) -> None:
    with pytest.raises(GgufHeaderInvalidError):
        read_local_tensor_index(tensor_file(tmp_path, metadata=metadata))


def test_truncated_payload_is_detected_without_loading_it(tmp_path: Path) -> None:
    artifact = tensor_file(tmp_path)
    assert artifact.local_path is not None
    with artifact.local_path.open("r+b") as stream:
        stream.truncate(artifact.local_path.stat().st_size - 100)
    with pytest.raises(GgufHeaderInvalidError, match="beyond the file"):
        read_local_tensor_index(artifact.model_copy(update={"size_bytes": None}))


def test_truncated_header_is_explicit(tmp_path: Path) -> None:
    artifact = tensor_file(tmp_path)
    assert artifact.local_path is not None
    artifact.local_path.write_bytes(b"GGUF\x03")
    with pytest.raises(GgufHeaderIncompleteError):
        read_local_tensor_index(artifact.model_copy(update={"size_bytes": None}))
