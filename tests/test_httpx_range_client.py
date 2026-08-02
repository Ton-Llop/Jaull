"""Tests for the production HTTP Range client.

Everything here runs against ``httpx.MockTransport`` — no network is touched.

The central guarantee under test is that ``HttpxRangeClient`` never buffers more
bytes than it asked for. A server that ignores ``Range`` and answers ``200 OK``
with a multi-gigabyte GGUF must not be able to drag that file into memory, so
the fake streams below are *lazy*: they count how many chunks the client
actually pulls. Pre-building a large ``bytes`` object would prove nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from local_ai_check.metadata.policies import INITIAL_HEADER_RANGE_BYTES
from local_ai_check.metadata.range_reader import (
    HttpxRangeClient,
    fetch_gguf_header,
)
from tests._gguf_fixtures import build_header

_MIB = 1024 * 1024


class _CountingStream(httpx.SyncByteStream):
    """Yields ``chunk`` up to ``chunk_count`` times, recording what was consumed.

    ``chunks_yielded`` staying far below ``chunk_count`` is the proof that the
    client stopped reading instead of draining the whole body.
    """

    def __init__(self, chunk: bytes, chunk_count: int) -> None:
        self._chunk = chunk
        self._chunk_count = chunk_count
        self.chunks_yielded = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for _ in range(self._chunk_count):
            self.chunks_yielded += 1
            yield self._chunk

    def close(self) -> None:
        self.closed = True

    @property
    def bytes_yielded(self) -> int:
        return self.chunks_yielded * len(self._chunk)


def _client(handler: object) -> HttpxRangeClient:
    return HttpxRangeClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _partial(body: bytes, start: int, total: int | str = "*") -> httpx.Response:
    """A well-formed 206 response for ``body`` placed at ``start``."""
    end = start + len(body) - 1
    return httpx.Response(
        206,
        content=body,
        headers={"Content-Range": f"bytes {start}-{end}/{total}"},
    )


# ---------------------------------------------------------------------------
# Well-behaved servers
# ---------------------------------------------------------------------------


def test_206_returns_exactly_the_requested_slice() -> None:
    payload = bytes(range(256)) * 8  # 2048 bytes

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=0-1023"
        return _partial(payload[:1024], start=0, total=len(payload))

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=1023, timeout=5.0
    )

    assert result.status_code == 206
    assert result.honored_range is True
    assert result.body == payload[:1024]
    assert len(result.body) == 1024


def test_content_range_at_a_non_zero_offset_is_honored() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=4096-8191"
        return _partial(b"y" * 4096, start=4096, total=1_000_000)

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=4096, end=8191, timeout=5.0
    )

    assert result.honored_range is True
    assert len(result.body) == 4096


def test_content_range_for_a_different_offset_is_not_honored() -> None:
    """A 206 whose Content-Range starts elsewhere is not the data we asked for."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206,
            content=b"z" * 512,
            headers={"Content-Range": "bytes 0-511/1000000"},
        )

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=4096, end=4607, timeout=5.0
    )

    assert result.honored_range is False


def test_multi_chunk_206_is_reassembled_in_order() -> None:
    chunks = [b"aaaa", b"bbbb", b"cccc", b"dddd"]

    class _Chunked(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield from chunks

        def close(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206,
            stream=_Chunked(),
            headers={"Content-Range": "bytes 0-15/16"},
        )

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=15, timeout=5.0
    )

    assert result.body == b"aaaabbbbccccdddd"
    assert result.honored_range is True


# ---------------------------------------------------------------------------
# The reason this client exists: servers that ignore Range
# ---------------------------------------------------------------------------


def test_server_ignoring_range_does_not_drain_the_stream() -> None:
    """A 200 backed by a 4 GiB lazy body must cost us one 1 MiB chunk, not 4 GiB."""
    stream = _CountingStream(chunk=b"\0" * _MIB, chunk_count=4096)  # 4 GiB if drained

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={"Content-Length": str(4096 * _MIB)},
        )

    result = _client(handler).fetch_range(
        url="https://example.test/huge.gguf", start=0, end=65535, timeout=5.0
    )

    assert result.status_code == 200
    assert result.honored_range is False
    assert len(result.body) == 65536
    # One 1 MiB chunk was enough to satisfy a 64 KiB request; the remaining
    # 4095 chunks were never generated.
    assert stream.chunks_yielded == 1
    assert stream.bytes_yielded == _MIB
    assert stream.closed is True


def test_plain_200_without_content_range_is_not_honored() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"q" * 100)

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=999, timeout=5.0
    )

    assert result.honored_range is False
    assert result.status_code == 200


def test_response_longer_than_requested_is_truncated_to_the_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(206, content=b"x" * 10_000)

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=99, timeout=5.0
    )

    assert len(result.body) == 100


def test_response_shorter_than_requested_is_returned_as_is() -> None:
    """A small file legitimately yields fewer bytes than the range asked for."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _partial(b"tiny file", start=0, total=9)

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=99_999, timeout=5.0
    )

    assert result.body == b"tiny file"
    assert result.honored_range is True


def test_stream_is_closed_after_an_early_break() -> None:
    stream = _CountingStream(chunk=b"a" * 4096, chunk_count=1000)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=4095, timeout=5.0
    )

    assert stream.closed is True
    assert stream.chunks_yielded == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_416_is_surfaced_as_a_status_not_an_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(416, content=b"Requested range not satisfiable")

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=1023, timeout=5.0
    )

    assert result.status_code == 416
    assert result.honored_range is False
    assert result.body == b""


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 503])
def test_http_error_statuses_raise_oserror(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"nope")

    with pytest.raises(OSError):
        _client(handler).fetch_range(
            url="https://example.test/x.gguf", start=0, end=1023, timeout=5.0
        )


def test_timeout_is_translated_to_timeouterror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated read timeout", request=request)

    with pytest.raises(TimeoutError):
        _client(handler).fetch_range(
            url="https://example.test/x.gguf", start=0, end=1023, timeout=0.01
        )


def test_transport_error_is_translated_to_oserror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    with pytest.raises(OSError):
        _client(handler).fetch_range(
            url="https://example.test/x.gguf", start=0, end=1023, timeout=5.0
        )


def test_timeout_raised_mid_stream_is_translated() -> None:
    """The mapping must also hold once we are already iterating the body."""

    class _FailingStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b"a" * 16
            raise httpx.ReadTimeout("stalled mid-body")

        def close(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(206, stream=_FailingStream())

    with pytest.raises(TimeoutError):
        _client(handler).fetch_range(
            url="https://example.test/x.gguf", start=0, end=99_999, timeout=5.0
        )


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_range_and_authorization_headers_are_sent() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return _partial(b"ok", start=10)

    client = HttpxRangeClient(token="s3cret", transport=httpx.MockTransport(handler))
    client.fetch_range(url="https://example.test/x.gguf", start=10, end=19, timeout=5.0)

    assert seen["range"] == "bytes=10-19"
    assert seen["authorization"] == "Bearer s3cret"


def test_redirects_are_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.test":
            return httpx.Response(302, headers={"Location": "https://cdn.test/x.gguf"})
        return _partial(b"redirected payload", start=0)

    result = _client(handler).fetch_range(
        url="https://example.test/x.gguf", start=0, end=99, timeout=5.0
    )

    assert result.body == b"redirected payload"
    assert result.honored_range is True


# ---------------------------------------------------------------------------
# Integration with the GGUF header reader
# ---------------------------------------------------------------------------


def test_fetch_gguf_header_reads_a_real_fixture_over_mock_transport() -> None:
    header = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 8192,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
    )
    # Pad so the "file" is much bigger than its header, as a real GGUF is.
    file_body = header + b"\0" * (2 * _MIB)

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.headers["Range"].removeprefix("bytes=")
        start_text, end_text = raw.split("-")
        start, end = int(start_text), int(end_text)
        return _partial(file_body[start : end + 1], start=start, total=len(file_body))

    result = fetch_gguf_header(
        url="https://example.test/model.gguf", http_client=_client(handler)
    )

    assert result.header is not None
    assert result.header.architecture == "llama"
    assert result.header.context_length == 8192
    assert result.header.block_count == 32
    assert result.warnings == []


def test_fetch_gguf_header_against_a_range_ignoring_server_stays_bounded() -> None:
    """End-to-end: a server that ignores Range must still cost us only a prefix."""
    header = build_header({"general.architecture": "llama", "llama.context_length": 4096})
    tail_chunks = 4096  # 4 GiB of tensor data behind the header, generated lazily
    streams: list[_CountingStream] = []

    class _HeaderThenHugeTail(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.chunks_yielded = 0
            self.closed = False

        def __iter__(self) -> Iterator[bytes]:
            self.chunks_yielded += 1
            yield header + b"\0" * (_MIB - len(header))
            for _ in range(tail_chunks):
                self.chunks_yielded += 1
                yield b"\0" * _MIB

        def close(self) -> None:
            self.closed = True

    def handler(request: httpx.Request) -> httpx.Response:
        stream = _HeaderThenHugeTail()
        streams.append(stream)  # type: ignore[arg-type]
        # No Content-Range: the server ignored the Range header entirely.
        return httpx.Response(200, stream=stream)

    result = fetch_gguf_header(
        url="https://example.test/huge.gguf", http_client=_client(handler)
    )

    assert result.header is not None
    assert result.header.architecture == "llama"
    assert any("Range" in warning for warning in result.warnings)
    # The header fit in the first 256 KiB request, so exactly one call was made
    # and it pulled a single 1 MiB chunk out of a nominally 4 GiB body.
    assert len(streams) == 1
    assert streams[0].chunks_yielded == 1
    assert result.bytes_downloaded == INITIAL_HEADER_RANGE_BYTES
    assert streams[0].closed is True
