from __future__ import annotations

from dataclasses import dataclass, field

from jaull.metadata.range_reader import (
    RangeResponse,
    fetch_gguf_header,
)
from tests._gguf_fixtures import build_header


@dataclass
class _StubHttpClient:
    body: bytes
    honor_range: bool = True
    status_code: int = 206
    fail_first: int = 0  # raise this many times before succeeding
    timeout_first: int = 0
    calls: list[tuple[int, int]] = field(default_factory=list)

    def fetch_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> RangeResponse:
        self.calls.append((start, end))
        if self.timeout_first > 0:
            self.timeout_first -= 1
            raise TimeoutError("simulated timeout")
        if self.fail_first > 0:
            self.fail_first -= 1
            raise OSError("simulated network glitch")
        if self.honor_range:
            slice_body = self.body[start : end + 1]
            return RangeResponse(
                body=slice_body, honored_range=True, status_code=self.status_code
            )
        # Server ignores Range: returns the full body regardless.
        return RangeResponse(body=self.body, honored_range=False, status_code=200)


def _big_header() -> bytes:
    # Build a header whose serialized form is > 256 KiB so we need at least one growth.
    long_string = "x" * 300_000
    return build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 8192,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "big_metadata_blob": long_string,
        }
    )


def test_first_range_is_enough_when_header_is_small() -> None:
    body = build_header({"general.architecture": "llama", "llama.context_length": 4096})
    client = _StubHttpClient(body=body)
    result = fetch_gguf_header(url="https://example/x.gguf", http_client=client)
    assert result.header is not None
    assert result.header.architecture == "llama"
    assert len(client.calls) == 1


def test_progressive_growth_reads_the_header() -> None:
    body = _big_header()
    client = _StubHttpClient(body=body)
    result = fetch_gguf_header(url="https://example/x.gguf", http_client=client)
    assert result.header is not None
    assert result.header.context_length == 8192
    assert len(client.calls) > 1
    # Successive ranges must be strictly larger.
    assert client.calls[1][1] > client.calls[0][1]


def test_server_ignoring_range_produces_warning() -> None:
    body = build_header({"general.architecture": "llama", "llama.context_length": 4096})
    client = _StubHttpClient(body=body, honor_range=False)
    result = fetch_gguf_header(url="https://example/x.gguf", http_client=client)
    # Body is small enough to parse anyway, but the warning must be present.
    assert any("Range" in w for w in result.warnings)
    assert result.header is not None


def test_timeout_returns_result_with_warning() -> None:
    body = build_header({"general.architecture": "llama"})
    client = _StubHttpClient(body=body, timeout_first=1)
    result = fetch_gguf_header(url="https://example/x.gguf", http_client=client)
    assert result.header is None
    assert any("Timed out" in w for w in result.warnings)


def test_network_error_returns_result_with_warning() -> None:
    body = build_header({"general.architecture": "llama"})
    client = _StubHttpClient(body=body, fail_first=1)
    result = fetch_gguf_header(url="https://example/x.gguf", http_client=client)
    assert result.header is None
    assert any("Network error" in w for w in result.warnings)
