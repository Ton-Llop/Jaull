"""HTTP Range fetcher that progressively grows the request until the GGUF header parses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from local_ai_check.domain.enrichment import GgufHeaderMetadata
from local_ai_check.exceptions import (
    GgufHeaderIncompleteError,
    GgufHeaderInvalidError,
)
from local_ai_check.metadata import gguf_reader
from local_ai_check.metadata.policies import (
    HTTP_TIMEOUT_SECONDS,
    INITIAL_HEADER_RANGE_BYTES,
    MAX_HEADER_DOWNLOAD_BYTES,
    RANGE_GROWTH_FACTOR,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RangeResponse:
    body: bytes
    honored_range: bool
    status_code: int


class HttpRangeClient(Protocol):
    """Injectable HTTP client so tests can feed synthetic bytes."""

    def fetch_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> RangeResponse: ...


@dataclass(frozen=True)
class HeaderFetchResult:
    header: GgufHeaderMetadata | None
    bytes_downloaded: int
    warnings: list[str]


def fetch_gguf_header(url: str, http_client: HttpRangeClient) -> HeaderFetchResult:
    """Fetch enough of ``url`` to parse the GGUF metadata table.

    Grows the requested range exponentially up to
    :data:`~local_ai_check.metadata.policies.MAX_HEADER_DOWNLOAD_BYTES` before
    giving up.
    """
    warnings: list[str] = []
    requested = INITIAL_HEADER_RANGE_BYTES
    last_downloaded = 0

    while requested <= MAX_HEADER_DOWNLOAD_BYTES:
        try:
            response = http_client.fetch_range(
                url=url,
                start=0,
                end=requested - 1,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            warnings.append(
                f"Timed out ({HTTP_TIMEOUT_SECONDS:.1f}s) fetching GGUF header from {url}."
            )
            return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )
        except OSError as exc:
            warnings.append(f"Network error fetching GGUF header: {exc}.")
            return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )

        last_downloaded = len(response.body)

        if response.status_code == 416:
            # Range not satisfiable — file might be smaller than requested.
            warnings.append(
                f"Server returned 416 for range {requested}; file may be smaller."
            )
            return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )

        if response.status_code not in {200, 206}:
            warnings.append(
                f"Unexpected HTTP status {response.status_code} while reading GGUF header."
            )
            return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )

        if not response.honored_range:
            warnings.append(
                "Server ignored Range header; using truncated buffer to avoid a full download."
            )
            # We still try to parse what we have.

        try:
            header = gguf_reader.parse_header(response.body)
        except GgufHeaderIncompleteError:
            if not response.honored_range:
                # No point growing: server keeps sending the full file.
                warnings.append(
                    "Truncated buffer is too small and the server ignores Range; giving up."
                )
                return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )
            requested *= RANGE_GROWTH_FACTOR
            continue
        except GgufHeaderInvalidError as exc:
            warnings.append(f"Invalid GGUF header: {exc}.")
            return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )

        if header is None:
            warnings.append("File does not start with a GGUF magic; skipping header read.")
            return HeaderFetchResult(
                header=None, bytes_downloaded=last_downloaded, warnings=warnings
            )

        return HeaderFetchResult(header=header, bytes_downloaded=last_downloaded, warnings=warnings)

    warnings.append(
        f"GGUF header did not fit within the maximum download budget "
        f"({MAX_HEADER_DOWNLOAD_BYTES // (1024 * 1024)} MiB); giving up."
    )
    return HeaderFetchResult(header=None, bytes_downloaded=last_downloaded, warnings=warnings)


class HttpxRangeClient:
    """Production Range client using ``httpx`` (already a huggingface_hub dependency).

    Follows redirects, respects the requested Range, forwards an optional bearer token
    and never buffers more than what was requested.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._client = httpx.Client(follow_redirects=True)

    def fetch_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> RangeResponse:
        headers = {"Range": f"bytes={start}-{end}"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = self._client.get(url, headers=headers, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise OSError(str(exc)) from exc

        honored = response.status_code == 206 or bool(
            response.headers.get("Content-Range")
        )
        return RangeResponse(
            body=response.content,
            honored_range=honored,
            status_code=response.status_code,
        )

    def close(self) -> None:
        self._client.close()


__all__ = [
    "HeaderFetchResult",
    "HttpRangeClient",
    "HttpxRangeClient",
    "RangeResponse",
    "fetch_gguf_header",
]
