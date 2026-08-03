"""HTTP Range fetcher that progressively grows the request until the GGUF header parses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.exceptions import (
    GgufHeaderIncompleteError,
    GgufHeaderInvalidError,
)
from jaull.metadata import gguf_reader
from jaull.metadata.policies import (
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
    :data:`~jaull.metadata.policies.MAX_HEADER_DOWNLOAD_BYTES` before
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


_CONTENT_RANGE_RE = re.compile(r"^\s*bytes\s+(\d+)-(\d+)/(?:\d+|\*)\s*$")


def _range_was_honored(response: httpx.Response, start: int) -> bool:
    """Decide whether the server actually served the range we asked for.

    A ``206`` is the unambiguous signal. A ``200`` means the server ignored the
    ``Range`` header and started streaming the whole file — unless it also sent
    a well-formed ``Content-Range`` that begins where we asked, which some
    proxies do. Anything else is treated as *not* honored, which is the safe
    direction: the caller then stops growing the range instead of hammering a
    server that would only ever resend the file from byte zero.
    """
    content_range = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE_RE.match(content_range)
    if match is not None and int(match.group(1)) == start:
        return True
    return response.status_code == 206 and not content_range


def _read_capped(response: httpx.Response, limit: int) -> bytes:
    """Consume at most ``limit`` bytes from an open streaming response.

    This is the whole point of the streaming client: iteration stops the moment
    the budget is met, so a server that ignores ``Range`` and replies with a
    multi-gigabyte GGUF never gets to send more than the header we wanted.
    """
    if limit <= 0:
        return b""
    buffer = bytearray()
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        remaining = limit - len(buffer)
        if remaining <= 0:
            break
        buffer.extend(chunk[:remaining])
        if len(buffer) >= limit:
            break
    return bytes(buffer)


class HttpxRangeClient:
    """Production Range client using ``httpx``.

    Follows redirects, forwards an optional bearer token and streams the
    response body so that no more than the requested number of bytes is ever
    buffered — even when the server ignores ``Range`` and answers ``200 OK``
    with the complete file.
    """

    def __init__(
        self,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token
        # ``transport`` exists so tests can inject an httpx.MockTransport
        # without reaching the network; production callers leave it as None.
        self._client = httpx.Client(follow_redirects=True, transport=transport)

    def fetch_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> RangeResponse:
        requested_size = end - start + 1
        headers = {"Range": f"bytes={start}-{end}"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            with self._client.stream(
                "GET", url, headers=headers, timeout=timeout
            ) as response:
                if response.status_code == 416:
                    # Range not satisfiable. The caller degrades on the status
                    # alone, so there is nothing worth reading from the body.
                    return RangeResponse(
                        body=b"", honored_range=False, status_code=416
                    )
                if response.status_code >= 400:
                    raise OSError(
                        f"HTTP {response.status_code} while fetching range from {url}"
                    )
                body = _read_capped(response, requested_size)
                honored = _range_was_honored(response, start)
                status_code = response.status_code
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise OSError(str(exc)) from exc

        if not honored:
            logger.warning(
                "Server ignored the Range header for %s (HTTP %s); "
                "stopped after %d of the %d requested bytes.",
                url,
                status_code,
                len(body),
                requested_size,
            )
        return RangeResponse(
            body=body,
            honored_range=honored,
            status_code=status_code,
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
