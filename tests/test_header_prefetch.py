"""The GGUF headers a ranking needs are read concurrently before it asks for them one by one."""

from __future__ import annotations

import threading

import pytest

from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.model import GgufVariant, ModelAnalysis
from jaull.metadata.range_reader import RangeResponse
from jaull.workflow.header_prefetch import prefetch_gguf_headers
from tests._gguf_fixtures import build_header
from tests._workflow_fixtures import candidate, gguf_analysis

_HEADER = build_header(
    {
        "general.architecture": "qwen2",
        "general.size_label": "14B",
        "qwen2.embedding_length": 3584,
        "qwen2.block_count": 28,
        "qwen2.attention.head_count": 28,
        "qwen2.attention.head_count_kv": 4,
    }
)


class _Cache:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], GgufHeaderMetadata] = {}

    def get(self, analysis: ModelAnalysis, variant: GgufVariant) -> GgufHeaderMetadata | None:
        return self.entries.get((analysis.repo.repo_id, variant.quantization))

    def put(
        self, analysis: ModelAnalysis, variant: GgufVariant, header: GgufHeaderMetadata,
    ) -> None:
        self.entries[(analysis.repo.repo_id, variant.quantization)] = header


class _Client:
    """Answers every range with a valid header, optionally only once N reads are in flight."""

    def __init__(
        self,
        together: int = 0,
        error: Exception | None = None,
        block: threading.Event | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._barrier = threading.Barrier(together) if together else None
        self._error = error
        self._block = block
        self._lock = threading.Lock()

    def fetch_range(self, url: str, start: int, end: int, timeout: float) -> RangeResponse:
        with self._lock:
            self.calls.append(url)
        if self._error is not None:
            raise self._error
        if self._block is not None:
            self._block.wait(timeout=10)  # a request that does not answer
        if self._barrier is not None:
            self._barrier.wait(timeout=5)  # breaks if the reads are made one at a time
        return RangeResponse(body=_HEADER, honored_range=True, status_code=206)


def _evaluated(*quantizations: str, repo_id: str = "org/Model-7B-GGUF") -> EvaluatedCandidate:
    analysis = gguf_analysis(repo_id=repo_id, quantizations=quantizations)
    return EvaluatedCandidate(candidate=candidate(repo_id=repo_id), analysis=analysis)


def _run(items: list[EvaluatedCandidate], client: object, cache: object, workers: int = 4) -> int:
    return prefetch_gguf_headers(
        items, range_client=client, header_cache=cache,  # type: ignore[arg-type]
        max_workers=workers, check_cancelled=lambda: None,
    )


def test_every_variant_header_is_read_at_the_same_time_and_cached() -> None:
    cache = _Cache()
    client = _Client(together=3)

    read = _run([_evaluated("Q4_K_M", "Q5_K_M", "Q6_K")], client, cache, workers=3)

    assert read == 3
    assert {q for _repo, q in cache.entries} == {"Q4_K_M", "Q5_K_M", "Q6_K"}
    assert len(client.calls) == 3


def test_headers_that_are_already_cached_are_not_requested_again() -> None:
    cache = _Cache()
    first = _evaluated("Q4_K_M", "Q5_K_M")
    _run([first], _Client(), cache)
    client = _Client()

    read = _run([first, _evaluated("Q8_0", repo_id="org/Other-7B-GGUF")], client, cache)

    assert read == 1
    assert len(client.calls) == 1


def test_a_failed_read_never_stops_the_search() -> None:
    cache = _Cache()

    read = _run([_evaluated("Q4_K_M")], _Client(error=OSError("network down")), cache)

    assert read == 0
    assert cache.entries == {}


def test_without_a_range_client_or_a_cache_nothing_is_read() -> None:
    item = _evaluated("Q4_K_M")
    assert _run([item], None, _Cache()) == 0
    assert _run([item], _Client(), None) == 0


class _Stop(Exception):
    pass


def test_cancelling_stops_the_prefetch() -> None:
    def cancelled() -> None:
        raise _Stop

    with pytest.raises(_Stop):
        prefetch_gguf_headers(
            [_evaluated("Q4_K_M", "Q5_K_M")], range_client=_Client(), header_cache=_Cache(),  # type: ignore[arg-type]
            max_workers=2, check_cancelled=cancelled,
        )


def test_the_size_a_header_declares_reaches_the_analysis_of_its_repository() -> None:
    """Qwen2.5-14B and Mellum2 carry no size in a form the name parser reads, but every real GGUF
    header declares `general.size_label`. It is the artifact's own statement, not the repo name."""
    from jaull.workflow.header_prefetch import declare_sizes

    cache = _Cache()
    labelled = _evaluated("Q4_K_M", repo_id="org/Model-GGUF")
    unread = _evaluated("Q4_K_M", repo_id="org/Other-GGUF")
    assert labelled.analysis is not None
    variant = labelled.analysis.classification.gguf_variants[0]
    cache.put(labelled.analysis, variant, GgufHeaderMetadata(size_label="14B"))

    result = declare_sizes([labelled, unread], cache)

    assert result[0].analysis is not None and result[0].analysis.declared_size_label == "14B"
    assert result[1] is unread  # nothing known: left exactly as it was


def test_cancelling_does_not_wait_for_a_request_that_is_still_in_flight() -> None:
    """Found by an audit: the executor's context manager joined the blocked request, so a
    cancelled search kept waiting until the network answered (or timed out). The prefetch now
    returns as soon as the cancellation is noticed; the stuck request ends on its own timeout."""
    import time

    release = threading.Event()
    polls = 0

    def cancel_after_the_requests_started() -> None:
        nonlocal polls
        polls += 1
        if polls > 2:
            raise _Stop

    started = time.monotonic()
    try:
        with pytest.raises(_Stop):
            prefetch_gguf_headers(
                [_evaluated("Q4_K_M", "Q5_K_M")], range_client=_Client(block=release),  # type: ignore[arg-type]
                header_cache=_Cache(), max_workers=2,  # type: ignore[arg-type]
                check_cancelled=cancel_after_the_requests_started,
            )
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 2.0
    assert polls > 2  # it asked again while the requests were pending, not only on completion
