"""Read the GGUF headers a ranking will need concurrently, before it asks for them one by one.

The estimator reads one header per quantization of every GGUF repository (a ranged request of
256 KiB or more each) while it ranks the plans, and it does so serially: on a cold cache that
was about 60 of the 64 seconds of a whole search. The requests are independent and wait on the
network, so they overlap well. Reading them first, into the header cache the estimator already
consults, changes nothing it computes, only how long it waits.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.gguf_variants import preferred_gguf_variants
from jaull.domain.model import GgufVariant, ModelAnalysis
from jaull.metadata.range_reader import HttpRangeClient
from jaull.metadata.service import read_variant_header
from jaull.ports.cache import GgufHeaderCacheProtocol

logger = logging.getLogger(__name__)

# How often a wait for the requests checks whether the search was cancelled.
_POLL_SECONDS = 0.1

_Job = tuple[ModelAnalysis, GgufVariant]


def prefetch_gguf_headers(
    evaluated: Sequence[EvaluatedCandidate],
    *,
    range_client: HttpRangeClient | None,
    header_cache: GgufHeaderCacheProtocol | None,
    max_workers: int,
    check_cancelled: Callable[[], None],
) -> int:
    """Fill ``header_cache`` for every GGUF variant not in it yet; return how many were read.

    Best effort: a failed read is left for the estimator, which then reads it as before.
    ``check_cancelled`` runs between completions and may raise to stop the run.
    """
    if range_client is None or header_cache is None or max_workers < 1:
        return 0
    jobs = _missing(evaluated, header_cache)
    if not jobs:
        return 0

    def read(job: _Job) -> bool:
        analysis, variant = job
        try:
            header = read_variant_header(variant, analysis.repo.repo_id, range_client, [])
        except Exception:  # best effort: the estimator repeats the read and reports it
            logger.debug("GGUF header prefetch failed for %s", analysis.repo.repo_id, exc_info=True)
            return False
        if header is None:
            return False
        header_cache.put(analysis, variant, header)
        return True

    read_count = 0
    pending: set[Future[bool]] = set()
    queue = iter(jobs)
    pool = ThreadPoolExecutor(
        max_workers=min(max_workers, len(jobs)), thread_name_prefix="gguf-header",
    )
    try:
        check_cancelled()
        while True:
            while len(pending) < max_workers:
                job = next(queue, None)
                if job is None:
                    break
                pending.add(pool.submit(read, job))
            if not pending:
                break
            done, pending = wait(pending, timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
            read_count += sum(1 for future in done if future.result())
            check_cancelled()  # also while every request is still pending
    except BaseException:
        # Do not join: a request that does not answer would keep a cancelled search waiting. The
        # queued jobs are dropped; one already on the network ends on the HTTP timeout.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)  # nothing is pending here
    return read_count


def declare_sizes(
    evaluated: Sequence[EvaluatedCandidate], header_cache: GgufHeaderCacheProtocol | None,
) -> list[EvaluatedCandidate]:
    """Attach to each GGUF analysis the size its cached headers declare, when they declare one.

    Every quantization of a repository carries the same `general.size_label`, so the first
    cached header that has one is enough. Candidates with nothing to add are returned as they
    were.
    """
    if header_cache is None:
        return list(evaluated)
    result: list[EvaluatedCandidate] = []
    for item in evaluated:
        label = _declared_size(item, header_cache)
        if label is None or item.analysis is None:
            result.append(item)
            continue
        result.append(item.model_copy(update={
            "analysis": item.analysis.model_copy(update={"declared_size_label": label}),
        }))
    return result


def _declared_size(
    item: EvaluatedCandidate, header_cache: GgufHeaderCacheProtocol,
) -> str | None:
    analysis = item.analysis
    if item.failed or analysis is None:
        return None
    if analysis.classification.primary_type is not RepositoryType.GGUF:
        return None
    for variant in preferred_gguf_variants(analysis.classification.gguf_variants):
        header = header_cache.get(analysis, variant)
        if header is not None and header.size_label:
            return header.size_label
    return None


def _missing(
    evaluated: Sequence[EvaluatedCandidate], header_cache: GgufHeaderCacheProtocol,
) -> list[_Job]:
    jobs: list[_Job] = []
    for item in evaluated:
        analysis = item.analysis
        if item.failed or analysis is None:
            continue
        if analysis.classification.primary_type is not RepositoryType.GGUF:
            continue
        for variant in preferred_gguf_variants(analysis.classification.gguf_variants):
            if header_cache.get(analysis, variant) is None:
                jobs.append((analysis, variant))
    return jobs
