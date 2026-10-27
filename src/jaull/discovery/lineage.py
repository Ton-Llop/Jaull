"""Task signals a GGUF re-quantization inherits from the base repository it declares.

A repository that re-quantizes another one usually copies neither its `pipeline_tag` nor its task
tags, so `tencent/Hy-MT2-7B-GGUF` read as a plain chat model while its base is a translation
model. The base is declared with a structured tag (`base_model:quantized:X`), the same evidence
that confirms a model's identity; reading what that repository says it is for needs no name
heuristic.

Only tags that name a task the recommendation understands are inherited. The `pipeline_tag` is
never inherited whole: the base of a text GGUF can be multimodal (`image-text-to-text`), and
the GGUF still loads as a text model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from jaull.domain.candidates import ModelCandidate
from jaull.recommendation.scoring import task_vocabulary

logger = logging.getLogger(__name__)

_QUANTIZED_BASE_PREFIX = "base_model:quantized:"
# How often a wait for the lookups checks whether the search was cancelled.
_POLL_SECONDS = 0.1

FetchInfo = Callable[[str], Any]


def inherit_task_signals(
    candidates: Sequence[ModelCandidate],
    *,
    fetch: FetchInfo,
    max_workers: int,
    check_cancelled: Callable[[], None],
) -> list[ModelCandidate]:
    """Add to each re-quantization without a pipeline tag the task tags its base declares.

    Best effort: a base that cannot be read leaves its candidates as they were. One lookup per
    base, however many repositories re-quantize it.
    """
    bases = {
        base
        for candidate in candidates
        if (base := _lookup_base(candidate)) is not None
    }
    if not bases or max_workers < 1:
        return list(candidates)
    declared = _fetch_all(sorted(bases), fetch, max_workers, check_cancelled)
    result: list[ModelCandidate] = []
    for candidate in candidates:
        base = _lookup_base(candidate)
        inherited = declared.get(base) if base is not None else None
        if not inherited:
            result.append(candidate)
            continue
        known = {tag.lower() for tag in candidate.tags}
        extra = [tag for tag in inherited if tag.lower() not in known]
        result.append(
            candidate.model_copy(update={"tags": [*candidate.tags, *extra]}) if extra else candidate
        )
    return result


def _lookup_base(candidate: ModelCandidate) -> str | None:
    """The declared base of a re-quantization that has no pipeline tag of its own."""
    if candidate.pipeline_tag is not None:
        return None
    for tag in candidate.tags:
        if tag.lower().startswith(_QUANTIZED_BASE_PREFIX):
            return tag[len(_QUANTIZED_BASE_PREFIX):]
    return None


def _fetch_all(
    bases: list[str], fetch: FetchInfo, max_workers: int, check_cancelled: Callable[[], None],
) -> dict[str, list[str]]:
    vocabulary = task_vocabulary()

    def read(base: str) -> list[str]:
        try:
            info = fetch(base)
        except Exception:  # best effort: the candidate stays as it was
            logger.debug("Could not read the base repository %s", base, exc_info=True)
            return []
        names = [*(getattr(info, "tags", None) or []), getattr(info, "pipeline_tag", None) or ""]
        return sorted({name for name in names if name and name.lower() in vocabulary})

    found: dict[str, list[str]] = {}
    pending: dict[Future[list[str]], str] = {}
    queue = iter(bases)
    pool = ThreadPoolExecutor(
        max_workers=min(max_workers, len(bases)), thread_name_prefix="lineage",
    )
    try:
        check_cancelled()
        while True:
            while len(pending) < max_workers:
                base = next(queue, None)
                if base is None:
                    break
                pending[pool.submit(read, base)] = base
            if not pending:
                break
            done, _ = wait(list(pending), timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                found[pending.pop(future)] = future.result()
            check_cancelled()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return found
