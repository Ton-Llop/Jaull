"""A GGUF re-quantization inherits what its declared base repository says it is for."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from jaull.discovery.lineage import inherit_task_signals
from tests._workflow_fixtures import candidate


def _requantized(repo: str, base: str, **kwargs: object) -> object:
    return candidate(
        repo,
        tags=["gguf", "conversational", f"base_model:{base}", f"base_model:quantized:{base}"],
        pipeline_tag=None,
        **kwargs,  # type: ignore[arg-type]
    )


def _info(tags: list[str], pipeline: str | None = None) -> object:
    return SimpleNamespace(tags=tags, pipeline_tag=pipeline)


class _Hub:
    def __init__(self, repos: dict[str, object]) -> None:
        self.repos = repos
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, repo_id: str) -> object:
        with self._lock:
            self.calls.append(repo_id)
        if repo_id not in self.repos:
            raise RuntimeError("not found")
        return self.repos[repo_id]


def _run(items: list[object], hub: _Hub) -> list[object]:
    return inherit_task_signals(
        items, fetch=hub, max_workers=4, check_cancelled=lambda: None,  # type: ignore[arg-type]
    )


def test_the_task_a_base_repository_declares_reaches_its_requantization() -> None:
    """Found on a B60 box: `tencent/Hy-MT2-7B-GGUF` carries only `gguf` and `conversational`,
    but the base it declares has pipeline `translation` and the tag `translation`. The quantized
    repository copies none of it, so a translation model was offered as a general chat model."""
    hub = _Hub({"tencent/Hy-MT2-7B": _info(["transformers", "text-generation", "translation", "zh"],
                                           "translation")})

    [enriched] = _run([_requantized("tencent/Hy-MT2-7B-GGUF", "tencent/Hy-MT2-7B")], hub)

    assert "translation" in enriched.tags  # type: ignore[attr-defined]
    assert enriched.pipeline_tag is None  # type: ignore[attr-defined]  # never inherited whole
    assert "zh" not in enriched.tags  # type: ignore[attr-defined]  # only what names a task


def test_a_multimodal_base_pipeline_is_not_inherited() -> None:
    """The Qwen3.8-27B base declares `image-text-to-text`, while its GGUF is a text model that
    llama.cpp loads without the vision projector. Inheriting the pipeline would reject it."""
    hub = _Hub({"Qwen/Qwen3.8-27B": _info(["transformers", "image-text-to-text"],
                                          "image-text-to-text")})
    original = _requantized("unsloth/Qwen3.8-27B-GGUF", "Qwen/Qwen3.8-27B")

    [enriched] = _run([original], hub)

    assert enriched.pipeline_tag is None  # type: ignore[attr-defined]
    assert "image-text-to-text" not in enriched.tags  # type: ignore[attr-defined]


def test_only_requantizations_without_their_own_pipeline_are_looked_up_once_per_base() -> None:
    hub = _Hub({"org/Base": _info(["text-generation", "code"])})
    own_pipeline = candidate(
        "org/Own-GGUF", tags=["gguf", "base_model:quantized:org/Base"],
    )  # declares text-generation itself
    plain = candidate("org/Plain-7B-Instruct", tags=["instruct"], pipeline_tag=None)
    twin_a = _requantized("a/Base-GGUF", "org/Base")
    twin_b = _requantized("b/Base-GGUF", "org/Base")

    result = _run([own_pipeline, plain, twin_a, twin_b], hub)

    assert hub.calls == ["org/Base"]  # one lookup for both copies; the others need none
    assert result[0] == own_pipeline and result[1] == plain
    assert "code" in result[2].tags and "code" in result[3].tags  # type: ignore[attr-defined]


def test_a_base_that_cannot_be_read_leaves_the_candidate_as_it_was() -> None:
    original = _requantized("org/Gone-GGUF", "org/Missing")

    [unchanged] = _run([original], _Hub({}))

    assert unchanged == original


def test_the_lookups_run_at_the_same_time_and_stop_on_cancellation() -> None:
    barrier = threading.Barrier(3)

    def fetch(repo_id: str) -> object:
        barrier.wait(timeout=5)  # breaks if the lookups are made one at a time
        return _info(["translation"])

    items = [_requantized(f"o/M{i}-GGUF", f"org/Base{i}") for i in range(3)]
    result = inherit_task_signals(
        items, fetch=fetch, max_workers=3, check_cancelled=lambda: None,  # type: ignore[arg-type]
    )
    assert all("translation" in item.tags for item in result)  # type: ignore[attr-defined]

    class _Stop(Exception):
        pass

    def stop() -> None:
        raise _Stop

    with pytest.raises(_Stop):
        inherit_task_signals(
            items, fetch=lambda repo: _info([]), max_workers=2, check_cancelled=stop,  # type: ignore[arg-type]
        )
