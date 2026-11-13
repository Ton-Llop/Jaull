"""Newer llama.cpp build notice: newest b-tag, daily cache, age threshold, no network."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from jaull.domain.hardware import ComputeBackend
from jaull.runtime import policies
from jaull.runtime.llama_cpp_release import UnsupportedPlatformError
from jaull.runtime.llama_cpp_updates import (
    BuildRelease,
    fetch_newest_build,
    newest_build,
    package_from_release,
    update_notice,
)

NOW = datetime(2026, 12, 20, tzinfo=UTC)


def _release(tag: str, published: str, *names: str, draft: bool = False) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "published_at": published,
        "draft": draft,
        "prerelease": True,
        "assets": [{"name": name, "digest": "sha256:" + "ab" * 32} for name in names],
    }


LISTING = [
    _release("v1.0.0", "2026-12-19T10:00:00Z", "nightly-tag.txt"),
    _release("b12999", "2026-12-19T09:00:00Z", "llama-b12999-bin-ubuntu-x64.tar.gz"),
    _release("b13001", "2026-12-19T11:00:00Z", draft=True),
    _release("b13000", "2026-12-19T10:30:00Z", "llama-b13000-bin-ubuntu-x64.tar.gz"),
]


def _client(payload: object, calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _offline() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_newest_build_is_the_highest_b_tag_ignoring_drafts_and_v_tags() -> None:
    newest = newest_build(LISTING)

    assert newest is not None
    assert newest.tag == "b13000"
    assert newest.digests == {"llama-b13000-bin-ubuntu-x64.tar.gz": "ab" * 32}


def test_fetch_uses_the_cache_for_a_day(tmp_path: Path) -> None:
    cache = tmp_path / "releases.json"
    calls: list[str] = []
    client = _client(LISTING, calls)

    first = fetch_newest_build(client, cache_path=cache, now=NOW)
    second = fetch_newest_build(client, cache_path=cache, now=NOW + timedelta(hours=23))
    third = fetch_newest_build(client, cache_path=cache, now=NOW + timedelta(hours=25))

    assert first == second == third
    assert first is not None and first.tag == "b13000"
    assert calls == [policies.LLAMA_CPP_RELEASES_API_URL] * 2


def test_offline_without_cache_gives_no_answer(tmp_path: Path) -> None:
    assert fetch_newest_build(_offline(), cache_path=tmp_path / "r.json", now=NOW) is None


def test_offline_with_a_stale_cache_uses_it(tmp_path: Path) -> None:
    cache = tmp_path / "r.json"
    fetch_newest_build(_client(LISTING, []), cache_path=cache, now=NOW)

    stale = fetch_newest_build(_offline(), cache_path=cache, now=NOW + timedelta(days=3))

    assert stale is not None and stale.tag == "b13000"


def test_corrupt_cache_is_ignored(tmp_path: Path) -> None:
    cache = tmp_path / "r.json"
    cache.write_text("not json", encoding="utf-8")

    result = fetch_newest_build(_client(LISTING, []), cache_path=cache, now=NOW)

    assert result is not None
    assert json.loads(cache.read_text(encoding="utf-8"))["releases"]


def _build(number: int, published: datetime) -> BuildRelease:
    return BuildRelease(tag=f"b{number}", number=number, published_at=published, digests={})


def test_notice_only_after_the_age_threshold() -> None:
    pinned = datetime(2026, 10, 25, tzinfo=UTC)
    kwargs = {"pinned_build": "b11889", "pinned_published": "2026-10-25T00:00:00Z",
              "age_days": 30}

    assert update_notice(_build(12360, pinned + timedelta(days=29)), **kwargs) is None
    notice = update_notice(_build(13000, pinned + timedelta(days=31)), **kwargs)
    assert notice is not None
    assert "b13000" in notice and "b11889" in notice and "31 days" in notice
    assert "--latest" in notice and "not validated" in notice
    assert update_notice(_build(11000, pinned + timedelta(days=90)), **kwargs) is None
    assert update_notice(None, **kwargs) is None


def test_package_from_release_swaps_the_build_and_uses_github_digests() -> None:
    cuda = [
        "llama-b13000-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]
    release = BuildRelease(tag="b13000", number=13000, published_at=NOW,
                           digests=dict.fromkeys(cuda, "cd" * 32))

    package = package_from_release(release, "windows", "x86_64", ComputeBackend.CUDA)

    assert package.build == "b13000"
    assert [asset.name for asset in package.assets] == cuda
    assert package.archive.url.endswith("/b13000/llama-b13000-bin-win-cuda-12.4-x64.zip")
    assert package.directory_name == "b13000-windows-x86_64-cuda"


def test_package_from_release_refuses_a_file_without_digest() -> None:
    release = BuildRelease(tag="b13000", number=13000, published_at=NOW, digests={})

    with pytest.raises(UnsupportedPlatformError, match="no llama-b13000-bin-ubuntu-x64"):
        package_from_release(release, "linux", "x86_64", ComputeBackend.CPU)
