"""Is there a llama.cpp build much newer than the pinned one?

Upstream publishes a build every few hours, so Jaull only mentions a newer one
when the pinned build is ``LLAMA_CPP_UPDATE_NOTICE_AGE_DAYS`` older. GitHub is
asked at most once a day; without network there is simply no notice. A newer
build is never installed on its own: ``runtime install --latest`` does it on
request, with the SHA-256 GitHub publishes for each file, and the result is not
validated by Jaull.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from jaull.domain.hardware import ComputeBackend
from jaull.runtime import policies
from jaull.runtime.llama_cpp_release import (
    LlamaCppPackage,
    ReleaseAsset,
    UnsupportedPlatformError,
    package_for,
)

_BUILD_TAG = re.compile(r"^b(\d+)$")
_SHA256_PREFIX = "sha256:"


@dataclass(frozen=True)
class BuildRelease:
    tag: str
    number: int
    published_at: datetime
    #: file name -> SHA-256, for the files GitHub publishes a digest for.
    digests: dict[str, str]


def newest_build(releases: list[dict[str, Any]]) -> BuildRelease | None:
    """The highest ``b<number>`` release in a GitHub ``/releases`` listing."""
    best: BuildRelease | None = None
    for release in releases:
        match = _BUILD_TAG.match(str(release.get("tag_name", "")))
        if match is None or release.get("draft"):
            continue
        number = int(match.group(1))
        if best is not None and number <= best.number:
            continue
        best = BuildRelease(
            tag=match.group(0),
            number=number,
            published_at=_parse_time(str(release["published_at"])),
            digests={
                str(asset["name"]): str(asset["digest"]).removeprefix(_SHA256_PREFIX)
                for asset in release.get("assets", [])
                if str(asset.get("digest") or "").startswith(_SHA256_PREFIX)
            },
        )
    return best


def fetch_newest_build(
    client: httpx.Client,
    *,
    cache_path: Path,
    now: datetime,
    ttl_seconds: float = policies.LLAMA_CPP_UPDATE_CHECK_TTL_SECONDS,
) -> BuildRelease | None:
    """The newest build, from a cache younger than ``ttl_seconds`` or from GitHub.

    ``None`` when GitHub cannot be reached and no cache exists: the notice is
    best effort and never blocks anything.
    """
    cached = _read_cache(cache_path)
    if cached is not None and now - cached[0] < timedelta(seconds=ttl_seconds):
        return newest_build(cached[1])
    try:
        response = client.get(policies.LLAMA_CPP_RELEASES_API_URL)
        response.raise_for_status()
        releases = response.json()
    except (httpx.HTTPError, ValueError):
        return newest_build(cached[1]) if cached is not None else None
    if not isinstance(releases, list):
        return None
    slim = [_slim(release) for release in releases if isinstance(release, dict)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": now.isoformat(), "releases": slim}), encoding="utf-8",
    )
    return newest_build(slim)


def update_notice(
    newest: BuildRelease | None,
    *,
    pinned_build: str = policies.LLAMA_CPP_PINNED_BUILD,
    pinned_published: str = policies.LLAMA_CPP_PINNED_BUILD_PUBLISHED,
    age_days: int = policies.LLAMA_CPP_UPDATE_NOTICE_AGE_DAYS,
) -> str | None:
    pinned = _BUILD_TAG.match(pinned_build)
    if newest is None or pinned is None or newest.number <= int(pinned.group(1)):
        return None
    days = (newest.published_at - _parse_time(pinned_published)).days
    if days < age_days:
        return None
    return (
        f"llama.cpp {newest.tag} is available. Jaull installs {pinned_build}, which is {days} "
        "days older and is the build validated with Jaull. To try the newer one, run "
        "`jaull runtime install --backend <backend> --latest` (not validated)."
    )


def package_from_release(
    release: BuildRelease, os_name: str, arch: str, backend: ComputeBackend,
) -> LlamaCppPackage:
    """The same files as the pinned package, from ``release``, with GitHub's digests."""
    pinned = package_for(os_name, arch, backend)
    assets = []
    for asset in pinned.assets:
        name = asset.name.replace(pinned.build, release.tag)
        sha256 = release.digests.get(name)
        if sha256 is None:
            raise UnsupportedPlatformError(
                f"llama.cpp {release.tag} has no {name} with a published SHA-256."
            )
        assets.append(ReleaseAsset(
            name=name, sha256=sha256,
            url=f"{policies.LLAMA_CPP_RELEASE_URL}/{release.tag}/{name}",
        ))
    return LlamaCppPackage(
        build=release.tag, os_name=os_name, arch=arch, backend=backend, assets=tuple(assets),
    )


def _slim(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag_name": release.get("tag_name"),
        "published_at": release.get("published_at"),
        "draft": release.get("draft", False),
        "assets": [
            {"name": asset.get("name"), "digest": asset.get("digest")}
            for asset in release.get("assets", []) if isinstance(asset, dict)
        ],
    }


def _read_cache(path: Path) -> tuple[datetime, list[dict[str, Any]]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(data["fetched_at"]), list(data["releases"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


__all__ = [
    "BuildRelease",
    "fetch_newest_build",
    "newest_build",
    "package_from_release",
    "update_notice",
]
