"""Hugging Face implementation of :class:`ArtifactHistoryProtocol`.

A repository can replace a file under the same name. Walking its commits from the newest,
the first one whose copy has a given digest is the last in which that version was still
the published one.
"""

from __future__ import annotations

from huggingface_hub import HfApi

from jaull.artifacts.ports import PublishedVersion


class HuggingFaceArtifactHistory:
    def __init__(self, api: HfApi) -> None:
        self._api = api

    def find_version(
        self, repo_id: str, filename: str, sha256: str, *, limit: int,
    ) -> PublishedVersion | None:
        wanted = sha256.lower()
        for commit in self._api.list_repo_commits(repo_id)[:limit]:
            for info in self._api.get_paths_info(
                repo_id, filename, revision=commit.commit_id,
            ):
                digest = getattr(getattr(info, "lfs", None), "sha256", None)
                if isinstance(digest, str) and digest.lower() == wanted:
                    return PublishedVersion(
                        sha256=digest.lower(), revision=commit.commit_id,
                        date=commit.created_at,
                    )
        return None


__all__ = ["HuggingFaceArtifactHistory"]
