"""Finding a digest among the versions Hugging Face has published for one file."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

from jaull.huggingface.artifact_history import HuggingFaceArtifactHistory


def _api(digests: list[str | None]) -> Mock:
    """A repo whose file had these digests, newest commit first (None: not in that commit)."""
    commits = [
        SimpleNamespace(
            commit_id=f"c{index}", created_at=datetime(2026, 11, 20 - index, tzinfo=UTC),
        )
        for index in range(len(digests))
    ]
    by_commit = {f"c{index}": digest for index, digest in enumerate(digests)}
    api = Mock()
    api.list_repo_commits.return_value = commits

    def paths_info(repo_id: str, paths: str, *, revision: str) -> list[object]:
        digest = by_commit[revision]
        if digest is None:
            return []
        return [SimpleNamespace(lfs=SimpleNamespace(sha256=digest))]

    api.get_paths_info.side_effect = paths_info
    return api


def test_it_finds_the_version_and_stops_looking() -> None:
    api = _api(["new", "mid", "old", "older", "oldest"])

    found = HuggingFaceArtifactHistory(api).find_version("o/r", "m.gguf", "old", limit=50)

    assert found is not None
    assert (found.sha256, found.revision, found.date.day) == ("old", "c2", 18)
    assert api.get_paths_info.call_count == 3  # nothing asked beyond the match


def test_it_gives_up_after_the_commits_it_is_allowed_to_look_at() -> None:
    api = _api(["a", "b", "c", "d", "target"])

    assert HuggingFaceArtifactHistory(api).find_version("o/r", "m.gguf", "target", limit=3) is None
    assert api.get_paths_info.call_count == 3


def test_commits_without_the_file_or_a_digest_are_skipped() -> None:
    api = _api([None, "x"])
    api.get_paths_info.side_effect = [
        [],
        [SimpleNamespace(lfs=None)],
    ]

    assert HuggingFaceArtifactHistory(api).find_version("o/r", "m.gguf", "x", limit=50) is None
