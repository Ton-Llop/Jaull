from __future__ import annotations

from pathlib import Path

import pytest

from jaull.artifacts.errors import ArtifactError
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact


def _artifact(filename: str = "model-q5_k_m.gguf") -> ModelArtifact:
    return ModelArtifact(
        repo_id="owner/repo",
        revision="abc123",
        filename=filename,
        format="gguf",
        quantization="Q5_K_M",
        size_bytes=1024,
    )


def test_path_for_builds_owner_repo_filename(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    path = storage.path_for(_artifact())
    assert path == (tmp_path / "owner" / "repo" / "model-q5_k_m.gguf").resolve()


def test_path_for_rejects_parent_dir_filename(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    with pytest.raises(ArtifactError):
        storage.path_for(_artifact(filename="../etc/passwd"))


def test_path_for_rejects_absolute_filename(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    with pytest.raises(ArtifactError):
        storage.path_for(_artifact(filename="/etc/passwd"))


def test_path_for_rejects_backslash_filename(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    with pytest.raises(ArtifactError):
        storage.path_for(_artifact(filename="sub\\file.gguf"))


def test_path_for_rejects_bad_repo_id(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    bad = ModelArtifact(
        repo_id="justone",
        revision="x",
        filename="a.gguf",
        format="gguf",
    )
    with pytest.raises(ArtifactError):
        storage.path_for(bad)


def test_sha256_sidecar_roundtrip(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    file_path = tmp_path / "some.gguf"
    file_path.write_bytes(b"payload")
    digest = "ab" * 32
    storage.save_sha256(file_path, digest)
    assert storage.load_sha256(file_path) == digest


def test_load_sha256_missing_returns_none(tmp_path: Path) -> None:
    storage = ArtifactStorage(root=tmp_path)
    file_path = tmp_path / "nope.gguf"
    file_path.write_bytes(b"")
    assert storage.load_sha256(file_path) is None


@pytest.mark.parametrize(
    "filename",
    ["BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "../escape.gguf", "/etc/passwd", "a\\b.gguf"],
)
def test_a_file_that_cannot_be_stored_here_is_simply_not_downloaded(
    tmp_path: Path, filename: str,
) -> None:
    """Found on a B60 box: Qwen3.8-27B publishes its BF16 parts inside a folder, `exists` raised
    "Filename must be a single basename" and one such file removed every execution path of the
    model. Asking whether a file is already stored must answer no; building a path to write to
    still refuses it, so nothing outside the root can be reached."""
    storage = ArtifactStorage(root=tmp_path)

    assert storage.exists(_artifact(filename)) is False
    with pytest.raises(ArtifactError):
        storage.path_for(_artifact(filename))
