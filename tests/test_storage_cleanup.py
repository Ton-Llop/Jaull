"""Freeing disk space: only Jaull's own model files and run folders, never by accident."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, Checkbox, Static

from jaull.artifacts.errors import ArtifactError
from jaull.artifacts.storage import ArtifactStorage, LocalModelFile
from jaull.tui.app import JaullApp
from jaull.tui.screens.storage import StorageScreen
from tests.test_tui_quality import FakeAdvisor


def _download(root: Path, repo: str, name: str, size: int) -> Path:
    folder = root / repo
    (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    path = folder / name
    path.write_bytes(b"x" * size)
    (folder / f"{name}.sha256").write_text("a" * 64 + "\n")
    for suffix in (".metadata", ".lock"):
        (folder / ".cache" / "huggingface" / "download" / f"{name}{suffix}").write_text("")
    return path


def test_listing_and_deleting_stay_inside_the_models_root(tmp_path: Path) -> None:
    root = tmp_path / "models"
    small = _download(root, "org/Small-GGUF", "small.gguf", 10)
    _download(root, "other/Big-GGUF", "big.gguf", 30)
    storage = ArtifactStorage(root)

    files = storage.local_files()
    assert [(item.repo_id, item.filename, item.size_bytes) for item in files] == [
        ("other/Big-GGUF", "big.gguf", 30), ("org/Small-GGUF", "small.gguf", 10),
    ]
    assert files[0].sha256 == "a" * 64  # The verified digest links it to stored results.

    assert storage.delete("org/Small-GGUF", "small.gguf") == 10
    assert not small.exists() and not (root / "org").exists()  # No empty leftovers.
    assert [item.filename for item in storage.local_files()] == ["big.gguf"]
    with pytest.raises(ArtifactError):
        storage.delete("../outside", "x.gguf")


class StorageAdvisor(FakeAdvisor):
    def __init__(self) -> None:
        super().__init__()
        self.models = [
            (LocalModelFile("org/A-GGUF", "a.gguf", Path("/m/a.gguf"), 2 * 1024**3, "a" * 64),
             True),
            (LocalModelFile("org/B-GGUF", "b.gguf", Path("/m/b.gguf"), 1024**3, None), False),
        ]
        self.runs = (3, 300 * 1024**2)
        self.deleted: list[str] = []

    def local_models(self) -> list[Any]:
        return list(self.models)

    def delete_local_model(self, repo_id: str, filename: str) -> int:
        self.deleted.append(filename)
        found = next(item for item, _ in self.models if item.filename == filename)
        self.models = [pair for pair in self.models if pair[0].filename != filename]
        return found.size_bytes

    def quality_run_folders(self) -> tuple[int, int]:
        return self.runs

    def delete_quality_run_folders(self) -> int:
        self.deleted.append("runs")
        size, self.runs = self.runs[1], (0, 0)
        return size


def test_nothing_is_deleted_until_the_second_press() -> None:
    async def scenario() -> None:
        advisor = StorageAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            for _ in range(50):
                await pilot.pause(0.02)
                if not screen._busy:
                    break
            delete = screen.query_one("#storage-delete", Button)
            status = screen.query_one("#storage-status", Static)
            assert delete.disabled  # Nothing chosen yet.
            assert "they stay after deleting" in "\n".join(
                str(note.render()) for note in screen.query(".storage-note")
            )
            screen.query_one("#storage-model-0", Checkbox).value = True
            screen.query_one("#storage-runs", Checkbox).value = True
            await pilot.pause()
            assert "2.29 GiB" in str(status.render())
            delete.press()
            await pilot.pause()
            assert advisor.deleted == [] and "Press Delete again" in str(status.render())
            delete.press()
            for _ in range(50):
                await pilot.pause(0.02)
                if advisor.deleted and not screen._busy:
                    break
            assert advisor.deleted == ["a.gguf", "runs"]
            assert "Freed 2.29 GiB" in str(status.render())
            assert not screen.query("#storage-runs")  # The list shows what is left.

    asyncio.run(scenario())
