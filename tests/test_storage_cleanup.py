"""Freeing disk space: only Jaull's own model files and run folders, never by accident."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from textual.containers import Vertical
from textual.widgets import Button, Checkbox, ListItem, Static

from jaull.artifacts.errors import ArtifactError
from jaull.artifacts.storage import ArtifactStorage, LocalModelFile
from jaull.tui.app import JaullApp
from jaull.tui.screens.advanced_tools import AdvancedToolsScreen
from jaull.tui.screens.storage import StorageScreen
from tests.test_tui_quality import FakeAdvisor, _wait_until


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
            await _wait_until(pilot, lambda: not screen._busy)
            delete = screen.query_one("#storage-delete", Button)
            status = screen.query_one("#storage-status", Static)
            assert delete.disabled  # Nothing chosen yet.
            assert "Quality results saved" in "\n".join(
                str(note.render()) for note in screen.query(".storage-note")
            )
            screen.query_one("#storage-model-0", Checkbox).value = True
            screen.query_one("#storage-runs", Checkbox).value = True
            await pilot.pause()
            assert "2.29 GiB" in str(status.render())
            delete.press()
            await pilot.pause()
            assert advisor.deleted == [] and "Press Confirm deletion" in str(status.render())
            delete.press()
            await _wait_until(pilot, lambda: advisor.deleted and not screen._busy)
            assert advisor.deleted == ["a.gguf", "runs"]
            assert "Freed 2.29 GiB" in str(status.render())
            assert not screen.query("#storage-runs")  # The list shows what is left.

    asyncio.run(scenario())


def test_storage_stays_busy_until_the_list_and_status_are_updated(monkeypatch) -> None:
    async def scenario() -> None:
        app = JaullApp(advisor=StorageAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: not screen._busy)
            container = screen.query_one("#storage-list", Vertical)
            remove_children = container.remove_children
            entered, release = asyncio.Event(), asyncio.Event()

            async def delayed_remove():
                entered.set()
                await release.wait()
                await remove_children()

            monkeypatch.setattr(container, "remove_children", delayed_remove)
            screen._load("Updated local files.")
            try:
                async with asyncio.timeout(10):
                    await entered.wait()
                assert screen._busy
                assert str(screen.query_one("#storage-status", Static).render()) != (
                    "Updated local files."
                )
            finally:
                release.set()
            await _wait_until(pilot, lambda: not screen._busy)
            assert "Updated local files." in str(
                screen.query_one("#storage-status", Static).render(),
            )
    asyncio.run(scenario())


def test_storage_is_reachable_from_welcome_and_advanced_without_a_search() -> None:
    async def scenario() -> None:
        advisor = StorageAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            items = [item.id for item in app.screen.query(ListItem)]
            assert items.index("welcome-storage") == items.index("welcome-advanced") + 1
            await pilot.click("#welcome-storage")
            await _wait_until(pilot, lambda: isinstance(app.screen, StorageScreen)
                              and not app.screen._busy)
            await pilot.press("escape")
            await pilot.click("#welcome-advanced")
            assert isinstance(app.screen, AdvancedToolsScreen)
            app.screen._activate("storage")
            await _wait_until(pilot, lambda: isinstance(app.screen, StorageScreen)
                              and not app.screen._busy)
            assert advisor.deleted == []
            assert app.workflow_state is None
            await pilot.press("escape")
            assert isinstance(app.screen, AdvancedToolsScreen)

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (160, 48)])
def test_bulk_selection_refresh_and_confirmation_reset(size: tuple[int, int]) -> None:
    async def scenario() -> None:
        advisor = StorageAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: not screen._busy)
            summary = str(screen.query_one("#storage-summary", Static).render())
            assert "Model files  2" in summary and "Evaluation folders  3" in summary
            assert "3.29 GiB" in summary
            screen.query_one("#storage-all", Button).press()
            await pilot.pause()
            assert all(c.value for c in screen.query(Checkbox))
            delete = screen.query_one("#storage-delete", Button)
            delete.press()
            await pilot.pause()
            assert str(delete.label) == "Confirm deletion"
            screen.query_one("#storage-model-1", Checkbox).value = False
            await pilot.pause()
            assert str(delete.label) == "Delete selected"
            assert screen._armed is None
            screen.query_one("#storage-clear", Button).press()
            await pilot.pause()
            assert not any(c.value for c in screen.query(Checkbox))
            assert delete.disabled
            for _ in range(2):
                screen.query_one("#storage-refresh", Button).press()
                await _wait_until(pilot, lambda: not screen._busy)
                await pilot.pause()
                assert len(screen.query(Checkbox)) == 3
            assert advisor.deleted == []
            for button in screen.query(Button):
                assert button.region.right <= size[0]
                assert button.region.bottom <= size[1]

    asyncio.run(scenario())


def test_inventory_errors_are_recoverable() -> None:
    class Advisor(StorageAdvisor):
        fail = True

        def local_models(self) -> list[Any]:
            if self.fail:
                raise OSError("Cannot read model folder")
            return super().local_models()

    async def scenario() -> None:
        advisor = Advisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test() as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: not screen._busy)
            assert "Cannot read model folder" in str(
                screen.query_one("#storage-status", Static).render(),
            )
            assert not screen.query_one("#storage-refresh", Button).disabled
            advisor.fail = False
            screen.query_one("#storage-refresh", Button).press()
            await _wait_until(pilot, lambda: not screen._busy and len(screen.query(Checkbox)) == 3)
            assert advisor.deleted == []

    asyncio.run(scenario())


def test_empty_storage_and_zero_byte_files_are_handled() -> None:
    async def scenario() -> None:
        advisor = StorageAdvisor()
        advisor.models = []
        advisor.runs = (0, 0)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test() as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: not screen._busy)
            assert not screen.query(Checkbox)
            assert screen.query_one("#storage-all", Button).disabled
            assert screen.query_one("#storage-delete", Button).disabled
            advisor.models = [(LocalModelFile(
                "org/Empty-GGUF", "empty.gguf", Path("/m/empty.gguf"), 0, None,
            ), False)]
            screen.query_one("#storage-refresh", Button).press()
            await _wait_until(pilot, lambda: not screen._busy and bool(screen.query(Checkbox)))
            screen.query_one("#storage-model-0", Checkbox).value = True
            await pilot.pause()
            delete = screen.query_one("#storage-delete", Button)
            assert not delete.disabled
            delete.press()
            await pilot.pause()
            delete.press()
            await _wait_until(pilot, lambda: advisor.deleted == ["empty.gguf"] and not screen._busy)
            assert not screen.query(Checkbox)

    asyncio.run(scenario())


def test_partial_deletion_and_evaluation_errors_keep_the_screen_usable() -> None:
    class Advisor(StorageAdvisor):
        def delete_local_model(self, repo_id: str, filename: str) -> int:
            if filename == "a.gguf":
                raise PermissionError("File is locked")
            return super().delete_local_model(repo_id, filename)

        def delete_quality_run_folders(self) -> int:
            raise OSError("Cannot remove logs")

    async def scenario() -> None:
        advisor = Advisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test() as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: not screen._busy)
            screen.query_one("#storage-all", Button).press()
            await pilot.pause()
            delete = screen.query_one("#storage-delete", Button)
            delete.press()
            await pilot.pause()
            delete.press()
            await _wait_until(pilot, lambda: advisor.deleted == ["b.gguf"] and not screen._busy)
            status = str(screen.query_one("#storage-status", Static).render())
            assert "Freed 1.00 GiB" in status
            assert "File is locked" in status and "Cannot remove logs" in status
            assert len(screen.query(Checkbox)) == 2
            assert screen._armed is None
            assert not screen.query_one("#storage-refresh", Button).disabled

    asyncio.run(scenario())


def test_leaving_during_inventory_does_not_wait_for_the_worker() -> None:
    entered, release = threading.Event(), threading.Event()

    class Advisor(StorageAdvisor):
        def local_models(self) -> list[Any]:
            entered.set()
            release.wait(timeout=5)
            return super().local_models()

    async def scenario() -> None:
        app = JaullApp(advisor=Advisor())  # type: ignore[arg-type]
        async with app.run_test() as pilot:
            screen = StorageScreen()
            await app.push_screen(screen)
            try:
                await _wait_until(pilot, entered.is_set)
                async with asyncio.timeout(2):
                    await pilot.press("escape")
                    assert app.screen is not screen
                assert screen._storage_closing.is_set()
            finally:
                release.set()
            await _wait_until(pilot, lambda: screen._future is not None and screen._future.done())

    asyncio.run(scenario())
