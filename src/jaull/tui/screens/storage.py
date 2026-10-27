"""Choose what to delete from Jaull's own data to free disk space.

Only downloaded model files and evaluation run folders: stored quality results,
benchmarks and experiments stay. Nothing is deleted without a second press.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Static

from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.artifacts.storage import LocalModelFile

logger = logging.getLogger(__name__)


def _size(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KiB"
    return (f"{value / 1024**3:.2f} GiB" if value >= 1024**3 // 10
            else f"{value / 1024**2:.1f} MiB")


class _Loaded(Message):
    def __init__(self, models: list[tuple[LocalModelFile, bool]], runs: tuple[int, int],
                 status: str) -> None:
        super().__init__()
        self.models, self.runs, self.status = models, runs, status


class _Failed(Message):
    def __init__(self, detail: str) -> None:
        super().__init__()
        self.detail = detail


class StorageScreen(Screen[None]):
    BINDINGS = [("escape", "back", "Back")]

    DEFAULT_CSS = """
    StorageScreen #storage-body { padding: 0 2; }
    StorageScreen #storage-list { height: auto; }
    StorageScreen #storage-summary { height: auto; margin: 1 0; color: $text; }
    StorageScreen #storage-toolbar { height: auto; }
    StorageScreen .storage-model { height: auto; margin: 0 0 1 0; }
    StorageScreen .storage-model-line { height: auto; }
    StorageScreen .storage-model-line Checkbox { width: 1fr; height: auto; }
    StorageScreen .storage-size { width: 12; text-align: right; }
    StorageScreen .storage-note { color: $text-muted; padding-left: 4; }
    StorageScreen #storage-actions { height: auto; padding: 0 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._models: list[tuple[LocalModelFile, bool]] = []
        self._runs = (0, 0)
        self._armed: tuple[str, ...] | None = None
        self._busy = True
        self._deleting = False
        self._storage_closing = Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="storage")
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield ContextBar("Free disk space", aside="Storage")
        with VerticalScroll(id="storage-body"):
            yield Static(
                "Saved quality results, benchmarks and experiments are kept. "
                "Removed models can be downloaded again when needed.",
                classes="text-secondary",
            )
            yield Static("Reading local files...", id="storage-summary", markup=False)
            with Horizontal(id="storage-toolbar"):
                yield ActionButton("Select all", id="storage-all", disabled=True)
                yield ActionButton("Clear selection", id="storage-clear", disabled=True)
                yield ActionButton("Refresh", id="storage-refresh", disabled=True)
            yield Vertical(id="storage-list")
        yield Static("Reading local files...", id="storage-status", classes="status-line",
                     markup=False)
        with Horizontal(id="storage-actions"):
            yield Button("Delete selected", id="storage-delete", classes="-primary",
                         disabled=True)
            yield ActionButton("Back", id="storage-back")
        yield Footer()

    def on_mount(self) -> None:
        self._load("")

    def on_unmount(self) -> None:
        self._storage_closing.set()
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _advisor(self) -> AdvisorService:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app.advisor

    def _load(self, status: str) -> None:
        self._busy = True
        self._armed = None
        self._refresh("Reading local files...")
        advisor = self._advisor()
        self._future = self._executor.submit(self._read, advisor, status)

    def _read(self, advisor: AdvisorService, status: str) -> None:
        if self._storage_closing.is_set():
            return
        try:
            models = advisor.local_models()
            if self._storage_closing.is_set():
                return
            runs = advisor.quality_run_folders()
        except Exception as exc:
            logger.exception("Could not read local storage")
            if not self._storage_closing.is_set():
                detail = f"Could not read local files: {exc}"
                self.post_message(_Failed(f"{status} {detail}".strip()))
            return
        if not self._storage_closing.is_set():
            self.post_message(_Loaded(models, runs, status))

    @on(_Loaded)
    async def _loaded(self, message: _Loaded) -> None:
        if self._storage_closing.is_set():
            return
        self._models, self._runs = message.models, message.runs
        self._armed = None
        container = self.query_one("#storage-list", Vertical)
        await container.remove_children()
        if not self._models and not self._runs[0]:
            await container.mount(Static("Nothing to delete.", classes="text-muted"))
        for index, (item, measured) in enumerate(self._models):
            checkbox = Checkbox(item.repo_id, id=f"storage-model-{index}")
            checkbox.tooltip = str(item.path)
            row = Vertical(
                Horizontal(checkbox, Static(_size(item.size_bytes), classes="storage-size"),
                           classes="storage-model-line"),
                Static(item.filename, classes="storage-note", markup=False),
                classes="storage-model",
            )
            if measured:
                row.compose_add_child(Static(
                    "Quality results saved",
                    classes="storage-note",
                ))
            await container.mount(row)
        if self._runs[0]:
            await container.mount(Checkbox(
                f"Evaluation run folders ({self._runs[0]})  {_size(self._runs[1])}",
                id="storage-runs",
            ))
            await container.mount(Static(
                "Logs and raw bundles. Stored results already embed what they need.",
                classes="storage-note",
            ))
        model_bytes = sum(item.size_bytes for item, _ in self._models)
        total = model_bytes + self._runs[1]
        self.query_one("#storage-summary", Static).update(
            f"Model files  {len(self._models)}  |  {_size(model_bytes)}"
            f"\nEvaluation folders  {self._runs[0]}  |  {_size(self._runs[1])}"
            f"\nTotal removable  {_size(total)}"
        )
        self._busy = False
        self._deleting = False
        self._refresh(message.status)

    @on(_Failed)
    def _failed(self, message: _Failed) -> None:
        if self._storage_closing.is_set():
            return
        self._busy = False
        self._deleting = False
        self._armed = None
        if not self._models and not self._runs[0]:
            self.query_one("#storage-summary", Static).update("Local storage could not be read.")
        self._refresh(message.detail)

    def _selected(self) -> tuple[list[LocalModelFile], bool]:
        models = [item for index, (item, _) in enumerate(self._models)
                  if self.query(f"#storage-model-{index}")
                  and self.query_one(f"#storage-model-{index}", Checkbox).value]
        runs = bool(self.query("#storage-runs")) and self.query_one(
            "#storage-runs", Checkbox).value
        return models, runs

    def _refresh(self, status: str = "") -> None:
        models, runs = self._selected()
        total = sum(item.size_bytes for item in models) + (self._runs[1] if runs else 0)
        delete = self.query_one("#storage-delete", Button)
        delete.disabled = self._busy or not (models or runs)
        delete.label = "Confirm deletion" if self._armed is not None else "Delete selected"
        for checkbox in self.query(Checkbox):
            checkbox.disabled = self._busy
        self.query_one("#storage-all", Button).disabled = self._busy or not (
            self._models or self._runs[0]
        )
        self.query_one("#storage-clear", Button).disabled = self._busy or not (models or runs)
        self.query_one("#storage-refresh", Button).disabled = self._busy
        self.query_one("#storage-back", Button).disabled = self._deleting
        self.query_one("#storage-status", Static).update(
            status or (f"Selected: {_size(total)}." if total else "Select what to delete.")
        )

    @on(Checkbox.Changed)
    def _changed(self) -> None:
        if self._busy:
            return
        self._armed = None  # A changed selection needs its own confirmation.
        self._refresh()

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "storage-back":
            self.action_back()
        elif self._busy:
            return
        elif event.button.id == "storage-refresh":
            self._load("Local files updated.")
        elif event.button.id in {"storage-all", "storage-clear"}:
            self._armed = None
            for checkbox in self.query(Checkbox):
                checkbox.value = event.button.id == "storage-all"
            self._refresh()
        elif event.button.id == "storage-delete" and not self._busy:
            models, runs = self._selected()
            chosen = (*(f"{item.repo_id}/{item.filename}" for item in models),
                      *(("runs",) if runs else ()))
            if self._armed != chosen:
                if not chosen:
                    return
                self._armed = chosen
                self._refresh()
                total = sum(item.size_bytes for item in models) + (self._runs[1] if runs else 0)
                self.query_one("#storage-status", Static).update(
                    f"Delete {len(chosen)} item(s), {_size(total)}? "
                    "Press Confirm deletion to confirm."
                )
                return
            self._delete(models, runs)

    def _delete(self, models: list[LocalModelFile], runs: bool) -> None:
        self._busy = True
        self._deleting = True
        self._refresh("Deleting...")
        advisor = self._advisor()

        def remove() -> None:
            freed, errors = 0, []
            for item in models:
                if self._storage_closing.is_set():
                    return
                try:
                    freed += advisor.delete_local_model(item.repo_id, item.filename)
                except Exception as exc:
                    logger.exception("Could not delete local model %s", item.filename)
                    errors.append(f"{item.filename}: {exc}")
            if runs and not self._storage_closing.is_set():
                try:
                    freed += advisor.delete_quality_run_folders()
                except Exception as exc:
                    logger.exception("Could not delete evaluation folders")
                    errors.append(f"Evaluation folders: {exc}")
            status = f"Freed {_size(freed)}." + (f" Failed: {'; '.join(errors)}" if errors else "")
            self._read(advisor, status)

        self._future = self._executor.submit(remove)

    def action_back(self) -> None:
        if not self._deleting:
            self.dismiss(None)
