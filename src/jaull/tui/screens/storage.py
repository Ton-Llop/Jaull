"""Choose what to delete from Jaull's own data to free disk space.

Only downloaded model files and evaluation run folders: stored quality results,
benchmarks and experiments stay. Nothing is deleted without a second press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Static

from jaull.exceptions import JaullError
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.artifacts.storage import LocalModelFile


def _size(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB" if value >= 1024**3 // 10 else f"{value / 1024**2:.0f} MB"


class _Loaded(Message):
    def __init__(self, models: list[tuple[LocalModelFile, bool]], runs: tuple[int, int],
                 status: str) -> None:
        super().__init__()
        self.models, self.runs, self.status = models, runs, status


class StorageScreen(Screen[None]):
    BINDINGS = [("escape", "back", "Back")]

    DEFAULT_CSS = """
    StorageScreen #storage-body { padding: 0 2; }
    StorageScreen #storage-list { height: auto; }
    StorageScreen .storage-note { color: $text-muted; padding-left: 4; }
    StorageScreen #storage-actions { height: auto; padding: 0 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._models: list[tuple[LocalModelFile, bool]] = []
        self._runs = (0, 0)
        self._armed: tuple[str, ...] | None = None
        self._busy = True

    def compose(self) -> ComposeResult:
        yield ContextBar("Free disk space", aside="Storage")
        with VerticalScroll(id="storage-body"):
            yield Static(
                "Choose what to delete. Stored quality results, benchmarks and experiments "
                "are kept; a deleted model is downloaded again if you evaluate it later.",
                classes="text-secondary",
            )
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

    def _advisor(self) -> AdvisorService:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app.advisor

    def _load(self, status: str) -> None:
        self._busy = True
        advisor = self._advisor()

        def read() -> None:
            # post_message is thread-safe and never waits: call_from_thread would
            # block this worker forever if the screen closed meanwhile.
            self.post_message(
                _Loaded(advisor.local_models(), advisor.quality_run_folders(), status),
            )

        self.run_worker(read, thread=True, exclusive=True)

    @on(_Loaded)
    async def _loaded(self, message: _Loaded) -> None:
        self._models, self._runs = message.models, message.runs
        self._busy, self._armed = False, None
        container = self.query_one("#storage-list", Vertical)
        await container.remove_children()
        if not self._models and not self._runs[0]:
            await container.mount(Static("Nothing to delete.", classes="text-muted"))
        for index, (item, measured) in enumerate(self._models):
            await container.mount(Checkbox(
                f"{item.repo_id} / {item.filename}  {_size(item.size_bytes)}",
                id=f"storage-model-{index}",
            ))
            if measured:
                await container.mount(Static(
                    "Has stored quality results; they stay after deleting the file.",
                    classes="storage-note",
                ))
        if self._runs[0]:
            await container.mount(Checkbox(
                f"Evaluation run folders ({self._runs[0]})  {_size(self._runs[1])}",
                id="storage-runs",
            ))
            await container.mount(Static(
                "Logs and raw bundles. Stored results already embed what they need.",
                classes="storage-note",
            ))
        self._refresh(message.status)

    def _selected(self) -> tuple[list[LocalModelFile], bool]:
        models = [item for index, (item, _) in enumerate(self._models)
                  if self.query_one(f"#storage-model-{index}", Checkbox).value]
        runs = bool(self.query("#storage-runs")) and self.query_one(
            "#storage-runs", Checkbox).value
        return models, runs

    def _refresh(self, status: str = "") -> None:
        models, runs = self._selected()
        total = sum(item.size_bytes for item in models) + (self._runs[1] if runs else 0)
        self.query_one("#storage-delete", Button).disabled = self._busy or not total
        self.query_one("#storage-status", Static).update(
            status or (f"Selected: {_size(total)}." if total else "Select what to delete.")
        )

    @on(Checkbox.Changed)
    def _changed(self) -> None:
        self._armed = None  # A changed selection needs its own confirmation.
        self._refresh()

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "storage-back":
            self.action_back()
        elif event.button.id == "storage-delete" and not self._busy:
            models, runs = self._selected()
            chosen = (*(f"{item.repo_id}/{item.filename}" for item in models),
                      *(("runs",) if runs else ()))
            if self._armed != chosen:
                self._armed = chosen
                total = sum(item.size_bytes for item in models) + (self._runs[1] if runs else 0)
                self.query_one("#storage-status", Static).update(
                    f"Delete {len(chosen)} item(s), {_size(total)}? Press Delete again to confirm."
                )
                return
            self._delete(models, runs)

    def _delete(self, models: list[LocalModelFile], runs: bool) -> None:
        self._busy = True
        self._refresh("Deleting...")
        advisor = self._advisor()

        def remove() -> None:
            freed, errors = 0, []
            for item in models:
                try:
                    freed += advisor.delete_local_model(item.repo_id, item.filename)
                except (OSError, ValueError, JaullError) as exc:
                    errors.append(f"{item.filename}: {exc}")
            if runs:
                freed += advisor.delete_quality_run_folders()
            status = f"Freed {_size(freed)}." + (f" Failed: {'; '.join(errors)}" if errors else "")
            self.post_message(
                _Loaded(advisor.local_models(), advisor.quality_run_folders(), status),
            )

        self.run_worker(remove, thread=True, exclusive=True)

    def action_back(self) -> None:
        if not self._busy:
            self.dismiss(None)
