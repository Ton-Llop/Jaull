from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, LoadingIndicator, Static

from jaull.advisor.service import AdvisorService
from jaull.domain.model import ModelAnalysis
from jaull.exceptions import (
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    JaullError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)
from jaull.huggingface.url_parser import normalize_repo_id
from jaull.tui.widgets.banner import Banner
from jaull.tui.widgets.cli_equivalent import CliEquivalent
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel


class InspectScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner(
            "Inspect a Hugging Face model",
            "Enter a repo_id (owner/name) or a huggingface.co URL.",
        )
        with Vertical(classes="card"):
            yield Static("Model", classes="card-title")
            yield Input(placeholder="Qwen/Qwen2.5-7B-Instruct", id="inspect-input")
            with Horizontal():
                yield Button("Inspect", id="inspect-btn", classes="-primary")
        yield LoadingIndicator(id="inspect-loading")
        yield VerticalScroll(Vertical(id="inspect-content"))
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#inspect-loading", LoadingIndicator).display = False
        self.query_one("#inspect-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "inspect-btn":
            self._run_inspect()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "inspect-input":
            self._run_inspect()

    def _run_inspect(self) -> None:
        reference = self.query_one("#inspect-input", Input).value.strip()
        content = self.query_one("#inspect-content", Vertical)
        content.remove_children()
        if not reference:
            content.mount(WarningsPanel(["Please enter a repo_id or URL."]))
            return
        try:
            repo_id = normalize_repo_id(reference)
        except InvalidModelReferenceError as exc:
            content.mount(WarningsPanel([str(exc)]))
            return

        self.query_one("#inspect-loading", LoadingIndicator).display = True
        self.run_worker(lambda: self._worker(repo_id), thread=True)

    def _worker(self, repo_id: str) -> None:
        try:
            analysis = _advisor(self).inspect_model(repo_id)
        except (
            ModelNotFoundError,
            ModelAccessDeniedError,
            HuggingFaceUnavailableError,
            JaullError,
        ) as exc:
            self.app.call_from_thread(self._render_error, str(exc))
            return
        self.app.call_from_thread(self._populate, analysis)

    def _render_error(self, message: str) -> None:
        self.query_one("#inspect-loading", LoadingIndicator).display = False
        content = self.query_one("#inspect-content", Vertical)
        content.remove_children()
        content.mount(WarningsPanel([message]))

    def _populate(self, analysis: ModelAnalysis) -> None:
        self.query_one("#inspect-loading", LoadingIndicator).display = False
        content = self.query_one("#inspect-content", Vertical)
        content.remove_children()

        repo = analysis.repo
        header_rows: list[tuple[str, str]] = [
            ("Repository", repo.repo_id),
            ("Type", analysis.classification.primary_type.value),
        ]
        if repo.pipeline_tag:
            header_rows.append(("Task", repo.pipeline_tag))
        if repo.license:
            header_rows.append(("License", repo.license))
        if analysis.classification.formats:
            header_rows.append(
                (
                    "Formats",
                    ", ".join(sorted(f.value for f in analysis.classification.formats)),
                )
            )
        size_label = (
            _fmt(analysis.total_size_bytes) if analysis.total_size_bytes else "unknown"
        )
        header_rows.append(("Repository size", size_label))
        if analysis.classification.gguf_variants:
            largest = max(
                v.total_bytes for v in analysis.classification.gguf_variants
            )
            header_rows.append(("Largest variant", _fmt(largest)))
        header_rows.append(("Files", str(len(analysis.files))))

        content.mount(SummaryCard("Repository", header_rows))

        if analysis.classification.gguf_variants:
            variant_rows = [
                (v.quantization, f"{len(v.files)} file(s) · {_fmt(v.total_bytes)}")
                for v in analysis.classification.gguf_variants
            ]
            content.mount(SummaryCard("GGUF variants", variant_rows))

        if analysis.relevant_files:
            content.mount(
                SummaryCard(
                    "Relevant files",
                    [("", name) for name in analysis.relevant_files],
                )
            )

        if analysis.warnings:
            content.mount(WarningsPanel(analysis.warnings))

        content.mount(CliEquivalent(f"jaull inspect {repo.repo_id}"))


def _advisor(screen: Screen[None]) -> AdvisorService:
    from jaull.tui.app import JaullApp

    assert isinstance(screen.app, JaullApp)
    return screen.app.advisor


def _fmt(byte_count: int | None) -> str:
    if byte_count is None:
        return "unknown"
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
