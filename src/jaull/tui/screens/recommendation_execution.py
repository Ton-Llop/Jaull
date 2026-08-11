from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static, TextArea

from jaull.artifacts.errors import ArtifactError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import InferenceResult
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.exceptions import InvalidModelReferenceError, QuantizationNotFoundError
from jaull.execution.errors import ExecutionError
from jaull.recommendation.models import ModelRecommendation

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp


class RecommendationExecutionScreen(Screen[None]):
    """Run a recommendation through single-turn llama.cpp calls."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._artifact: ModelArtifact | None = None
        self._log_messages: list[str] = []

    @property
    def recommendation(self) -> ModelRecommendation:
        return self._recommendation

    @property
    def log_messages(self) -> tuple[str, ...]:
        return tuple(self._log_messages)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield _RunHeader(self._recommendation)
        with VerticalScroll(id="run-history"):
            yield Static(
                "Enter a prompt to prepare the local artifact and run a single-turn generation.",
                id="run-empty-state",
                classes="text-muted",
            )
        with Vertical(id="run-console"):
            yield Static("", id="run-error-message", classes="run-error-message")
            yield Static("Ready", id="run-status", classes="run-status")
            with Horizontal(id="run-composer"):
                yield TextArea(
                    "",
                    placeholder="Ask a single-turn question",
                    id="run-prompt-input",
                    soft_wrap=True,
                    show_line_numbers=False,
                )
                yield Button("Generate", id="run-generate", classes="-primary")
                yield Button("Back", id="run-back")
        yield Footer()

    def on_mount(self) -> None:
        self._clear_error()
        if self._runtime() is None:
            self._render_error(
                "This recommendation has no executable llama.cpp runtime configuration."
            )
            self.query_one("#run-generate", Button).disabled = True
            return
        self.query_one("#run-prompt-input", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "run-back":
                self.app.pop_screen()
            case "run-generate":
                self._start_generate()

    def _start_generate(self) -> None:
        composer = self.query_one("#run-prompt-input", TextArea)
        prompt = composer.text.strip()
        if not prompt:
            self._render_error("Prompt must not be empty.")
            composer.focus()
            return
        runtime = self._runtime()
        if runtime is None:
            self._render_error(
                "This recommendation has no executable llama.cpp runtime configuration."
            )
            return

        self._clear_error()
        self._append_prompt(prompt)
        composer.clear()
        self._set_busy(True, "Preparing model..." if self._artifact is None else "Generating...")
        self.run_worker(lambda: self._generate_worker(prompt, runtime), thread=True)

    def _generate_worker(self, prompt: str, runtime: RuntimeRecommendation) -> None:
        app = self._app()
        try:
            artifact = self._artifact
            if artifact is None:
                artifact = self._prepare_artifact_from_worker()
                app.call_from_thread(self._remember_artifact, artifact)
            app.call_from_thread(self._record_step, "Generating")
            result = app.advisor.run_artifact(
                artifact=artifact,
                prompt=prompt,
                runtime=runtime,
            )
        except (
            InvalidModelReferenceError,
            QuantizationNotFoundError,
            ArtifactError,
            ExecutionError,
        ) as exc:
            app.call_from_thread(self._generation_failed, str(exc))
            return
        app.call_from_thread(self._generation_done, artifact, result)

    def _prepare_artifact_from_worker(self) -> ModelArtifact:
        app = self._app()
        rec = self._recommendation
        config = rec.evaluated.selected_configuration
        quantization = config.quantization if config is not None else None

        app.call_from_thread(
            self._record_step,
            f"Resolve artifact for {rec.repo_id}"
            + (f" ({quantization})" if quantization else ""),
        )
        artifact = app.advisor.resolve_artifact(
            rec.repo_id,
            quantization=quantization,
            revision=None,
        )
        if not artifact.is_downloaded:
            app.call_from_thread(self._record_step, "Downloading artifact")
            artifact = app.advisor.download_artifact(artifact)
        else:
            app.call_from_thread(self._record_step, f"Reuse local artifact {artifact.filename}")

        app.call_from_thread(self._record_step, "Verifying artifact")
        artifact = app.advisor.verify_artifact(artifact)
        app.call_from_thread(self._record_step, "Model ready")
        return artifact

    def _generation_done(self, artifact: ModelArtifact, result: InferenceResult) -> None:
        self._remember_artifact(artifact)
        self._append_response(result)
        self._set_busy(False, "Model ready")
        composer = self.query_one("#run-prompt-input", TextArea)
        composer.focus()
        self._scroll_history_end()

    def _generation_failed(self, message: str) -> None:
        self._set_busy(False, "Ready to retry")
        self._render_error(message)
        self.query_one("#run-prompt-input", TextArea).focus()

    def _remember_artifact(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact

    def _append_prompt(self, prompt: str) -> None:
        self.query_one("#run-empty-state", Static).display = False
        history = self.query_one("#run-history", VerticalScroll)
        history.mount(InferencePrompt(prompt))
        self._scroll_history_end()

    def _append_response(self, result: InferenceResult) -> None:
        self.query_one("#run-empty-state", Static).display = False
        history = self.query_one("#run-history", VerticalScroll)
        history.mount(InferenceResponse(self._model_label(), result))

    def _set_busy(self, busy: bool, status: str) -> None:
        self.query_one("#run-generate", Button).disabled = busy
        self.query_one("#run-prompt-input", TextArea).disabled = busy
        self._set_status(("● " if busy else "✓ ") + status)

    def _set_status(self, status: str) -> None:
        self.query_one("#run-status", Static).update(status)

    def _render_error(self, message: str) -> None:
        widget = self.query_one("#run-error-message", Static)
        widget.update(message)
        widget.display = True

    def _clear_error(self) -> None:
        widget = self.query_one("#run-error-message", Static)
        widget.update("")
        widget.display = False

    def _record_step(self, message: str) -> None:
        self._log_messages.append(message)
        if message in {"Downloading artifact", "Generating", "Verifying artifact", "Model ready"}:
            self._set_status(("✓ " if message == "Model ready" else "● ") + message)

    def _runtime(self) -> RuntimeRecommendation | None:
        estimate = self._recommendation.evaluated.memory_estimate
        if estimate is None or estimate.runtime_recommendation is None:
            return None
        runtime = estimate.runtime_recommendation
        if runtime.runtime is not RuntimeName.LLAMA_CPP:
            return None
        return runtime

    def _model_label(self) -> str:
        return self._recommendation.repo_id.rsplit("/", maxsplit=1)[-1]

    def _scroll_history_end(self) -> None:
        history = self.query_one("#run-history", VerticalScroll)
        self.call_after_refresh(history.scroll_end, animate=False)

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class _RunHeader(Vertical):
    DEFAULT_CLASSES = "run-header"

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation

    def compose(self) -> ComposeResult:
        rec = self._recommendation
        yield Static(f"Run · {rec.repo_id}", classes="run-title")
        meta = " · ".join(_run_metadata(rec))
        if meta:
            yield Static(meta, classes="run-meta")
        yield Static(_breadcrumb(), classes="workflow-breadcrumb")


class InferencePrompt(Vertical):
    DEFAULT_CLASSES = "inference-entry inference-prompt"

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield Static("YOU", classes="message-label")
        yield Static(Text(self._prompt), classes="message-text")


class InferenceResponse(Vertical):
    DEFAULT_CLASSES = "inference-entry inference-response"

    def __init__(self, model_label: str, result: InferenceResult) -> None:
        super().__init__()
        self._model_label = model_label
        self._result = result

    def compose(self) -> ComposeResult:
        yield Static(self._model_label.upper(), classes="message-label")
        yield Static(Text(_response_text(self._result.text)), classes="message-text")
        yield Static(
            f"{self._result.duration_seconds:.2f} s · {self._result.runtime}",
            classes="message-meta",
        )


def _run_metadata(rec: ModelRecommendation) -> list[str]:
    config = rec.evaluated.selected_configuration
    estimate = rec.evaluated.memory_estimate
    runtime = estimate.runtime_recommendation if estimate is not None else None

    values: list[str] = []
    if config is not None and config.quantization:
        values.append(config.quantization)
    if runtime is not None and runtime.runtime is not RuntimeName.UNKNOWN:
        values.append(runtime.runtime.value)
    if config is not None and config.context_length:
        values.append(f"ctx {config.context_length}")
    offload = _flag_value(runtime, "--n-gpu-layers")
    if offload:
        values.append(f"GPU offload {offload}")
    if estimate is not None and estimate.total_bytes is not None:
        values.append(f"{estimate.total_bytes / 1024**3:.2f} GiB")
    return values


def _breadcrumb() -> str:
    parts = [
        ("Hardware", False),
        ("Your needs", False),
        ("Search", False),
        ("Run", True),
    ]
    return "  >  ".join(
        f"[b]{label}[/b]" if current else f"[dim]{label}[/dim]"
        for label, current in parts
    )


def _flag_value(runtime: RuntimeRecommendation | None, name: str) -> str | None:
    if runtime is None:
        return None
    return next((flag.value for flag in runtime.flags if flag.name == name), None)


def _response_text(text: str) -> str:
    return text or "(empty response)"


__all__ = [
    "InferencePrompt",
    "InferenceResponse",
    "RecommendationExecutionScreen",
]
