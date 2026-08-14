from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Static, TextArea

from jaull.artifacts.errors import ArtifactError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import InferenceResult
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.exceptions import InvalidModelReferenceError, QuantizationNotFoundError
from jaull.execution.errors import ExecutionError
from jaull.presentation.execution_report import inline_observation_summary
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.artifact_preparation import (
    prepare_recommendation_artifact,
    transformers_recommendation_artifact,
)

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class _GenerationStep(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class _GenerationDone(Message):
    def __init__(self, artifact: ModelArtifact, result: InferenceResult) -> None:
        super().__init__()
        self.artifact = artifact
        self.result = result


class _GenerationFailed(Message):
    def __init__(self, message: str, artifact: ModelArtifact | None = None) -> None:
        super().__init__()
        self.message = message
        self.artifact = artifact


class RecommendationExecutionScreen(Screen[None]):
    """Run a recommendation through single-turn llama.cpp calls."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._artifact: ModelArtifact | None = None
        self._log_messages: list[str] = []
        self._run_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-run",
        )
        self._future: Future[None] | None = None

    @property
    def recommendation(self) -> ModelRecommendation:
        return self._recommendation

    @property
    def log_messages(self) -> tuple[str, ...]:
        return tuple(self._log_messages)

    def compose(self) -> ComposeResult:
        # Three zones: a compact header, the history (which owns the screen),
        # and the composer with a one-line status above it.
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
                # No Back button: Escape and the footer already navigate, and
                # a second button next to Generate reads as a second choice.
                yield Button("Generate", id="run-generate", classes="-primary")
        yield Footer()

    def on_mount(self) -> None:
        self._run_closing.clear()
        self._clear_error()
        if self._runtime() is None:
            self._render_error(
                "This recommendation has no executable runtime configuration."
            )
            self.query_one("#run-generate", Button).disabled = True
            return
        self.query_one("#run-prompt-input", TextArea).focus()

    def on_unmount(self) -> None:
        self._run_closing.set()
        self._shutdown_executor()

    def _shutdown_executor(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-generate":
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
                "This recommendation has no executable runtime configuration."
            )
            return

        self._clear_error()
        self._append_prompt(prompt)
        composer.clear()
        self._set_busy(True, "Preparing model" if self._artifact is None else "Generating")
        self._future = self._executor.submit(
            self._generate_worker,
            self._app().advisor,
            prompt,
            runtime,
            self._artifact,
        )

    def _generate_worker(
        self,
        advisor: AdvisorService,
        prompt: str,
        runtime: RuntimeRecommendation,
        artifact: ModelArtifact | None,
    ) -> None:
        prepared_artifact = artifact
        try:
            if artifact is None:
                if runtime.runtime is RuntimeName.TRANSFORMERS:
                    self._post_step("Preparing Transformers runtime")
                    artifact = transformers_recommendation_artifact(
                        self._recommendation
                    )
                else:
                    artifact = self._prepare_artifact_from_worker(advisor)
                prepared_artifact = artifact
            self._post_step("Generating")
            result = advisor.run_artifact(
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
            self._post_generation_failed(str(exc), prepared_artifact)
            return
        if not self._run_closing.is_set():
            self.post_message(_GenerationDone(artifact, result))

    def _prepare_artifact_from_worker(self, advisor: AdvisorService) -> ModelArtifact:
        return prepare_recommendation_artifact(
            advisor=advisor,
            recommendation=self._recommendation,
            report_step=self._post_step,
        )

    @on(_GenerationStep)
    def _generation_step_message(self, message: _GenerationStep) -> None:
        if self._run_closing.is_set():
            return
        self._record_step(message.message)

    @on(_GenerationDone)
    def _generation_done_message(self, message: _GenerationDone) -> None:
        if self._run_closing.is_set():
            return
        self._generation_done(message.artifact, message.result)

    @on(_GenerationFailed)
    def _generation_failed_message(self, message: _GenerationFailed) -> None:
        if self._run_closing.is_set():
            return
        if message.artifact is not None:
            self._remember_artifact(message.artifact)
        self._generation_failed(message.message)

    def _post_step(self, message: str) -> None:
        if not self._run_closing.is_set():
            self.post_message(_GenerationStep(message))

    def _post_generation_failed(
        self, message: str, artifact: ModelArtifact | None
    ) -> None:
        if not self._run_closing.is_set():
            self.post_message(_GenerationFailed(message, artifact))

    def _generation_done(self, artifact: ModelArtifact, result: InferenceResult) -> None:
        self._remember_artifact(artifact)
        self._append_response(result)
        self._set_busy(False, "Model ready")
        composer = self.query_one("#run-prompt-input", TextArea)
        composer.focus()
        self._scroll_history_end()

    def _generation_failed(self, message: str) -> None:
        # Idle, not successful: the tick belongs to a run that produced output.
        self._set_busy(False, "Ready to retry", marker="○ ")
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
        history.mount(InferenceResponse(result))

    def _set_busy(self, busy: bool, status: str, marker: str | None = None) -> None:
        self.query_one("#run-generate", Button).disabled = busy
        self.query_one("#run-prompt-input", TextArea).disabled = busy
        self._set_status((marker or ("● " if busy else "✓ ")) + status)

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
        if message in {
            "Downloading artifact",
            "Generating",
            "Preparing Transformers runtime",
            "Verifying artifact",
            "Model ready",
        }:
            self._set_status(("✓ " if message == "Model ready" else "● ") + message)

    def _runtime(self) -> RuntimeRecommendation | None:
        estimate = self._recommendation.evaluated.memory_estimate
        if estimate is None or estimate.runtime_recommendation is None:
            return None
        runtime = estimate.runtime_recommendation
        if runtime.runtime not in {RuntimeName.LLAMA_CPP, RuntimeName.TRANSFORMERS}:
            return None
        return runtime

    def _scroll_history_end(self) -> None:
        history = self.query_one("#run-history", VerticalScroll)
        self.call_after_refresh(history.scroll_end, animate=False)

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class _RunHeader(Vertical):
    """Which model is loaded and how it is configured. Two lines, no box.

    The guided breadcrumb is gone: running is not one of the four steps, and
    repeating the path here only added a row.
    """

    DEFAULT_CLASSES = "run-header"

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation

    def compose(self) -> ComposeResult:
        rec = self._recommendation
        yield Static(rec.repo_id, classes="run-title")
        meta = " · ".join(_run_metadata(rec))
        if meta:
            yield Static(meta, classes="run-meta")


class InferencePrompt(Vertical):
    """What the user asked. A label and a left rule, not a speech bubble."""

    DEFAULT_CLASSES = "inference-entry inference-prompt"

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield Static("YOU", classes="message-label")
        yield Static(Text(self._prompt), classes="message-text")


class InferenceResponse(Vertical):
    """What the model answered: same shape, accent rule and a faint ground.

    Labelled MODEL rather than with the repository name — the header already
    says which model is loaded, and a 26-character label shouted above every
    answer competes with the answer itself.
    """

    DEFAULT_CLASSES = "inference-entry inference-response"

    def __init__(self, result: InferenceResult) -> None:
        super().__init__()
        self._result = result

    def compose(self) -> ComposeResult:
        yield Static("MODEL", classes="message-label")
        yield Static(Text(_response_text(self._result.text)), classes="message-text")
        yield Static(
            f"{inline_observation_summary(self._result.observation)} · {self._result.runtime}",
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
