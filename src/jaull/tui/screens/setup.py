"""Setup: leave a model running as a local server, in five steps on one screen.

1. Machine and backend: detected, and changeable when there is more than one.
2. llama.cpp: the build Jaull installed for that backend, or a button to install it.
3. Server: this PC only or the local network, how many people at once, context, port.
4. Model: one already downloaded, or a repository and quantization to download.
5. Start: the same composition as ``jaull serve start``, through ``AdvisorService``.

Everything slow runs in a worker thread; the screen only collects choices and
shows the text that ``presentation.serving`` writes for the CLI too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
)
from textual.worker import Worker, WorkerState

from jaull.application.model_reference import normalize_repo_id
from jaull.application.serving import policies as serving_policies
from jaull.domain.hardware import ComputeBackend
from jaull.exceptions import InvalidModelReferenceError
from jaull.presentation.serving import plan_lines, start_lines, status_lines
from jaull.runtime import policies as runtime_policies
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.artifacts.storage import LocalModelFile
    from jaull.domain.hardware import HardwareProfile
    from jaull.domain.runtime import RuntimeBackendSelection
    from jaull.execution.server_process import ServerStatus

_MODEL_HINT = (
    "Not sure which model? Run Guided analysis from the home screen; the model it "
    "downloads appears in this list."
)


class SetupScreen(Screen[None]):
    BINDINGS = [("escape", "back", "Back")]

    DEFAULT_CSS = """
    SetupScreen #setup-body { padding: 0 2; }
    SetupScreen .setup-heading { text-style: bold; margin: 1 0 0 0; }
    SetupScreen .setup-text { height: auto; color: $text; }
    SetupScreen .setup-note { height: auto; color: $text-muted; }
    SetupScreen Select, SetupScreen RadioSet { margin: 0 0 1 0; }
    SetupScreen #setup-numbers { height: auto; }
    SetupScreen .setup-field { width: 1fr; height: auto; }
    SetupScreen #setup-model-inputs { height: auto; }
    SetupScreen #setup-model-inputs Input { width: 1fr; }
    SetupScreen #setup-output { height: auto; margin: 1 0; }
    SetupScreen #setup-actions { height: auto; padding: 0 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._busy = True
        self._running = False
        self._models: list[LocalModelFile] = []
        self._worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        yield ContextBar("Set up a local server", aside="Setup")
        with VerticalScroll(id="setup-body"):
            yield Static("1. Machine and backend", classes="setup-heading")
            yield Static("Scanning hardware...", id="setup-hardware", classes="setup-text",
                         markup=False)
            yield Select[ComputeBackend]([], prompt="Backend", id="setup-backend",
                                         allow_blank=True)
            yield Static("2. llama.cpp", classes="setup-heading")
            yield Static("", id="setup-runtime", classes="setup-text", markup=False)
            yield ActionButton("Install llama.cpp", id="setup-install", disabled=True)
            yield Static("3. Server", classes="setup-heading")
            with RadioSet(id="setup-exposure"):
                yield RadioButton("Only this PC", value=True, id="setup-local")
                yield RadioButton("Share on the local network", id="setup-lan")
            with Horizontal(id="setup-numbers"):
                with Vertical(classes="setup-field"):
                    yield Label("People at once")
                    yield Input(str(serving_policies.SERVE_DEFAULT_USERS), type="integer",
                                id="setup-users")
                with Vertical(classes="setup-field"):
                    yield Label("Context per person (tokens)")
                    yield Input(str(serving_policies.SERVE_DEFAULT_CONTEXT_PER_USER),
                                type="integer", id="setup-context")
                with Vertical(classes="setup-field"):
                    yield Label("Port")
                    yield Input(str(runtime_policies.LLAMA_SERVER_DEFAULT_PORT),
                                type="integer", id="setup-port")
            yield Static("4. Model", classes="setup-heading")
            yield Select[int]([], prompt="Choose a downloaded model", id="setup-model",
                              allow_blank=True)
            with Horizontal(id="setup-model-inputs"):
                yield Input(placeholder="or a Hugging Face repo id", id="setup-repo")
                yield Input(placeholder="quantization, e.g. Q4_K_M", id="setup-quant")
            yield Static(_MODEL_HINT, classes="setup-note", markup=False)
            yield Static("5. Start", classes="setup-heading")
            yield Static("", id="setup-output", classes="setup-text", markup=False)
        yield Static("Reading this machine...", id="setup-status", classes="status-line",
                     markup=False)
        with Horizontal(id="setup-actions"):
            yield Button("Start server", id="setup-start", classes="-primary", disabled=True)
            yield ActionButton("Stop server", id="setup-stop", disabled=True)
            yield ActionButton("Back", id="setup-back")
        yield Footer()

    def on_mount(self) -> None:
        self._run(self._load, "Reading this machine...")

    # ------------------------------------------------------------------ workers

    def _run(self, work: Callable[[AdvisorService], None], status: str) -> None:
        self._busy = True
        self._refresh(status)
        advisor = self._advisor()
        self._worker = self.run_worker(lambda: work(advisor), thread=True,
                                       exit_on_error=False, group="setup")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # A step can start the next one; only the newest worker ends the busy state.
        if event.worker is not self._worker:
            return
        if event.state is WorkerState.ERROR:
            self._busy = False
            self._refresh(f"Failed: {event.worker.error}")
        elif event.state in {WorkerState.SUCCESS, WorkerState.CANCELLED} and self._busy:
            self._busy = False
            self._refresh()

    def _load(self, advisor: AdvisorService) -> None:
        hardware = advisor.scan_hardware()
        selection = advisor.select_runtime_backend(hardware)
        models = [item for item, _ in advisor.local_models()]
        status = advisor.server_status()
        self.app.call_from_thread(self._loaded, hardware, selection, models, status)

    def _loaded(
        self,
        hardware: HardwareProfile,
        selection: RuntimeBackendSelection,
        models: list[LocalModelFile],
        status: ServerStatus,
    ) -> None:
        accelerator = selection.selected_accelerator
        self.query_one("#setup-hardware", Static).update(
            f"{hardware.cpu.model or 'CPU'}; "
            + (f"accelerator {accelerator.name}." if accelerator is not None
               else "no usable accelerator, the server runs on the CPU.")
        )
        backends = list(dict.fromkeys([
            selection.selected_backend,
            *(item.backend for item in selection.alternatives),
            ComputeBackend.CPU,
        ]))
        backend = self.query_one("#setup-backend", Select)
        backend.set_options([(item.value, item) for item in backends])
        backend.value = selection.selected_backend
        self._models = models
        self.query_one("#setup-model", Select).set_options(
            [(f"{item.repo_id}  {item.filename}", index) for index, item in enumerate(models)]
        )
        self._show_server(status)
        self._check_runtime(selection.selected_backend)

    def _check_runtime(self, backend: ComputeBackend) -> None:
        def check(advisor: AdvisorService) -> None:
            installation = advisor.managed_llama_cpp(backend)
            self.app.call_from_thread(self._runtime_checked, backend, installation is not None,
                                      installation.version_text if installation else None)

        self._run(check, f"Looking for llama.cpp ({backend.value})...")

    def _runtime_checked(self, backend: ComputeBackend, found: bool, version: str | None) -> None:
        self.query_one("#setup-runtime", Static).update(
            f"llama.cpp {version} ({backend.value}), installed by Jaull." if found else
            f"No llama.cpp installed by Jaull for {backend.value}. Install the pinned build "
            "(downloaded from the official release, SHA-256 checked), or Jaull uses the "
            "llama.cpp it finds on this PC."
        )
        self.query_one("#setup-install", Button).display = not found

    def _install(self, backend: ComputeBackend) -> None:
        def install(advisor: AdvisorService) -> None:
            installation = advisor.install_llama_cpp(backend)
            self.app.call_from_thread(self._runtime_checked, backend, True,
                                      installation.version_text)

        self._run(install, f"Downloading and checking llama.cpp ({backend.value})...")

    def _start(self, choice: _Choice) -> None:
        def start(advisor: AdvisorService) -> None:
            def progress(message: str) -> None:
                self.app.call_from_thread(self._refresh, message)

            artifact = advisor.prepare_serve_artifact(
                choice.repo_id, quantization=choice.quantization, on_progress=progress,
            )
            progress("Sizing the server...")
            plan = advisor.plan_server(
                repo_id=choice.repo_id, artifact=artifact, users=choice.users,
                context_per_user=choice.context, backend=choice.backend,
            )
            lines = plan_lines(plan)
            self.app.call_from_thread(self._output, lines)
            progress("Starting llama-server and waiting until it is ready...")
            started = advisor.start_server(plan, lan=choice.lan, port=choice.port)
            lines += ["", *start_lines(started)]
            notice = advisor.llama_cpp_update_notice()
            if notice is not None:
                lines += ["", f"Note: {notice}"]
            self.app.call_from_thread(self._output, lines)
            # Keep the start output: it holds the API key, which is shown only once.
            self.app.call_from_thread(self._show_server, advisor.server_status(), describe=False)

        self._run(start, "Preparing the model...")

    def _stop(self) -> None:
        def stop(advisor: AdvisorService) -> None:
            stopped = advisor.stop_server()
            self.app.call_from_thread(self._show_server, advisor.server_status())
            self.app.call_from_thread(
                self._output, ["Server stopped." if stopped else "No Jaull server is running."],
            )

        self._run(stop, "Stopping the server...")

    # --------------------------------------------------------------- rendering

    def _output(self, lines: list[str]) -> None:
        self.query_one("#setup-output", Static).update("\n".join(lines))

    def _show_server(self, status: ServerStatus, *, describe: bool = True) -> None:
        self._running = status.running
        if status.running and describe:
            self._output(status_lines(status))
        self._refresh()

    def _refresh(self, status: str = "") -> None:
        running = self._running
        self.query_one("#setup-start", Button).disabled = self._busy or running
        self.query_one("#setup-stop", Button).disabled = self._busy or not running
        self.query_one("#setup-install", Button).disabled = self._busy
        self.query_one("#setup-backend", Select).disabled = self._busy
        self.query_one("#setup-status", Static).update(
            status or ("A Jaull server is running." if running
                       else "Choose a model, then Start server.")
        )

    # ------------------------------------------------------------------ events

    @on(Select.Changed, "#setup-backend")
    def _backend_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, ComputeBackend) and not self._busy:
            self._check_runtime(event.value)

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-back":
            self.action_back()
            return
        if self._busy:
            return
        if event.button.id == "setup-install":
            backend = self._backend()
            if backend is not None:
                self._install(backend)
        elif event.button.id == "setup-stop":
            self._stop()
        elif event.button.id == "setup-start":
            choice = self._choice()
            if isinstance(choice, str):
                self._refresh(choice)
            else:
                self._start(choice)

    def _backend(self) -> ComputeBackend | None:
        value = self.query_one("#setup-backend", Select).value
        return value if isinstance(value, ComputeBackend) else None

    def _choice(self) -> _Choice | str:
        """The choices from the form, or what is missing."""
        numbers: dict[str, int] = {}
        for key, low, high in (("users", 1, None), ("context", 1, None), ("port", 1, 65535)):
            text = self.query_one(f"#setup-{key}", Input).value.strip()
            if not text.isdigit() or int(text) < low or (high is not None and int(text) > high):
                return f"Enter a valid {key}."
            numbers[key] = int(text)
        repo = self.query_one("#setup-repo", Input).value.strip()
        quantization = self.query_one("#setup-quant", Input).value.strip() or None
        if repo:
            try:
                repo_id = normalize_repo_id(repo)
            except InvalidModelReferenceError as exc:
                return str(exc)
        else:
            selected = self.query_one("#setup-model", Select).value
            if not isinstance(selected, int):
                return "Choose a downloaded model or enter a repository."
            model = self._models[selected]
            repo_id = model.repo_id
            quantization = self._advisor().gguf_quantization(model.filename)
        return _Choice(
            repo_id=repo_id, quantization=quantization, backend=self._backend(),
            lan=self.query_one("#setup-lan", RadioButton).value,
            users=numbers["users"], context=numbers["context"], port=numbers["port"],
        )

    def _advisor(self) -> AdvisorService:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app.advisor

    def action_back(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True)
class _Choice:
    repo_id: str
    quantization: str | None
    backend: ComputeBackend | None
    lan: bool
    users: int
    context: int
    port: int


__all__ = ["SetupScreen"]
