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
from typing import TYPE_CHECKING, Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
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
    from jaull.application.serving import ContextLimit
    from jaull.artifacts.storage import LocalModelFile
    from jaull.domain.hardware import HardwareProfile
    from jaull.domain.runtime import RuntimeBackendSelection
    from jaull.execution.server_process import ServerStatus

_MODEL_HINT = (
    "Not sure which model? Find one with the guided search. The plan you choose there "
    "comes back to this screen."
)


def backend_choices(
    recommended: ComputeBackend,
    alternatives: list[ComputeBackend],
    installed: dict[ComputeBackend, str],
) -> tuple[list[tuple[str, ComputeBackend]], ComputeBackend]:
    """Select options and the default: a backend with llama.cpp already installed wins.

    The detected backend stays the default when it is installed. Otherwise the first
    installed alternative is preselected, so a working build is not skipped.
    """
    backends = list(dict.fromkeys([recommended, *alternatives, ComputeBackend.CPU]))
    default = next(
        (item for item in backends if item in installed and item is recommended),
        next((item for item in backends if item in installed), recommended),
    )
    options = []
    for item in backends:
        notes = [*(["recommended"] if item is default else []),
                 *([f"llama.cpp {installed[item]} installed"] if item in installed else [])]
        options.append((f"{item.value}  ({', '.join(notes)})" if notes else item.value, item))
    return options, default


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
        self._picked: str | None = None
        # Status shown while idle, until the person starts or stops a server.
        self._note: str | None = None
        # A backend change re-checks llama.cpp first, then the context limit.
        self._limit_pending = False
        self._selection: RuntimeBackendSelection | None = None
        self._installed: dict[ComputeBackend, str] = {}

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
            yield Checkbox("Enable the chat page (it asks for the API key)", id="setup-webui")
            yield Checkbox(
                "Share the context: one person can use all of it, several split it",
                id="setup-shared",
            )
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
            yield Static("Choose a model to see the largest context this machine can serve.",
                         id="setup-limit", classes="setup-note", markup=False)
            yield Static("4. Model", classes="setup-heading")
            yield Select[int]([], prompt="Choose a downloaded model", id="setup-model",
                              allow_blank=True)
            with Horizontal(id="setup-model-inputs"):
                yield Input(placeholder="or a Hugging Face repo id", id="setup-repo")
                yield Input(placeholder="quantization, e.g. Q4_K_M", id="setup-quant")
            yield Static(_MODEL_HINT, classes="setup-note", markup=False)
            yield ActionButton("Find a model", id="setup-find")
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
            if self._limit_pending:
                self._limit_pending = False
                self._check_limit()

    def _load(self, advisor: AdvisorService) -> None:
        hardware = advisor.scan_hardware()
        selection = advisor.select_runtime_backend(hardware)
        installed = {}
        for backend in dict.fromkeys([selection.selected_backend,
                                      *(item.backend for item in selection.alternatives),
                                      ComputeBackend.CPU]):
            installation = advisor.managed_llama_cpp(backend)
            if installation is not None:
                installed[backend] = installation.version_text or ""
        models = [item for item, _ in advisor.local_models()]
        status = advisor.server_status()
        self._later(self._loaded, hardware, selection, installed, models, status)

    def _loaded(
        self,
        hardware: HardwareProfile,
        selection: RuntimeBackendSelection,
        installed: dict[ComputeBackend, str],
        models: list[LocalModelFile],
        status: ServerStatus,
    ) -> None:
        accelerator = selection.selected_accelerator
        self.query_one("#setup-hardware", Static).update(
            f"CPU: {hardware.cpu.model or 'unknown'}\n"
            + (f"Accelerator: {accelerator.name}" if accelerator is not None
               else "Accelerator: none. The server runs on the CPU.")
        )
        self._selection, self._installed = selection, installed
        default = self._set_backends()
        self._models = models
        self.query_one("#setup-model", Select).set_options(
            [(f"{item.repo_id}  {item.filename}", index) for index, item in enumerate(models)]
        )
        self._show_server(status)
        self._runtime_checked(default, default in installed, installed.get(default))

    def _set_backends(self) -> ComputeBackend:
        assert self._selection is not None
        options, default = backend_choices(
            self._selection.selected_backend,
            [item.backend for item in self._selection.alternatives],
            self._installed,
        )
        select = self.query_one("#setup-backend", Select)
        current = select.value
        with self.prevent(Select.Changed):
            select.set_options(options)
            select.value = current if isinstance(current, ComputeBackend) else default
        return default

    def _check_runtime(self, backend: ComputeBackend) -> None:
        def check(advisor: AdvisorService) -> None:
            installation = advisor.managed_llama_cpp(backend)
            self._later(self._runtime_checked, backend, installation is not None,
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
            self._later(self._installed_now, backend,
                                      installation.version_text or "")

        self._run(install, f"Downloading and checking llama.cpp ({backend.value})...")

    def _installed_now(self, backend: ComputeBackend, version: str) -> None:
        self._installed = {**self._installed, backend: version}
        self._set_backends()
        self._runtime_checked(backend, True, version)

    def _start(self, choice: _Choice) -> None:
        def start(advisor: AdvisorService) -> None:
            def progress(message: str) -> None:
                self._later(self._refresh, message)

            artifact = advisor.prepare_serve_artifact(
                choice.repo_id, quantization=choice.quantization, on_progress=progress,
            )
            progress("Sizing the server...")
            plan = advisor.plan_server(
                repo_id=choice.repo_id, artifact=artifact, users=choice.users,
                context_per_user=choice.context, backend=choice.backend,
                shared_context=choice.shared_context,
            )
            lines = plan_lines(plan)
            self._later(self._output, lines)
            progress("Starting llama-server and waiting until it is ready...")
            started = advisor.start_server(
                plan, lan=choice.lan, port=choice.port, web_ui=choice.web_ui,
            )
            lines += ["", *start_lines(started)]
            notice = advisor.llama_cpp_update_notice()
            if notice is not None:
                lines += ["", f"Note: {notice}"]
            self._later(self._output, lines)
            # Keep the start output: it holds the API key, which is shown only once.
            self._later(self._show_server, advisor.server_status(), describe=False)

        self._run(start, "Preparing the model...")

    def _stop(self) -> None:
        def stop(advisor: AdvisorService) -> None:
            stopped = advisor.stop_server()
            self._later(self._show_server, advisor.server_status())
            self._later(
                self._output, ["Server stopped." if stopped else "No Jaull server is running."],
            )

        self._run(stop, "Stopping the server...")

    # --------------------------------------------------------------- rendering

    def _later(self, update: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """From a worker thread: run a screen update on the UI thread."""
        self.app.call_from_thread(self._if_mounted, update, *args, **kwargs)

    def _if_mounted(self, update: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Run a worker's screen update unless the person already left the screen."""
        if self.is_mounted:
            update(*args, **kwargs)

    def _output(self, lines: list[str]) -> None:
        self.query_one("#setup-output", Static).update("\n".join(lines))

    def _show_server(self, status: ServerStatus, *, describe: bool = True) -> None:
        self._running = status.running
        if status.running and describe:
            self._output(status_lines(status))
        self._refresh()

    def _refresh(self, status: str = "") -> None:
        if not self.is_mounted:
            return  # A worker can finish after the person left the screen.
        running = self._running
        self.query_one("#setup-start", Button).disabled = self._busy or running
        self.query_one("#setup-stop", Button).disabled = self._busy or not running
        self.query_one("#setup-install", Button).disabled = self._busy
        self.query_one("#setup-find", Button).disabled = self._busy
        self.query_one("#setup-backend", Select).disabled = self._busy
        self.query_one("#setup-status", Static).update(
            status or ("A Jaull server is running." if running
                       else self._note or "Choose a model, then Start server.")
        )

    # ------------------------------------------------------------------ events

    @on(Select.Changed, "#setup-backend")
    def _backend_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, ComputeBackend) and not self._busy:
            self._check_runtime(event.value)
            self._limit_pending = True

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-back":
            self.action_back()
            return
        if self._busy:
            return
        if event.button.id == "setup-find":
            self._find_model()
        elif event.button.id == "setup-install":
            backend = self._backend()
            if backend is not None:
                self._install(backend)
        elif event.button.id == "setup-stop":
            self._note = None
            self._stop()
        elif event.button.id == "setup-start":
            choice = self._choice()
            if isinstance(choice, str):
                self._refresh(choice)
            else:
                self._note = None
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
        model = self._model_reference()
        if isinstance(model, str):
            return model
        repo_id, quantization = model
        return _Choice(
            repo_id=repo_id, quantization=quantization, backend=self._backend(),
            lan=self.query_one("#setup-lan", RadioButton).value,
            users=numbers["users"], context=numbers["context"], port=numbers["port"],
            web_ui=self.query_one("#setup-webui", Checkbox).value,
            shared_context=self.query_one("#setup-shared", Checkbox).value,
        )

    def _model_reference(self) -> tuple[str, str | None] | str:
        """(repo id, quantization) of the chosen model, or what is missing."""
        repo = self.query_one("#setup-repo", Input).value.strip()
        if repo:
            try:
                return (normalize_repo_id(repo),
                        self.query_one("#setup-quant", Input).value.strip() or None)
            except InvalidModelReferenceError as exc:
                return str(exc)
        selected = self.query_one("#setup-model", Select).value
        if not isinstance(selected, int):
            return "Choose a downloaded model or enter a repository."
        model = self._models[selected]
        return model.repo_id, self._advisor().gguf_quantization(model.filename)

    def _check_limit(self) -> None:
        """Show the largest context per person for the chosen model and people at once."""
        if self._busy:
            self._limit_pending = True
            return
        model = self._model_reference()
        users = self.query_one("#setup-users", Input).value.strip()
        if isinstance(model, str) or not users.isdigit() or int(users) < 1:
            return
        repo_id, quantization = model
        count, backend = int(users), self._backend()

        def check(advisor: AdvisorService) -> None:
            limit = advisor.server_context_limit(
                repo_id, quantization=quantization, users=count, backend=backend,
            )
            self._later(self._show_limit, limit)

        self._run(check, f"Looking for the largest context for {repo_id}...")

    def _show_limit(self, limit: ContextLimit) -> None:
        who = "one person" if limit.users == 1 else f"each of {limit.users} people"
        if limit.largest_per_user is None:
            text = f"This model does not fit for serving {who} on this machine."
        else:
            native = f" The model was trained for {limit.native}." if limit.native else ""
            text = (f"Largest context for {who} on this machine: "
                    f"{limit.largest_per_user} tokens.{native}")
        self.query_one("#setup-limit", Static).update(text)

    @on(Select.Changed, "#setup-model")
    def _model_changed(self) -> None:
        self._check_limit()

    @on(Input.Submitted, "#setup-repo, #setup-quant, #setup-users")
    def _model_typed(self) -> None:
        self._check_limit()

    def _find_model(self) -> None:
        """Open the guided search. Choosing a plan there calls :meth:`use_model`."""
        from jaull.tui.app import JaullApp

        app = self.app
        assert isinstance(app, JaullApp)
        app.serve_handoff = self.use_model
        app.start_guided_workflow()

    def use_model(self, repo_id: str, quantization: str | None) -> None:
        self.query_one("#setup-repo", Input).value = repo_id
        self.query_one("#setup-quant", Input).value = quantization or ""
        self.query_one("#setup-model", Select).clear()
        chosen = repo_id + (f" ({quantization})" if quantization else "")
        self._picked = f"Model from the guided search: {chosen}. Press Start server."

    def on_screen_resume(self) -> None:
        from jaull.tui.app import JaullApp

        if isinstance(self.app, JaullApp):
            self.app.serve_handoff = None
        if self._picked is not None:
            self._note, self._picked = self._picked, None
            self._refresh()
            self._check_limit()

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
    web_ui: bool
    shared_context: bool


__all__ = ["SetupScreen"]
