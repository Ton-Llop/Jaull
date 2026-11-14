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

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
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
from jaull.tui import palette
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.bars import bar_markup
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.application.serving import ContextLimit
    from jaull.application.serving.models import FoundLlamaServer
    from jaull.artifacts.storage import LocalModelFile
    from jaull.domain.hardware import HardwareProfile
    from jaull.domain.runtime import RuntimeBackendSelection
    from jaull.execution.server_process import ServerStatus

_MODEL_HINT = (
    "Not sure which model? Find one with the guided search. The plan you choose there "
    "comes back to this screen."
)

_STEPS = ("Machine", "llama.cpp", "Server", "Model", "Start")
_TITLES = ("Machine and backend", "llama.cpp", "Server", "Model", "Start")

_EXPOSURE_NOTE = {
    False: "Only programs on this PC can connect.",
    True: "Other devices on your network can connect with the API key. "
          "Jaull does not open the firewall for you.",
}

# Quick picks for the context per person, in tokens.
_PRESETS = (("8K", 8192), ("16K", 16384), ("32K", 32768), ("64K", 65536), ("128K", 131072))

# Step states, each with a glyph and a colour so the meaning survives without colour.
_DONE, _ACTIVE, _WARN, _PENDING = "done", "active", "warn", "pending"
_GLYPHS = {
    _DONE: ("✓", palette.OK),
    _ACTIVE: ("●", palette.ACCENT),
    _WARN: ("!", palette.WARN),
    _PENDING: ("○", palette.INK_3),
}
_BAR_WIDTH = 28
# Typing pause before the largest context is worked out again (UI timing only).
_LIMIT_CHECK_DELAY_SECONDS = 0.8
# (repo id, quantization, people at once, backend)
_LimitKey = tuple[str, "str | None", int, "ComputeBackend | None"]


def _tokens(count: int) -> str:
    """32768 -> '32 768', grouped so long contexts read at a glance."""
    return f"{count:,}".replace(",", " ")


def _styled_output(lines: list[str]) -> str:
    """The plain output lines, coloured by role. The text itself is unchanged."""
    styled: list[str] = []
    key_next = False
    for line in lines:
        text = escape(line)
        if key_next and line.startswith("  "):
            styled.append(f"[b {palette.WARN} on {palette.WARN_TINT}] {text.strip()} [/]")
            key_next = False
            continue
        key_next = line.startswith("API key (shown once")
        if line.startswith(("Ready at", "On the local network", "Chat page")):
            styled.append(f"[b {palette.OK}]{text}[/]")
        elif key_next:
            styled.append(f"[b {palette.WARN}]{text}[/]")
        elif line.startswith("Serving "):
            styled.append(f"[b {palette.INK}]{text}[/]")
        elif line.startswith(("Start script", "Stop it with", "API key:", "  URL", "  PID",
                              "  Log", "Server stopped", "No Jaull server")):
            styled.append(f"[{palette.INK_2}]{text}[/]")
        elif line.startswith(("Jaull does not change the firewall", "Note:")) or (
            line.startswith("  sudo")
        ):
            styled.append(f"[{palette.WARN}]{text}[/]")
        else:
            styled.append(text)
    return "\n".join(styled)


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
    # Styles live in styles.tcss (section "Setup"), next to the tokens they use.

    def __init__(self) -> None:
        super().__init__()
        self._busy = True
        self._running = False
        self._models: list[LocalModelFile] = []
        self._worker: Worker[None] | None = None
        self._picked: str | None = None
        # The last failure, shown while idle until the next action the person starts.
        self._error: str | None = None
        # Status shown while idle, until the person starts or stops a server.
        self._note: str | None = None
        # A backend change re-checks llama.cpp first, then the context limit.
        self._limit_pending = False
        self._selection: RuntimeBackendSelection | None = None
        self._installed: dict[ComputeBackend, str] = {}
        self._limit: ContextLimit | None = None
        # (repo, quantization, people, backend) the limit above was worked out for,
        # and the one last asked for: a limit for other inputs is never shown.
        self._limit_for: _LimitKey | None = None
        self._limit_asked: _LimitKey | None = None
        self._limit_timer: Timer | None = None
        # Per backend: the llama-server found on this PC when Jaull installed none.
        self._found: dict[ComputeBackend, FoundLlamaServer | None] = {}

    def compose(self) -> ComposeResult:
        yield ContextBar("Set up a local server", aside="Setup")
        yield Static("", id="setup-steps", classes="setup-steps")
        with VerticalScroll(id="setup-body"):
            with Vertical(classes="setup-section"):
                yield from self._heading(1)
                yield Static("Scanning hardware...", id="setup-hardware", classes="setup-text")
                yield Select[ComputeBackend]([], prompt="Backend", id="setup-backend",
                                             allow_blank=True)
            with Vertical(classes="setup-section"):
                yield from self._heading(2)
                yield Static("", id="setup-runtime", classes="setup-text", markup=False)
                yield ActionButton("Install llama.cpp", id="setup-install", disabled=True)
            with Vertical(classes="setup-section"):
                yield from self._heading(3)
                with RadioSet(id="setup-exposure"):
                    yield RadioButton("Only this PC", value=True, id="setup-local")
                    yield RadioButton("Share on the local network", id="setup-lan")
                yield Static(_EXPOSURE_NOTE[False], id="setup-exposure-note",
                             classes="setup-note", markup=False)
                yield Checkbox("Enable the chat page (it asks for the API key)",
                               id="setup-webui")
                yield Checkbox(
                    "Share the context: one person can use all of it, several split it",
                    id="setup-shared",
                )
                with Horizontal(id="setup-numbers"):
                    yield from self._field("People at once", "setup-users",
                                           serving_policies.SERVE_DEFAULT_USERS,
                                           "chats served at the same time")
                    yield from self._field("Context per person", "setup-context",
                                           serving_policies.SERVE_DEFAULT_CONTEXT_PER_USER,
                                           "tokens each chat remembers")
                    yield from self._field("Port", "setup-port",
                                           runtime_policies.LLAMA_SERVER_DEFAULT_PORT,
                                           "change it if 8090 is taken")
                with Horizontal(id="setup-presets"):
                    yield Static("Context", classes="setup-presets-label")
                    for label, tokens in _PRESETS:
                        yield Button(label, id=f"setup-preset-{tokens}", classes="chip")
                    yield Button("Max", id="setup-preset-max", classes="chip", disabled=True)
                yield Static("", id="setup-capacity", classes="setup-capacity")
                yield Static("Choose a model to see the largest context this machine can serve.",
                             id="setup-limit", classes="setup-note", markup=False)
            with Vertical(classes="setup-section"):
                yield from self._heading(4)
                yield Select[int]([], prompt="Choose a downloaded model", id="setup-model",
                                  allow_blank=True)
                yield Button("Use another model instead", id="setup-other", classes="chip")
                with Vertical(id="setup-download"):
                    yield Static("or download one", classes="setup-or")
                    with Horizontal(id="setup-model-inputs"):
                        yield Input(placeholder="Hugging Face repo id", id="setup-repo")
                        yield Input(placeholder="quantization, e.g. Q4_K_M", id="setup-quant")
                yield Static(_MODEL_HINT, classes="setup-note", markup=False)
                yield ActionButton("Find a model", id="setup-find")
            with Vertical(classes="setup-section -last", id="setup-result"):
                yield from self._heading(5)
                yield Static("", id="setup-output", classes="setup-output -empty")
        yield Static("Reading this machine...", id="setup-status", classes="status-line")
        with Horizontal(id="setup-actions"):
            yield Button("Start server", id="setup-start", classes="-primary", disabled=True)
            yield ActionButton("Stop server", id="setup-stop", disabled=True)
            yield ActionButton("Back", id="setup-back")
        yield Footer()

    @staticmethod
    def _heading(step: int) -> ComposeResult:
        """A numbered title with the step's state on the right."""
        with Horizontal(classes="setup-heading-row"):
            yield Static(f"[b {palette.ACCENT}]{step:02d}[/]  [b]{_TITLES[step - 1]}[/]",
                         classes="setup-title")
            yield Static("", id=f"setup-state-{step}", classes="setup-state")

    @staticmethod
    def _field(label: str, field_id: str, default: int, help_text: str) -> ComposeResult:
        with Vertical(classes="setup-field"):
            yield Label(label)
            yield Input(str(default), type="integer", id=field_id)
            yield Static(help_text, classes="setup-help", markup=False)

    def on_mount(self) -> None:
        self.query_one("#setup-other", Button).display = False
        self._run(self._load, "Reading this machine...")

    # ------------------------------------------------------------------ workers

    def _run(
        self, work: Callable[[AdvisorService], None], status: str, *, keeps_error: bool = False,
    ) -> None:
        self._busy = True
        if not keeps_error:
            self._error = None  # a new action starts afresh
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
            self._limit_asked = None  # A failed limit check may be retried.
            self._error = f"Failed: {event.worker.error}"
            self._refresh()
            self._run_pending_limit_check()
        elif event.state in {WorkerState.SUCCESS, WorkerState.CANCELLED} and self._busy:
            self._busy = False
            self._refresh()
            self._run_pending_limit_check()

    def _run_pending_limit_check(self) -> None:
        """Inputs changed while a worker ran: work out their limit now, success or not."""
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
        _, default = backend_choices(
            selection.selected_backend, [item.backend for item in selection.alternatives],
            installed,
        )
        if default not in installed:
            self._found[default] = advisor.found_llama_server(default, selection)
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
        label = f"[{palette.INK_2}]{{:<13}}[/]"
        self.query_one("#setup-hardware", Static).update(
            label.format("CPU") + escape(hardware.cpu.model or "unknown") + "\n"
            + label.format("Accelerator")
            + (f"[b]{escape(accelerator.name)}[/]" if accelerator is not None
               else f"[{palette.WARN}]none[/], the server runs on the CPU")
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
        selection = self._selection

        def check(advisor: AdvisorService) -> None:
            installation = advisor.managed_llama_cpp(backend)
            if installation is None and backend not in self._found:
                self._found[backend] = advisor.found_llama_server(backend, selection)
            self._later(self._runtime_checked, backend, installation is not None,
                        installation.version_text if installation else None)

        self._run(check, f"Looking for llama.cpp ({backend.value})...")

    def _runtime_checked(self, backend: ComputeBackend, found: bool, version: str | None) -> None:
        local = self._found.get(backend)
        if found:
            text = f"llama.cpp {version} ({backend.value}), installed by Jaull."
        elif local is not None and local.supports_backend:
            version_note = f" ({local.version})" if local.version else ""
            text = (f"Using the llama.cpp found on this PC{version_note}:"
                    f"\n{local.path}\nYou can also install the pinned build for {backend.value}, "
                    "the one Jaull was tested with (SHA-256 checked).")
        elif local is not None:
            text = (f"The llama.cpp found on this PC does not run {backend.value}:"
                    f"\n{local.path}\nInstall the pinned build for {backend.value} "
                    "(SHA-256 checked), or choose another backend.")
        else:
            text = (f"No llama.cpp found for {backend.value}. Install the pinned build "
                    "(downloaded from the official release, SHA-256 checked).")
        self.query_one("#setup-runtime", Static).update(text)
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
        output = self.query_one("#setup-output", Static)
        output.update(_styled_output(lines))
        output.set_class(not lines, "-empty")
        if lines:
            # The result holds the URL and the key: bring it into view.
            self.call_after_refresh(output.scroll_visible)

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
        self.query_one("#setup-result").set_class(running, "-running")
        idle = ("Ready. Press Start server." if not isinstance(self._model_reference(), str)
                else "Choose a model, then Start server.")
        text = status or ("A Jaull server is running." if running
                          else self._error or self._note or idle)
        mark = (f"[{palette.ACCENT}]●[/] " if self._busy
                else f"[{palette.OK}]●[/] " if running else "")
        self.query_one("#setup-status", Static).update(mark + escape(text))
        self._render_progress()

    # ---------------------------------------------------------- live summary

    def _step_states(self) -> list[tuple[str, str]]:
        """(state, short note) for each of the five steps, from what is on screen now."""
        loaded = self._selection is not None
        backend = self._backend()
        numbers = self._numbers()
        limit = self._current_limit()
        over = (isinstance(numbers, dict) and limit is not None
                and limit.largest_per_user is not None
                and numbers["context"] > limit.largest_per_user)
        model = self._model_reference()
        local = self._found.get(backend) if backend is not None else None
        states = [
            (_DONE, "ready") if loaded else (_PENDING, "scanning"),
            (_PENDING, "") if not loaded
            else (_DONE, f"{self._installed[backend]}".strip() or "installed")
            if backend in self._installed
            else (_DONE, "found on this PC") if local is not None and local.supports_backend
            else (_WARN, f"no {backend.value if backend else ''} build") if local is not None
            else (_WARN, "not installed"),
            (_WARN, numbers) if isinstance(numbers, str)
            else (_WARN, "context too large") if over
            else (_DONE, "this PC" if not self._lan() else "local network"),
            (_PENDING, "choose one") if isinstance(model, str)
            # The check found no context at all that fits: nothing to start.
            else (_WARN, "does not fit here")
            if limit is not None and limit.largest_per_user is None
            else (_DONE, model[0].split("/")[-1]),
        ]
        if self._running:
            states.append((_DONE, "running"))
        elif self._busy:
            states.append((_ACTIVE, "working"))
        elif all(state == _DONE for state, _ in states):
            states.append((_ACTIVE, "ready to start"))
        else:
            states.append((_PENDING, ""))
        return states

    def _render_progress(self) -> None:
        states = self._step_states()
        # The first step that still needs something is the one to look at.
        current = next((i for i, (state, _) in enumerate(states) if state != _DONE), None)
        parts = []
        for index, (name, (state, note)) in enumerate(zip(_STEPS, states, strict=True)):
            shown = _ACTIVE if index == current and state == _PENDING else state
            glyph, colour = _GLYPHS[shown]
            weight = "b " if index == current else ""
            parts.append(f"[{weight}{colour}]{glyph} {name}[/]")
            heading = self.query_one(f"#setup-state-{index + 1}", Static)
            heading.update(f"[{colour}]{glyph} {escape(note)}[/]" if note else "")
        self.query_one("#setup-steps", Static).update(
            f" [{palette.LINE_2}]──[/] ".join(parts)
        )
        self._render_capacity()

    def _render_capacity(self) -> None:
        """Requested context per person against the largest that fits, as a bar."""
        numbers = self._numbers()
        context = numbers["context"] if isinstance(numbers, dict) else None
        for _label, tokens in _PRESETS:
            self.query_one(f"#setup-preset-{tokens}", Button).set_class(
                context == tokens, "-active")
        limit = self._current_limit()
        largest = limit.largest_per_user if limit is not None else None
        maximum = self.query_one("#setup-preset-max", Button)
        maximum.disabled = largest is None
        maximum.set_class(largest is not None and context == largest, "-active")
        capacity = self.query_one("#setup-capacity", Static)
        if context is None:
            capacity.update(f"[{palette.WARN}]! Enter the context as a whole number of tokens.[/]")
            return
        asked = f"[b]{_tokens(context)}[/] tokens per person"
        if largest is None:
            capacity.update(asked)  # the note below says to choose a model first
            return
        ratio = context / largest if largest else 1.0
        if context > largest:
            verdict = f"[b {palette.BAD}]✗ more than this machine can serve[/]"
            bar = f"[{palette.BAD}]{'━' * _BAR_WIDTH}[/]"
        else:
            verdict = f"[{palette.OK}]✓ fits[/]"
            bar = bar_markup(ratio, _BAR_WIDTH)
        capacity.update(f"{asked}  {bar}  [{palette.INK_2}]max {_tokens(largest)}[/]  {verdict}")

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
        if (event.button.id or "").startswith("setup-preset-"):
            self._preset(event.button.id or "")
            return
        if event.button.id == "setup-other":
            self.query_one("#setup-model", Select).clear()  # shows the download fields again
            self.query_one("#setup-repo", Input).focus()
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

    def _numbers(self) -> dict[str, int] | str:
        """People, context and port from the form, or what is wrong with them."""
        numbers: dict[str, int] = {}
        for key, low, high in (("users", 1, None), ("context", 1, None), ("port", 1, 65535)):
            text = self.query_one(f"#setup-{key}", Input).value.strip()
            if not text.isdigit() or int(text) < low or (high is not None and int(text) > high):
                return f"Enter a valid {key}."
            numbers[key] = int(text)
        return numbers

    def _lan(self) -> bool:
        return self.query_one("#setup-lan", RadioButton).value

    def _choice(self) -> _Choice | str:
        """The choices from the form, or what is missing."""
        numbers = self._numbers()
        if isinstance(numbers, str):
            return numbers
        model = self._model_reference()
        if isinstance(model, str):
            return model
        repo_id, quantization = model
        return _Choice(
            repo_id=repo_id, quantization=quantization, backend=self._backend(),
            lan=self._lan(),
            users=numbers["users"], context=numbers["context"], port=numbers["port"],
            web_ui=self.query_one("#setup-webui", Checkbox).value,
            shared_context=self.query_one("#setup-shared", Checkbox).value,
        )

    def _model_reference(self) -> tuple[str, str | None] | str:
        """(repo id, quantization) of the chosen model, or what is missing."""
        # The downloaded model on screen wins: the typed fields are hidden while
        # one is chosen, so they must never decide what starts.
        selected = self.query_one("#setup-model", Select).value
        if isinstance(selected, int):
            model = self._models[selected]
            return model.repo_id, self._advisor().gguf_quantization(model.filename)
        repo = self.query_one("#setup-repo", Input).value.strip()
        if not repo:
            return "Choose a downloaded model or enter a repository."
        try:
            return (normalize_repo_id(repo),
                    self.query_one("#setup-quant", Input).value.strip() or None)
        except InvalidModelReferenceError as exc:
            return str(exc)

    def _limit_key(self) -> _LimitKey | None:
        """What the largest context depends on, or None while the form is incomplete."""
        model = self._model_reference()
        users = self.query_one("#setup-users", Input).value.strip()
        if isinstance(model, str) or not users.isdigit() or int(users) < 1:
            return None
        return model[0], model[1], int(users), self._backend()

    def _current_limit(self) -> ContextLimit | None:
        """The worked-out limit, only while it still matches what is on screen."""
        if self._limit is None or self._limit_for != self._limit_key():
            return None
        return self._limit

    def _schedule_limit_check(self) -> None:
        """Re-check the limit shortly after typing stops, so no Enter is needed."""
        if self._limit_timer is not None:
            self._limit_timer.stop()
        key = self._limit_key()
        if key is None or key == self._limit_asked:
            return
        self.query_one("#setup-limit", Static).update("Working out the largest context...")
        self._limit_timer = self.set_timer(_LIMIT_CHECK_DELAY_SECONDS, self._check_limit)

    def _check_limit(self) -> None:
        """Show the largest context per person for the chosen model and people at once."""
        if self._busy:
            self._limit_pending = True
            return
        key = self._limit_key()
        if key is None or key == self._limit_asked:
            return
        repo_id, quantization, count, backend = key
        self._limit_asked = key

        def check(advisor: AdvisorService) -> None:
            limit = advisor.server_context_limit(
                repo_id, quantization=quantization, users=count, backend=backend,
            )
            self._later(self._show_limit, limit, key)

        # A limit check runs on its own as the form changes: it must not hide the error
        # of the action the person pressed.
        self._run(check, f"Looking for the largest context for {repo_id}...", keeps_error=True)

    def _show_limit(self, limit: ContextLimit, key: _LimitKey) -> None:
        self._limit, self._limit_for = limit, key
        self._render_progress()
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
        chosen = isinstance(self.query_one("#setup-model", Select).value, int)
        if chosen:
            for field in ("#setup-repo", "#setup-quant"):
                self.query_one(field, Input).value = ""
        self._show_download(not chosen)
        self._check_limit()
        self._render_progress()

    def _show_download(self, show: bool) -> None:
        """A downloaded model hides the download fields; one button brings them back."""
        self.query_one("#setup-download").display = show
        self.query_one("#setup-other", Button).display = not show

    @on(Input.Changed)
    @on(Checkbox.Changed)
    def _form_changed(self) -> None:
        """Every keystroke updates the step bar and the capacity bar; nothing slow runs."""
        if self.is_mounted:
            self._render_progress()

    @on(RadioSet.Changed, "#setup-exposure")
    def _exposure_changed(self) -> None:
        self.query_one("#setup-exposure-note", Static).update(_EXPOSURE_NOTE[self._lan()])
        self._render_progress()

    def _preset(self, button_id: str) -> None:
        """A quick pick fills the context field; Max uses the largest that fits."""
        if button_id == "setup-preset-max":
            limit = self._current_limit()
            if limit is None or limit.largest_per_user is None:
                return
            tokens = limit.largest_per_user
        else:
            tokens = int(button_id.removeprefix("setup-preset-"))
        self.query_one("#setup-context", Input).value = str(tokens)

    @on(Input.Submitted, "#setup-repo, #setup-quant, #setup-users")
    def _model_typed(self) -> None:
        self._check_limit()

    @on(Input.Changed, "#setup-repo, #setup-quant, #setup-users")
    def _limit_inputs_changed(self) -> None:
        if self.is_mounted:
            self._schedule_limit_check()

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
