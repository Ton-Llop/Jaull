"""Server monitor: what the running llama-server is doing, refreshed every second.

People answered right now and waiting, the speed they get, the shared context pool
and, when the build allows it, how much context each conversation holds. Opened by
Setup after Start server, and from Setup's Monitor button while a server runs.
Every number comes from the server itself (``execution.server_monitor``); nothing
here estimates anything.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Sparkline, Static
from textual.worker import Worker, WorkerState

from jaull.application.serving import policies as serving_policies
from jaull.execution.server_monitor import (
    MonitorError,
    ServerMonitor,
    ServerSnapshot,
    throughput,
)
from jaull.tui import palette
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.bars import bar_markup
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.execution.server_process import ServerState

_REFRESH_SECONDS = 1.0
# One minute of generation speed in the sparkline.
_HISTORY = 60
_BAR = 30


def _gib(count: int) -> str:
    return f"{count / 1024**3:.1f} GiB"


def _tokens(count: int) -> str:
    return f"{count:,}".replace(",", " ")


def _uptime(started_at: str) -> str:
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return "unknown"
    seconds = max(0, int((datetime.now(UTC) - started).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    return f"{hours} h {rest // 60:02d} min" if hours else f"{rest // 60} min {rest % 60:02d} s"


class ServerMonitorScreen(Screen[None]):
    BINDINGS = [("escape", "back", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._monitor: ServerMonitor | None = None
        self._state: ServerState | None = None
        self._last: ServerSnapshot | None = None
        self._history: deque[float] = deque([0.0] * _HISTORY, maxlen=_HISTORY)
        self._reading = False
        self._stopping = False

    def compose(self) -> ComposeResult:
        yield ContextBar("Server monitor", aside="Serve")
        with VerticalScroll(id="monitor-body"):
            yield Static("Connecting to the server...", id="monitor-header",
                         classes="monitor-header")
            with Horizontal(id="monitor-cards"):
                for key, label in (("active", "Answering now"), ("waiting", "Waiting"),
                                   ("generation", "Generation"), ("prompt", "Reading prompts")):
                    with Vertical(classes="monitor-card"):
                        yield Static(label, classes="monitor-card-label")
                        yield Static("-", id=f"monitor-{key}", classes="monitor-card-value")
            yield Static("Generation speed, last minute (tokens/s)", classes="monitor-section")
            yield Sparkline(list(self._history), id="monitor-spark")
            yield Static("", id="monitor-pool", classes="monitor-text")
            yield Static("", id="monitor-slots", classes="monitor-text")
            yield Static("", id="monitor-memory", classes="monitor-text")
            yield Static("", id="monitor-note", classes="monitor-note", markup=False)
        with Horizontal(id="monitor-actions"):
            yield ActionButton("Stop server", id="monitor-stop")
            yield ActionButton("Back", id="monitor-back")
        yield Footer()

    def on_mount(self) -> None:
        advisor = self._advisor()

        def connect() -> None:
            monitor = advisor.server_monitor()
            status = advisor.server_status()
            self.app.call_from_thread(self._connected, monitor, status.state)

        self.run_worker(connect, thread=True, exit_on_error=False, group="monitor-connect")
        self.set_interval(_REFRESH_SECONDS, self._tick)

    def _connected(self, monitor: ServerMonitor | None, state: ServerState | None) -> None:
        if not self.is_mounted:
            return
        self._monitor, self._state = monitor, state
        if monitor is None or state is None:
            self.query_one("#monitor-header", Static).update(
                f"[{palette.WARN}]No Jaull server is running.[/]")
            self.query_one("#monitor-stop", Button).disabled = True
            return
        self._render_header()
        self.query_one("#monitor-note", Static).update(
            "The connection details and the API key stay on the Setup screen: press Esc to see "
            "them. Numbers come from llama-server itself, refreshed every second."
        )

    # ------------------------------------------------------------------ polling

    def _tick(self) -> None:
        if self._monitor is None or self._reading or self._stopping:
            return
        self._reading = True
        monitor = self._monitor

        def read() -> None:
            try:
                snapshot = monitor.read()
            except MonitorError as exc:
                self.app.call_from_thread(self._failed, str(exc))
                return
            self.app.call_from_thread(self._show, snapshot)

        self.run_worker(read, thread=True, exit_on_error=False, group="monitor-read")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group == "monitor-read" and event.state in {
            WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED,
        }:
            self._reading = False

    def _failed(self, message: str) -> None:
        if self.is_mounted:
            self.query_one("#monitor-note", Static).update(f"Not answering: {message}")

    # ---------------------------------------------------------------- rendering

    def _render_header(self) -> None:
        state = self._state
        if state is None:
            return
        self.query_one("#monitor-header", Static).update(
            f"[b {palette.OK}]●[/] [b]{escape(state.alias)}[/]  "
            f"[{palette.INK_2}]{escape(state.url)}/v1 · up {_uptime(state.started_at)}[/]"
        )

    def _show(self, snapshot: ServerSnapshot) -> None:
        if not self.is_mounted:
            return
        self._render_header()
        rates = throughput(self._last, snapshot) if self._last is not None else None
        self._last = snapshot
        if rates is not None:
            self._history.append(rates.generation)
            self.query_one("#monitor-spark", Sparkline).data = list(self._history)
        self._card("active", str(snapshot.requests_processing),
                   palette.OK if snapshot.requests_processing else palette.INK_2)
        self._card("waiting", str(snapshot.requests_deferred),
                   palette.WARN if snapshot.requests_deferred else palette.INK_2)
        self._card("generation", f"{rates.generation:.1f} t/s" if rates else "-", palette.ACCENT)
        self._card("prompt", f"{rates.prompt:.0f} t/s" if rates else "-", palette.INK_2)
        self._render_slots(snapshot)
        self._render_memory(snapshot)
        self.query_one("#monitor-note", Static).update(
            snapshot.slots_note or "The connection details and the API key stay on the Setup "
            "screen: press Esc to see them. Numbers come from llama-server itself."
        )

    def _card(self, key: str, value: str, colour: str) -> None:
        self.query_one(f"#monitor-{key}", Static).update(f"[b {colour}]{value}[/]")

    def _render_slots(self, snapshot: ServerSnapshot) -> None:
        pool = self.query_one("#monitor-pool", Static)
        slots_view = self.query_one("#monitor-slots", Static)
        state = self._state
        if snapshot.slots is None:
            pool.update("")
            slots_view.update("")
            return
        used = sum(slot.used_tokens for slot in snapshot.slots)
        total = state.total_context if state is not None else None
        if total:
            share = used / total
            colour = palette.BAD if share > 0.9 else palette.WARN if share > 0.7 else palette.OK
            pool.update(
                f"[b]Context in use[/]  {bar_markup(min(1.0, share), _BAR)}  "
                f"[{colour}]{_tokens(used)} / {_tokens(total)} tokens ({share:.0%})[/]"
            )
        lines = ["[b]Per conversation[/]"]
        for slot in snapshot.slots:
            cap = slot.n_ctx or 1
            state_text = (f"[{palette.OK}]answering[/]" if slot.busy
                          else f"[{palette.INK_3}]idle[/]")
            lines.append(
                f"  slot {slot.id + 1:<2} {bar_markup(min(1.0, slot.used_tokens / cap), _BAR)}  "
                f"{_tokens(slot.used_tokens):>9} / {_tokens(slot.n_ctx)}  {state_text}"
            )
        slots_view.update("\n".join(lines))

    def _render_memory(self, snapshot: ServerSnapshot) -> None:
        lines = ["[b]Memory[/]"]
        if snapshot.process_ram_bytes is not None:
            lines.append(f"  llama-server RAM   {_gib(snapshot.process_ram_bytes)}")
        available, total = snapshot.ram_available_bytes, snapshot.ram_total_bytes
        if available is not None and total:
            free = available / total
            low = free < serving_policies.SERVE_RAM_LOW_FRACTION
            colour = palette.BAD if low else palette.INK_2
            lines.append(
                f"  System RAM in use  {bar_markup(1 - free, _BAR)}  "
                f"[{colour}]{_gib(available)} free of {_gib(total)}[/]"
            )
            if low:
                lines.append(
                    f"  [{palette.WARN}]! System RAM is running low. Close other programs, or "
                    "restart the server with less context or with the prompt cache off.[/]"
                )
        lines.append(
            f"  VRAM (this server) {_gib(snapshot.process_vram_bytes)}"
            if snapshot.process_vram_bytes is not None else
            f"  [{palette.INK_3}]VRAM per process is only readable on NVIDIA GPUs.[/]"
        )
        self.query_one("#monitor-memory", Static).update("\n".join(lines))

    # ------------------------------------------------------------------ actions

    @on(Button.Pressed, "#monitor-back")
    def _back_pressed(self) -> None:
        self.action_back()

    @on(Button.Pressed, "#monitor-stop")
    def _stop_pressed(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.query_one("#monitor-stop", Button).disabled = True
        self.query_one("#monitor-note", Static).update("Stopping the server...")
        advisor = self._advisor()

        def stop() -> None:
            advisor.stop_server()
            self.app.call_from_thread(self._stopped)

        self.run_worker(stop, thread=True, exit_on_error=False, group="monitor-stop")

    def _stopped(self) -> None:
        if self.is_mounted:
            self.dismiss(None)

    def action_back(self) -> None:
        self.dismiss(None)

    def _advisor(self) -> AdvisorService:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app.advisor


__all__ = ["ServerMonitorScreen"]
