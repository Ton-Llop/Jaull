"""What a running llama-server is doing now, for the server monitor.

Two endpoints, both behind the API key: ``/metrics`` (Prometheus counters, always on)
and ``/slots`` (token counts per slot, only on builds verified to expose no
conversation text, see ``runtime.llama_server.slots_allowed``). HTTP keeps no
connection open, so "people active" means requests being answered right now.

Safeguard: every ``/slots`` answer is checked for text-bearing fields. If one ever
appears (a newer build, a changed format), the monitor stops reading slots for the
rest of the session and says so, instead of holding other people's prompts.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import psutil

from jaull.hardware.nvidia import NvidiaProcessMemorySampler

# Field names that would carry what people typed or received.
_TEXT_FIELDS = frozenset({"prompt", "prompts", "messages", "content", "text", "generated_text"})
_METRIC_RE = re.compile(r"^llamacpp:([a-z_]+)\s+([-+0-9.eE]+|nan|inf)\s*$")
_REQUEST_TIMEOUT_SECONDS = 2.0
_ASLEEP_NOTE = (
    "The model was released from the GPU while nobody used the server. It loads again with the "
    "next request, which takes longer than usual."
)

HttpGet = Callable[[str, dict[str, str]], httpx.Response]


class MonitorError(RuntimeError):
    """The server did not answer the monitor, or the key was refused."""


@dataclass(frozen=True)
class SlotUsage:
    id: int
    #: Context this slot may use (the per-person cap), in tokens.
    n_ctx: int
    #: Tokens this slot holds: its prompt (with the conversation so far) plus the answer.
    used_tokens: int
    busy: bool
    #: The request the slot is (or was last) working on, to tell one answer from the next.
    task: int = -1
    #: Answer tokens produced so far for that request; grows while it is generated.
    decoded: int = 0
    #: Prompt tokens already read for that request.
    prompt_read: int = 0


@dataclass(frozen=True)
class ServerSnapshot:
    #: ``time.monotonic()`` when the snapshot was taken, for rates between two.
    at: float
    requests_processing: int
    requests_deferred: int
    prompt_tokens_total: float
    prompt_seconds_total: float
    predicted_tokens_total: float
    predicted_seconds_total: float
    #: None when /slots is off or was dropped by the safeguard.
    slots: tuple[SlotUsage, ...] | None
    #: Why there are no slots, in words for the person reading the panel.
    slots_note: str | None = None
    #: Resident system RAM of the llama-server process (weights mapped in RAM, KV on the
    #: host, prompt cache). None when the process cannot be read.
    process_ram_bytes: int | None = None
    ram_available_bytes: int | None = None
    ram_total_bytes: int | None = None
    #: Device memory of the process; only NVIDIA (NVML) attributes it per process.
    process_vram_bytes: int | None = None
    #: The server released its model after sitting idle; asking for its slots would load it.
    asleep: bool = False


@dataclass(frozen=True)
class Throughput:
    """Tokens per second between two snapshots: what the people using it see now."""

    generation: float
    prompt: float


class ServerMonitor:
    def __init__(
        self,
        base_url: str,
        api_key_path: Path,
        *,
        slots: bool,
        pid: int | None = None,
        http_get: HttpGet | None = None,
        clock: Callable[[], float] = time.monotonic,
        memory: MemoryReader | None = None,
        sleeps: bool = False,
    ) -> None:
        self._base = base_url.rstrip("/")
        #: The server releases its model when idle, and any /slots request counts as activity:
        #: read it only while something happens, not on every refresh.
        self._sleeps = sleeps
        self._last_slots: tuple[SlotUsage, ...] | None = None
        self._last_counters: tuple[float, float] | None = None
        self._key_path = api_key_path
        self._slots = slots
        self._slots_note: str | None = (
            None if slots else "Per-person context is off for this llama.cpp build."
        )
        self._get = http_get or _http_get
        self._clock = clock
        self._pid = pid
        self._memory = memory or ProcessMemoryReader()

    def read(self) -> ServerSnapshot:
        headers = {"Authorization": f"Bearer {self._key()}"}
        metrics = parse_metrics(self._fetch("/metrics", headers).text)
        slots = None
        # /metrics and /props leave an idle server asleep; /slots wakes it (llama.cpp b11889,
        # measured 2026-11-17), which reloaded the model on every refresh.
        asleep = self._is_asleep(headers)
        counters = (metrics.get("prompt_tokens_total", 0.0),
                    metrics.get("tokens_predicted_total", 0.0))
        worked = int(metrics.get("requests_processing", 0)) > 0 or counters != self._last_counters
        self._last_counters = counters
        if self._slots and not asleep and self._sleeps and not worked and self._last_slots:
            slots = self._last_slots  # nothing has changed: the last figures still hold
        elif self._slots and not asleep:
            try:
                slots = parse_slots(self._fetch("/slots", headers).json())
                self._last_slots = slots
            except SlotsCarryTextError:
                self._slots = False
                self._slots_note = (
                    "This llama.cpp build's /slots includes conversation text, so Jaull "
                    "stopped reading it. Restart the server with the pinned build."
                )
            except ValueError:
                slots = None
        process_ram, process_vram = (
            self._memory.process(self._pid) if self._pid is not None else (None, None)
        )
        available, total = self._memory.system()
        return ServerSnapshot(
            process_ram_bytes=process_ram,
            process_vram_bytes=process_vram,
            ram_available_bytes=available,
            ram_total_bytes=total,
            at=self._clock(),
            requests_processing=int(metrics.get("requests_processing", 0)),
            requests_deferred=int(metrics.get("requests_deferred", 0)),
            prompt_tokens_total=metrics.get("prompt_tokens_total", 0.0),
            prompt_seconds_total=metrics.get("prompt_seconds_total", 0.0),
            predicted_tokens_total=metrics.get("tokens_predicted_total", 0.0),
            predicted_seconds_total=metrics.get("tokens_predicted_seconds_total", 0.0),
            slots=slots,
            slots_note=(
                None if slots is not None else _ASLEEP_NOTE if asleep else self._slots_note
            ),
            asleep=asleep,
        )

    def _is_asleep(self, headers: dict[str, str]) -> bool:
        """Whether the server says it released its model. Not knowing counts as awake."""
        try:
            response = self._get(self._base + "/props", headers)
            if response.status_code != 200:
                return False
            return response.json().get("is_sleeping") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    def _key(self) -> str:
        try:
            return self._key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise MonitorError(f"The server's API key file could not be read: {exc}") from exc

    def _fetch(self, path: str, headers: dict[str, str]) -> httpx.Response:
        try:
            response = self._get(self._base + path, headers)
        except httpx.TimeoutException as exc:
            raise MonitorError(
                f"The server did not answer: {exc}. If it was idle it may be loading the "
                "model again, which can take a minute or two; do not stop it yet."
            ) from exc
        except httpx.HTTPError as exc:
            raise MonitorError(f"The server did not answer: {exc}") from exc
        if response.status_code == 401:
            raise MonitorError("The server refused the API key Jaull stored for it.")
        if response.status_code != 200:
            raise MonitorError(f"{path} answered HTTP {response.status_code}.")
        return response


class MemoryReader(Protocol):
    def process(self, pid: int) -> tuple[int | None, int | None]: ...

    def system(self) -> tuple[int | None, int | None]: ...


class ProcessMemoryReader:
    """Resident RAM of a process, its NVIDIA VRAM when NVML can attribute it, and free RAM."""

    def __init__(self) -> None:
        self._nvidia: NvidiaProcessMemorySampler | None = None

    def process(self, pid: int) -> tuple[int | None, int | None]:
        try:
            ram: int | None = int(psutil.Process(pid).memory_info().rss)
        except (psutil.Error, OSError):
            ram = None
        if self._nvidia is None:
            self._nvidia = NvidiaProcessMemorySampler()
        return ram, self._nvidia.sample_pid_bytes(pid)

    def system(self) -> tuple[int | None, int | None]:
        try:
            memory = psutil.virtual_memory()
        except (psutil.Error, OSError):
            return None, None
        return int(memory.available), int(memory.total)


class SlotsCarryTextError(ValueError):
    """A /slots answer contained a field that would hold conversation text."""


def parse_metrics(text: str) -> dict[str, float]:
    """``llamacpp:<name> <value>`` lines into ``{name: value}``; comments are skipped."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC_RE.match(line.strip())
        if match is not None:
            values[match.group(1)] = float(match.group(2))
    return values


def parse_slots(payload: Any) -> tuple[SlotUsage, ...]:
    if not isinstance(payload, list):
        raise ValueError("/slots did not answer a list.")
    if _has_text_field(payload):
        raise SlotsCarryTextError("/slots carries conversation text.")
    slots: list[SlotUsage] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        decoded = sum(
            int(token.get("n_decoded", 0))
            for token in item.get("next_token", []) if isinstance(token, dict)
        )
        slots.append(SlotUsage(
            id=int(item.get("id", len(slots))),
            n_ctx=int(item.get("n_ctx", 0)),
            used_tokens=int(item.get("n_prompt_tokens", 0)) + decoded,
            busy=bool(item.get("is_processing", False)),
            task=int(item.get("id_task", -1)),
            decoded=decoded,
            prompt_read=int(item.get("n_prompt_tokens_processed", 0)),
        ))
    return tuple(slots)


def throughput(before: ServerSnapshot, after: ServerSnapshot) -> Throughput:
    """Tokens per second of wall time between two snapshots, all slots together.

    llama-server adds to its /metrics token counters only when a request finishes, so
    while answers are being generated those stay still. The slots move live, so they
    are used when available; without them the counters give a rate that updates as
    requests finish.
    """
    elapsed = after.at - before.at
    if elapsed <= 0:
        return Throughput(generation=0.0, prompt=0.0)
    if before.slots is not None and after.slots is not None:
        generated, read = _slot_progress(before.slots, after.slots)
    else:
        generated = max(0.0, after.predicted_tokens_total - before.predicted_tokens_total)
        read = max(0.0, after.prompt_tokens_total - before.prompt_tokens_total)
    return Throughput(generation=generated / elapsed, prompt=read / elapsed)


def _slot_progress(
    before: tuple[SlotUsage, ...], after: tuple[SlotUsage, ...],
) -> tuple[float, float]:
    """Answer and prompt tokens produced between two slot readings."""
    previous = {slot.id: slot for slot in before}
    generated = read = 0
    for slot in after:
        old = previous.get(slot.id)
        if old is not None and old.task == slot.task:
            generated += max(0, slot.decoded - old.decoded)
            read += max(0, slot.prompt_read - old.prompt_read)
        elif slot.task != -1:
            # A request that started in between: everything it did is new.
            generated += slot.decoded
            read += slot.prompt_read
    return float(generated), float(read)


def _has_text_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in _TEXT_FIELDS or _has_text_field(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_text_field(item) for item in value)
    return False


def _http_get(url: str, headers: dict[str, str]) -> httpx.Response:
    # Local probe of our own server; a self-signed certificate is expected with TLS.
    return httpx.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS, verify=False)


__all__ = [
    "MemoryReader",
    "MonitorError",
    "ProcessMemoryReader",
    "ServerMonitor",
    "ServerSnapshot",
    "SlotUsage",
    "SlotsCarryTextError",
    "Throughput",
    "parse_metrics",
    "parse_slots",
    "throughput",
]
