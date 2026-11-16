"""Server monitor: live counters from llama-server, and never the conversations.

The /metrics and /slots samples below are the shapes b11889 answered on 2026-11-16
(SmolLM2 on the CPU build, two slots, one request running).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from textual.widgets import Button, Static

from jaull.execution.server_monitor import (
    MonitorError,
    ServerMonitor,
    ServerSnapshot,
    SlotsCarryTextError,
    parse_metrics,
    parse_slots,
    throughput,
)
from jaull.execution.server_process import ServerState, ServerStatus
from jaull.runtime import policies
from jaull.runtime.llama_server import slots_allowed
from jaull.tui.app import JaullApp
from jaull.tui.screens.server_monitor import ServerMonitorScreen
from tests.test_tui_setup import _wait_until

METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 34
llamacpp:prompt_seconds_total 0.055804
llamacpp:tokens_predicted_total 7
llamacpp:tokens_predicted_seconds_total 0.037394
llamacpp:n_tokens_max 273
llamacpp:predicted_tokens_seconds 160.454
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
"""

SLOT_BUSY: dict[str, Any] = {
    "id": 0, "n_ctx": 4096, "speculative": False, "is_processing": True, "id_task": 0,
    "n_prompt_tokens": 268, "n_prompt_tokens_processed": 43, "n_prompt_tokens_cache": 0,
    "params": {"temperature": 0.8, "max_tokens": 600,
               "generation_prompt": "<|im_start|>assistant\n", "lora": []},
    "next_token": [{"has_next_token": True, "has_new_line": True, "n_remain": 375,
                    "n_decoded": 225}],
}
SLOT_IDLE: dict[str, Any] = {**SLOT_BUSY, "id": 1, "is_processing": False,
                             "n_prompt_tokens": 0, "next_token": [{"n_decoded": 0}]}


def test_metrics_lines_are_read_and_comments_skipped() -> None:
    values = parse_metrics(METRICS)

    assert values["requests_processing"] == 1
    assert values["tokens_predicted_total"] == 7
    assert "prompt_tokens_total" in values and len(values) == 8


def test_slots_give_context_in_use_per_conversation() -> None:
    busy, idle = parse_slots([SLOT_BUSY, SLOT_IDLE])

    assert (busy.id, busy.n_ctx, busy.used_tokens, busy.busy) == (0, 4096, 268 + 225, True)
    assert (idle.used_tokens, idle.busy) == (0, False)


@pytest.mark.parametrize("field", ["prompt", "messages", "content"])
def test_slots_with_conversation_text_are_refused(field: str) -> None:
    leaky = {**SLOT_BUSY, field: "what someone typed"}

    with pytest.raises(SlotsCarryTextError):
        parse_slots([leaky])


def _response(status: int, *, text: str = "", payload: Any = None) -> httpx.Response:
    request = httpx.Request("GET", "http://x")
    if payload is not None:
        return httpx.Response(status, json=payload, request=request)
    return httpx.Response(status, text=text, request=request)


def _monitor(tmp_path: Path, routes: dict[str, httpx.Response], *, slots: bool = True,
             seen: list[dict[str, str]] | None = None) -> ServerMonitor:
    key = tmp_path / "api-key"
    key.write_text("the-key\n")

    def get(url: str, headers: dict[str, str]) -> httpx.Response:
        if seen is not None:
            seen.append(headers)
        return routes[url.rsplit("/", 1)[1]]

    return ServerMonitor("http://127.0.0.1:8090/", key, slots=slots, http_get=get,
                         clock=lambda: 10.0)


def test_read_uses_the_stored_key_and_combines_both_endpoints(tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []
    monitor = _monitor(tmp_path, {"metrics": _response(200, text=METRICS),
                                  "slots": _response(200, payload=[SLOT_BUSY, SLOT_IDLE])},
                       seen=seen)

    snapshot = monitor.read()

    assert seen[0] == {"Authorization": "Bearer the-key"}
    assert snapshot.requests_processing == 1 and snapshot.requests_deferred == 0
    assert snapshot.slots is not None and len(snapshot.slots) == 2
    assert snapshot.slots_note is None


def test_a_leaky_slots_answer_switches_slots_off_for_good(tmp_path: Path) -> None:
    routes = {"metrics": _response(200, text=METRICS),
              "slots": _response(200, payload=[{**SLOT_BUSY, "prompt": "secret"}])}
    monitor = _monitor(tmp_path, routes)

    first = monitor.read()
    routes["slots"] = _response(200, payload=[SLOT_BUSY])
    second = monitor.read()

    assert first.slots is None and second.slots is None
    assert first.slots_note is not None and "conversation text" in first.slots_note


def test_without_slots_only_aggregates_are_shown(tmp_path: Path) -> None:
    snapshot = _monitor(tmp_path, {"metrics": _response(200, text=METRICS)}, slots=False).read()

    assert snapshot.slots is None
    assert snapshot.slots_note == "Per-person context is off for this llama.cpp build."


def test_a_refused_key_is_a_monitor_error(tmp_path: Path) -> None:
    monitor = _monitor(tmp_path, {"metrics": _response(401)})

    with pytest.raises(MonitorError, match="refused the API key"):
        monitor.read()


def _snapshot(at: float, predicted: float, prompt: float) -> ServerSnapshot:
    return ServerSnapshot(at=at, requests_processing=0, requests_deferred=0,
                          prompt_tokens_total=prompt, prompt_seconds_total=0,
                          predicted_tokens_total=predicted, predicted_seconds_total=0,
                          slots=None)


def test_throughput_is_tokens_over_wall_time_between_snapshots() -> None:
    rates = throughput(_snapshot(10.0, 100, 1000), _snapshot(12.0, 160, 1400))

    assert rates.generation == pytest.approx(30.0)
    assert rates.prompt == pytest.approx(200.0)
    restarted = throughput(_snapshot(10.0, 500, 0), _snapshot(11.0, 20, 0))
    assert restarted.generation == 0.0


def _slots(*, task: int, decoded: int, read: int) -> tuple[Any, ...]:
    busy = {**SLOT_BUSY, "id_task": task, "n_prompt_tokens_processed": read,
            "next_token": [{"n_decoded": decoded}]}
    return parse_slots([busy, SLOT_IDLE])


def test_live_speed_comes_from_the_slots_while_counters_stay_still() -> None:
    def snap(at: float, slots: tuple[Any, ...]) -> ServerSnapshot:
        return ServerSnapshot(at=at, requests_processing=1, requests_deferred=0,
                              prompt_tokens_total=0, prompt_seconds_total=0,
                              predicted_tokens_total=0, predicted_seconds_total=0, slots=slots)

    same = throughput(snap(1.0, _slots(task=7, decoded=100, read=268)),
                      snap(2.0, _slots(task=7, decoded=130, read=268)))
    assert (same.generation, same.prompt) == (30.0, 0.0)
    # A new request started in between: all its progress counts, the old one's does not.
    new = throughput(snap(1.0, _slots(task=7, decoded=600, read=268)),
                     snap(2.0, _slots(task=8, decoded=12, read=40)))
    assert (new.generation, new.prompt) == (12.0, 40.0)


def test_slots_are_only_allowed_on_a_verified_build() -> None:
    assert slots_allowed(policies.LLAMA_CPP_PINNED_BUILD)
    assert not slots_allowed("b13000")
    assert not slots_allowed(None)


# ---------------------------------------------------------------- the screen

STATE = ServerState(
    pid=1, create_time=1.0, host="127.0.0.1", port=8090, alias="Qwen3.6-35B-A3B-GGUF",
    argv=["llama-server"], log_path="/x", started_at="2026-11-16T10:00:00+00:00",
    total_context=8192, parallel=2, context_per_user=4096, slots=True,
)


class FakeMonitor:
    def __init__(self) -> None:
        self.reads = 0

    def read(self) -> ServerSnapshot:
        # One answer growing by 20 tokens a second; the /metrics counters stay still,
        # as llama-server's do until a request finishes.
        self.reads += 1
        busy = {**SLOT_BUSY, "next_token": [{"n_decoded": 20 * self.reads}]}
        return ServerSnapshot(
            at=float(self.reads), requests_processing=1, requests_deferred=2,
            prompt_tokens_total=0, prompt_seconds_total=0,
            predicted_tokens_total=0, predicted_seconds_total=0,
            slots=parse_slots([busy, SLOT_IDLE]),
        )


class MonitorAdvisor:
    services = None

    def __init__(self) -> None:
        self.monitor = FakeMonitor()
        self.stopped = False

    def server_monitor(self) -> FakeMonitor:
        return self.monitor

    def server_status(self) -> ServerStatus:
        return ServerStatus(state=STATE, running=not self.stopped, healthy=True)

    def stop_server(self) -> bool:
        self.stopped = True
        return True


def _text(screen: ServerMonitorScreen, selector: str) -> str:
    return str(screen.query_one(selector, Static).render())


@pytest.mark.parametrize("size", [(80, 24), (160, 48)])
def test_the_monitor_shows_people_speed_and_context(size: tuple[int, int]) -> None:
    async def scenario() -> None:
        advisor = MonitorAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = ServerMonitorScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: advisor.monitor.reads >= 2)
            await pilot.pause()

            assert "Qwen3.6-35B-A3B-GGUF" in _text(screen, "#monitor-header")
            assert _text(screen, "#monitor-active") == "1"
            assert _text(screen, "#monitor-waiting") == "2"
            assert _text(screen, "#monitor-generation") == "20.0 t/s"
            assert re.search(r"\d+ / 8 192 tokens \(\d+%\)", _text(screen, "#monitor-pool"))
            slots = _text(screen, "#monitor-slots")
            assert "slot 1" in slots and "answering" in slots and "idle" in slots
            for button in screen.query(Button):
                assert button.region.right <= size[0]

            screen.query_one("#monitor-stop", Button).press()
            await _wait_until(pilot, lambda: advisor.stopped
                              and not isinstance(app.screen, ServerMonitorScreen))

    asyncio.run(scenario())


def test_the_monitor_says_when_no_server_runs() -> None:
    class Idle(MonitorAdvisor):
        def server_monitor(self) -> None:  # type: ignore[override]
            return None

    async def scenario() -> None:
        app = JaullApp(advisor=Idle())  # type: ignore[arg-type]
        async with app.run_test(size=(100, 30)) as pilot:
            screen = ServerMonitorScreen()
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: "No Jaull server" in _text(screen, "#monitor-header"))
            assert screen.query_one("#monitor-stop", Button).disabled

    asyncio.run(scenario())
