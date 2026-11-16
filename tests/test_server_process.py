"""Detached llama-server lifecycle, against a fake server that answers /health."""

from __future__ import annotations

import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from jaull.execution.server_process import (
    ServerProcessError,
    ServerProcessManager,
    ServerState,
    port_available,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="fake server is a shebang script")

FAKE_SERVER = """\
import http.server, os, sys, time
argv = sys.argv
port = int(argv[argv.index("--port") + 1])
mode = os.environ.get("FAKE_LLAMA_MODE", "ok")
print("fake llama-server starting", flush=True)
if mode == "exit":
    print("error: model file not found", flush=True)
    sys.exit(3)
if mode == "hang":
    time.sleep(600)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def fake_server(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "llama-server"
    path.parent.mkdir()
    path.write_text(f"#!{sys.executable}\n{FAKE_SERVER}", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def manager(tmp_path: Path) -> ServerProcessManager:
    return ServerProcessManager(tmp_path / "serve")


@posix_only
def test_start_status_stop(manager: ServerProcessManager, fake_server: Path) -> None:
    port = _free_port()
    state = manager.start([str(fake_server), "--port", str(port)], host="127.0.0.1",
                          port=port, alias="qwen", health_timeout_seconds=30, poll_seconds=0.1)
    try:
        assert state.url == f"http://127.0.0.1:{port}"
        assert manager.state_path.is_file()
        status = manager.status()
        assert status.running and status.healthy
        assert status.state == state
        assert "fake llama-server starting" in manager.log_tail()
    finally:
        assert manager.stop() is True
    assert not manager.state_path.exists()
    assert manager.status().running is False
    assert manager.stop() is False


@posix_only
def test_second_start_is_refused_while_running(
    manager: ServerProcessManager, fake_server: Path,
) -> None:
    port = _free_port()
    manager.start([str(fake_server), "--port", str(port)], host="127.0.0.1", port=port,
                  alias="qwen", health_timeout_seconds=30, poll_seconds=0.1)
    try:
        with pytest.raises(ServerProcessError, match="already running"):
            manager.start([str(fake_server), "--port", str(_free_port())], host="127.0.0.1",
                          port=_free_port(), alias="qwen")
    finally:
        manager.stop()


@posix_only
def test_early_exit_reports_the_log_and_leaves_no_state(
    manager: ServerProcessManager, fake_server: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_MODE", "exit")
    port = _free_port()

    with pytest.raises(ServerProcessError, match="exited with code 3") as error:
        manager.start([str(fake_server), "--port", str(port)], host="127.0.0.1", port=port,
                      alias="qwen", health_timeout_seconds=30, poll_seconds=0.1)

    assert "model file not found" in str(error.value)
    assert not manager.state_path.exists()


@posix_only
def test_health_timeout_stops_the_process(
    manager: ServerProcessManager, fake_server: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    port = _free_port()
    started: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)  # type: ignore[call-overload]
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)

    with pytest.raises(ServerProcessError, match="did not become ready"):
        manager.start([str(fake_server), "--port", str(port)], host="127.0.0.1", port=port,
                      alias="qwen", health_timeout_seconds=0.5, poll_seconds=0.1)

    assert started[0].poll() is not None
    assert not manager.state_path.exists()


def test_busy_port_is_refused_with_a_free_suggestion(manager: ServerProcessManager) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = int(busy.getsockname()[1])
        assert port_available("127.0.0.1", port) is False

        expected = rf"Port {port} is already in use\. Port \d+ is free"
        with pytest.raises(ServerProcessError, match=expected):
            manager.start(["llama-server"], host="127.0.0.1", port=port, alias="qwen")


def test_stop_refuses_a_pid_that_is_not_our_server(manager: ServerProcessManager) -> None:
    # This test process is alive, but it is not llama-server: never signal it.
    import os

    import psutil

    manager.directory.mkdir(parents=True)
    me = psutil.Process(os.getpid())
    manager.state_path.write_text(ServerState(
        pid=me.pid, create_time=me.create_time(), host="127.0.0.1", port=1, alias="x",
        argv=["llama-server"], log_path="x", started_at="2026-11-13T00:00:00+00:00",
    ).model_dump_json(), encoding="utf-8")

    assert manager.status().running is False
    assert manager.stop() is False
    assert not manager.state_path.exists()


def test_unreadable_state_is_an_error(manager: ServerProcessManager) -> None:
    manager.directory.mkdir(parents=True)
    manager.state_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ServerProcessError, match="not readable"):
        manager.status()


def test_api_key_is_created_once_and_private(manager: ServerProcessManager) -> None:
    path, key = manager.ensure_api_key()

    assert key is not None and len(key) >= 32
    assert path.read_text(encoding="utf-8").strip() == key
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    again_path, again_key = manager.ensure_api_key()
    assert again_path == path and again_key is None
    assert path.read_text(encoding="utf-8").strip() == key


def test_start_scripts_repeat_the_command(manager: ServerProcessManager) -> None:
    argv = ["/opt/llama b/llama-server", "--model", "/m/q.gguf", "--port", "8090"]

    sh = manager.write_start_script(argv, windows=False)
    assert sh.name == "start-server.sh"
    text = sh.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert "exec '/opt/llama b/llama-server' --model /m/q.gguf --port 8090" in text
    if sys.platform != "win32":
        assert sh.stat().st_mode & stat.S_IXUSR

    bat = manager.write_start_script(argv, windows=True)
    assert bat.name == "start-server.bat"
    assert b'"/opt/llama b/llama-server" --model /m/q.gguf --port 8090\r\n' in bat.read_bytes()


def test_the_client_config_is_readable_only_by_its_owner(
    manager: ServerProcessManager,
) -> None:
    import json
    import stat
    import sys

    path = manager.write_client_config("opencode.json", {"provider": {"jaull": {}}})

    assert json.loads(path.read_text(encoding="utf-8")) == {"provider": {"jaull": {}}}
    assert path.parent == manager.directory
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600



def test_the_stored_key_and_client_files_are_read_back(manager: ServerProcessManager) -> None:
    assert manager.read_api_key() is None and manager.read_client_config("x.json") is None

    _, key = manager.ensure_api_key()
    manager.write_client_config("x.json", {"a": 1})

    assert manager.read_api_key() == key
    assert manager.read_client_config("x.json") is not None
    manager.api_key_path.write_text("  \n", encoding="utf-8")
    assert manager.read_api_key() is None  # an empty file is not a key


def test_running_inside_wsl_is_told_by_its_environment_or_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaull.execution.server_process import running_in_wsl

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert running_in_wsl()
    monkeypatch.delenv("WSL_DISTRO_NAME")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", lambda self, **kw: "Linux 5.15 microsoft-standard-WSL2")
        assert running_in_wsl()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", lambda self, **kw: "Linux 6.8 generic")
        assert not running_in_wsl()

    def missing(self: Path, **kwargs: object) -> str:
        raise OSError("no /proc here")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", missing)
        assert not running_in_wsl()
