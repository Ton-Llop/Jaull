"""What the facade hands over about a server: key, opencode config, certificate, address."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.application.serving.client_config import OPENCODE_CONFIG_FILENAME
from jaull.execution.server_process import ServerProcessManager
from tests.test_cli_serve import STATE, _plan


@pytest.fixture
def facade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[AdvisorService, ServerProcessManager]:
    manager = ServerProcessManager(tmp_path / "serve")
    monkeypatch.setattr(AdvisorService, "_server_manager", lambda self: manager)
    return AdvisorService(services=SimpleNamespace()), manager  # type: ignore[arg-type]


def test_the_key_and_the_opencode_config_are_read_back_for_copying(
    facade: tuple[AdvisorService, ServerProcessManager],
) -> None:
    advisor, manager = facade
    assert advisor.server_api_key() is None and advisor.server_opencode_config() is None

    _, key = manager.ensure_api_key()
    manager.write_client_config(OPENCODE_CONFIG_FILENAME, {"provider": {}})

    assert advisor.server_api_key() == key
    assert json.loads(advisor.server_opencode_config() or "") == {"provider": {}}


def test_the_certificate_is_only_used_once_it_has_been_made(
    facade: tuple[AdvisorService, ServerProcessManager],
) -> None:
    advisor, _ = facade
    assert advisor.detected_server_certificate() is None

    made = advisor.create_server_certificate()
    found = advisor.detected_server_certificate()

    assert made.created and found is not None and not found.created
    assert found.cert_path == made.cert_path


def test_the_share_of_a_server_only_this_pc_reaches(
    facade: tuple[AdvisorService, ServerProcessManager],
) -> None:
    advisor, _ = facade
    local = STATE.model_copy(update={"host": "127.0.0.1"})

    plain = advisor.server_share(local)
    chat = advisor.server_share(local.model_copy(update={"web_ui": True}))

    assert not plain.shared and plain.base_url == local.url and plain.chat_url is None
    assert chat.chat_url == f"{local.url}/" and chat.firewall_rule is None


@pytest.mark.parametrize("wsl", [False, True])
def test_the_share_of_a_server_on_the_network(
    facade: tuple[AdvisorService, ServerProcessManager],
    monkeypatch: pytest.MonkeyPatch, wsl: bool,
) -> None:
    advisor, _ = facade
    monkeypatch.setattr(AdvisorService, "_lan_share",
                        lambda self, port, tls: (f"https://192.168.1.107:{port}", "RULE"))
    monkeypatch.setattr("jaull.execution.server_process.running_in_wsl", lambda: wsl)

    share = advisor.server_share(STATE.model_copy(update={"web_ui": True, "tls": True}))

    assert share.shared and share.base_url == "https://192.168.1.107:8090"
    assert share.chat_url == "https://192.168.1.107:8090/"
    # Inside WSL no rule helps: the network is internal to the PC.
    assert (share.firewall_rule, share.wsl) == (None, True) if wsl else ("RULE", False)


@pytest.mark.parametrize("address", [("192.168.1.107", "255.255.255.0"), None])
def test_the_network_address_and_its_firewall_rule(
    facade: tuple[AdvisorService, ServerProcessManager],
    monkeypatch: pytest.MonkeyPatch, address: tuple[str, str] | None,
) -> None:
    advisor, _ = facade
    monkeypatch.setattr("jaull.execution.server_process.lan_ipv4", lambda: address)

    url, rule = advisor._lan_share(8090, tls=True)

    assert url == ("https://192.168.1.107:8090" if address else None)
    assert rule is not None and "8090" in rule


class _Manager:
    """Stands in for the process manager: records the start instead of launching a server."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.started: dict[str, Any] = {}
        self.argv: list[str] = []
        self.client_config: dict[str, Any] = {}

    def ensure_api_key(self) -> tuple[Path, str | None]:
        return self.directory / "api-key", "NEWKEY"

    def write_start_script(self, argv: list[str], *, windows: bool) -> Path:
        return self.directory / "start-server"

    def start(self, argv: list[str], **kwargs: Any) -> Any:
        self.started = kwargs
        self.argv = argv
        return STATE.model_copy(update={
            "host": kwargs["host"], "port": kwargs["port"], "web_ui": kwargs["web_ui"]})

    def write_client_config(self, name: str, config: dict[str, Any]) -> Path:
        self.client_config = config
        return self.directory / name


@pytest.mark.parametrize(("lan", "wsl"), [(False, False), (True, False), (True, True)])
def test_starting_a_server_reports_how_to_reach_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lan: bool, wsl: bool,
) -> None:
    manager = _Manager(tmp_path)
    binary = tmp_path / "llama-server"
    binary.write_text("fake server", encoding="utf-8")
    binary.chmod(0o755)
    from tests.test_llama_cpp_capability import _FakeExecutionBackend

    host = _FakeExecutionBackend({
        (str(binary), "--list-devices"): "Vulkan0: Intel Arc Pro B60\n",
    })
    monkeypatch.setattr("jaull.execution.host.HostExecutionBackend", lambda: host)
    monkeypatch.setattr(AdvisorService, "_server_manager", lambda self: manager)
    monkeypatch.setattr(AdvisorService, "_lan_share",
                        lambda self, port, tls: ("http://192.168.1.107:8090", "RULE"))
    monkeypatch.setattr("jaull.execution.server_process.running_in_wsl", lambda: wsl)
    advisor = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]
    opencode_home = Path(os.environ["XDG_CONFIG_HOME"]) / "opencode"
    opencode_home.mkdir(parents=True)  # this PC has opencode

    plan = replace(_plan(), server_binary=binary)
    started = advisor.start_server(plan, lan=lan, web_ui=True)

    # opencode on this PC reaches the server on this PC's own address, shared or not.
    assert started.opencode is not None and started.opencode.outcome == "created"
    local = json.loads((opencode_home / "opencode.json").read_text(encoding="utf-8"))
    assert local["provider"]["jaull"]["options"]["baseURL"] == f"{started.state.url}/v1"
    assert manager.started["web_ui"] is True
    assert started.new_api_key == "NEWKEY" and started.web_ui
    assert started.client_config == tmp_path / OPENCODE_CONFIG_FILENAME
    reach = started.lan_url or started.state.url
    provider = manager.client_config["provider"]["jaull"]["options"]
    assert provider == {"baseURL": f"{reach}/v1", "apiKey": "NEWKEY"}
    assert (started.lan_url is not None) is lan
    assert started.wsl is (lan and wsl)
    assert started.firewall_rule == ("RULE" if lan and not wsl else None)
    assert advisor.start_server(plan, opencode=False).opencode is None  # asked not to


def test_the_monitor_knows_whether_the_server_sleeps_when_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from jaull.execution.server_process import ServerStatus

    state = STATE.model_copy(update={"argv": ["llama-server", "--sleep-idle-seconds", "45"]})
    manager = SimpleNamespace(
        status=lambda: ServerStatus(state=state, running=True, healthy=True),
        api_key_path=tmp_path / "api-key",
    )
    monkeypatch.setattr(AdvisorService, "_server_manager", lambda self: manager)
    advisor = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]

    assert advisor.server_monitor()._sleeps is True  # type: ignore[union-attr]

    manager.status = lambda: ServerStatus(  # type: ignore[method-assign]
        state=state.model_copy(update={"argv": ["llama-server"]}), running=True, healthy=True)
    assert advisor.server_monitor()._sleeps is False  # type: ignore[union-attr]

    manager.status = lambda: ServerStatus(state=None, running=False, healthy=False)  # type: ignore[method-assign]
    assert advisor.server_monitor() is None


def test_the_artifact_history_is_the_hugging_face_one() -> None:
    from jaull.advisor.service import _artifact_history
    from jaull.huggingface.artifact_history import HuggingFaceArtifactHistory

    assert isinstance(_artifact_history(), HuggingFaceArtifactHistory)


def _start_with_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[tuple[str, str]],
    *, devices: str = "Vulkan0: Intel Arc Pro B60\n",
) -> Any:
    """Start a Vulkan plan whose launch flags are ``flags``, on a machine that lists ``devices``."""
    from jaull.domain.runtime import RuntimeFlag, RuntimeFlagSource
    from tests.test_llama_cpp_capability import _FakeExecutionBackend

    binary = tmp_path / "llama-server"
    binary.write_text("fake server", encoding="utf-8")
    binary.chmod(0o755)
    host = _FakeExecutionBackend({(str(binary), "--list-devices"): devices})
    monkeypatch.setattr("jaull.execution.host.HostExecutionBackend", lambda: host)
    monkeypatch.setattr(AdvisorService, "_server_manager", lambda self: _Manager(tmp_path))
    advisor = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]
    base = _plan()
    launch = base.launch.model_copy(update={"flags": [
        RuntimeFlag(name=name, value=value, source=RuntimeFlagSource.POLICY, explanation="test")
        for name, value in flags]})
    return advisor.start_server(replace(base, server_binary=binary, launch=launch),
                                opencode=False)


@pytest.mark.parametrize("flags", [
    # What the launch planner writes for a model placed on the CPU while the backend is
    # still Vulkan (an integrated GPU shares the PC's RAM): the confirmed card is pinned,
    # but no layer goes to it.
    [("--device", "Vulkan0"), ("--n-gpu-layers", "0")],
    [("--n-gpu-layers", "0")],
])
def test_a_vulkan_plan_that_runs_on_the_cpu_is_not_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[tuple[str, str]],
) -> None:
    started = _start_with_flags(monkeypatch, tmp_path, flags)

    assert started.state is not None


@pytest.mark.parametrize("flags", [
    [("--device", "Vulkan1"), ("--n-gpu-layers", "0")],  # pinned to a card that is not the one
    [("--n-gpu-layers", "-1")],  # layers on a GPU with no device to say which
    [("--device", "Vulkan1"), ("--n-gpu-layers", "-1")],  # not the device that was confirmed
    [("--device", "Vulkan0")],  # a card with no word on how much goes to it
])
def test_a_plan_that_does_not_match_the_confirmed_device_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[tuple[str, str]],
) -> None:
    from jaull.application.execution.planner import ExecutionPlanningError

    with pytest.raises(ExecutionPlanningError, match="no longer matches the confirmed runtime"):
        _start_with_flags(monkeypatch, tmp_path, flags)
