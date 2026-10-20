"""Daemon topology and in-container reachability, without Docker or a server."""

import json
import sys
from types import SimpleNamespace

import pytest
from pilot.quality_eval import evaluate, setup


@pytest.mark.parametrize("desktop", [False, True])
def test_network_selection_depends_on_daemon_not_the_wsl_kernel(monkeypatch, desktop):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)

    def output(command, **kwargs):
        if command[1] == "info":
            return json.dumps({"OSType": "linux", "OperatingSystem": (
                "Docker Desktop" if desktop else "Ubuntu 24.04"
            ), "SecurityOptions": []})
        assert command[1:3] == ["context", "inspect"]
        return json.dumps("unix:///var/run/docker.sock")

    monkeypatch.setattr(setup.subprocess, "check_output", output)
    assert setup.docker_host_network() is not desktop


@pytest.mark.parametrize("damage", ["remote", "unknown", "rootless", "windows", "docker_down"])
def test_unverifiable_daemon_topology_fails_before_model_execution(monkeypatch, damage):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    if damage == "remote":
        monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")

    def output(command, **kwargs):
        if damage == "docker_down":
            raise OSError("Docker is not running")
        info = {"OSType": "linux", "OperatingSystem": "Ubuntu", "SecurityOptions": []}
        if damage == "unknown":
            info.pop("OperatingSystem")
        if damage == "windows":
            info["OSType"] = "windows"
        if damage == "rootless":
            info["SecurityOptions"] = ["name=rootless"]
        return json.dumps(info if command[1] == "info" else "unix:///run/docker.sock")

    monkeypatch.setattr(setup.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="Docker"):
        setup.docker_host_network()


@pytest.mark.parametrize("failure", [None, "timeout", "unhealthy", "malformed"])
def test_evaluator_verifies_http_from_inside_its_own_container(monkeypatch, failure):
    def get(url, **kwargs):
        assert url == "http://127.0.0.1:18083/health" and kwargs["timeout"] == 10
        if failure == "timeout":
            raise OSError("unreachable")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: (
            [] if failure == "malformed" else {"status": "loading" if failure else "ok"}
        ))

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get, RequestException=OSError))
    if failure:
        with pytest.raises(ValueError, match="no quality inference was started"):
            evaluate.verify_server_connection("http://127.0.0.1:18083")
    else:
        evaluate.verify_server_connection("http://127.0.0.1:18083")


def test_failed_container_connectivity_never_starts_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--base-url", "http://127.0.0.1:18083",
                                     "--output", str(tmp_path)])
    monkeypatch.setattr(evaluate, "verify_server_connection", lambda _: (
        _ for _ in ()
    ).throw(ValueError("Container cannot reach host")))
    monkeypatch.setattr(evaluate, "run", lambda *args: pytest.fail("Must not start inference"))
    with pytest.raises(ValueError, match="Container cannot reach host"):
        evaluate.main()
    assert "Container cannot reach host" in (tmp_path / "evaluator-error.json").read_text()
