"""``jaull serve``: the CLI composes the facade and reports; no policy here."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from jaull.application.serving import ServerSizing, ServerSizingError
from jaull.application.serving.models import ServerPlan, ServerStart
from jaull.cli.app import app
from jaull.cli.serve import ServeOptions, run_serve_start, run_serve_status, run_serve_stop
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import (
    EstimationConfidence,
    HardwareFitMode,
    HardwareFitResult,
    HardwareMemoryTopology,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.execution.server_process import ServerProcessError, ServerState, ServerStatus

GiB = 1024**3

ARTIFACT = ModelArtifact(
    repo_id="unsloth/Qwen3.6-35B-A3B-GGUF", revision="abc", filename="q.gguf", format="gguf",
    quantization="IQ4_XS", local_path=Path("/models/q.gguf"), is_downloaded=True,
)
STATE = ServerState(
    pid=4242, create_time=1.0, host="0.0.0.0", port=8090, alias="Qwen3.6-35B-A3B-GGUF",
    argv=["llama-server"], log_path="/data/serve/server.log",
    started_at="2026-11-13T12:00:00+00:00",
)


def _plan(parallel: int = 8) -> ServerPlan:
    fit = HardwareFitResult(
        mode=HardwareFitMode.GPU_RESIDENT, memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
        weights_bytes=16 * GiB, kv_cache_bytes=4 * GiB, overhead_bytes=0,
        available_vram_bytes=24 * GiB, gpu_required_bytes=20 * GiB, reason="test",
    )
    sizing = ServerSizing(
        parallel=parallel, context_per_user=131072, total_context=196608,
        estimate=SimpleNamespace(hardware_fit=fit),  # type: ignore[arg-type]
        gpu_budget_bytes=21 * GiB,
    )
    return ServerPlan(
        artifact=ARTIFACT, alias="Qwen3.6-35B-A3B-GGUF", backend=ComputeBackend.VULKAN,
        sizing=sizing, launch=RuntimeRecommendation(
            runtime=RuntimeName.LLAMA_CPP, confidence=EstimationConfidence.HIGH,
        ),
        server_binary=Path("/opt/llama/llama-server"),
    )


class FakeAdvisor:
    def __init__(self, *, sizing_error: Exception | None = None) -> None:
        self.sizing_error = sizing_error
        self.plan_calls: list[dict[str, Any]] = []
        self.start_calls: list[dict[str, Any]] = []
        self.stopped = False

    def prepare_serve_artifact(self, repo_id: str, **_: object) -> ModelArtifact:
        return ARTIFACT

    def plan_server(self, **kwargs: Any) -> ServerPlan:
        self.plan_calls.append(kwargs)
        if self.sizing_error is not None:
            raise self.sizing_error
        return _plan()

    def start_server(self, plan: ServerPlan, **kwargs: Any) -> ServerStart:
        self.start_calls.append(kwargs)
        return ServerStart(
            state=STATE, api_key_path=Path("/data/serve/api-key"), new_api_key="KEY123",
            start_script=Path("/data/serve/start-server.sh"),
            firewall_rule="sudo ufw allow from 192.168.1.0/24 to any port 8090 proto tcp",
            lan_url="http://192.168.1.107:8090",
        )

    def llama_cpp_update_notice(self) -> str | None:
        return "llama.cpp b13000 is available"

    def stop_server(self) -> bool:
        self.stopped = True
        return True

    def server_status(self) -> ServerStatus:
        return ServerStatus(state=STATE, running=True, healthy=True)


def test_start_plans_starts_and_reports(capsys: pytest.CaptureFixture[str]) -> None:
    advisor = FakeAdvisor()
    options = ServeOptions(quantization="IQ4_XS", users=8, context_per_user=131072, lan=True)

    assert run_serve_start("unsloth/Qwen3.6-35B-A3B-GGUF", options, advisor) == 0  # type: ignore[arg-type]

    assert advisor.plan_calls[0]["users"] == 8
    assert advisor.plan_calls[0]["context_per_user"] == 131072
    assert advisor.start_calls[0]["lan"] is True
    out = " ".join(capsys.readouterr().out.split())
    assert "8 users at once, up to 131072 tokens each, sharing a pool of 196608 tokens" in out
    assert "KEY123" in out and "shown once" in out
    assert "http://192.168.1.107:8090/v1" in out
    assert "does not change the firewall" in out and "sudo ufw allow" in out
    assert str(Path("/data/serve/start-server.sh")) in out
    assert "b13000" in out


def test_sizing_error_exits_6_without_starting(capsys: pytest.CaptureFixture[str]) -> None:
    advisor = FakeAdvisor(sizing_error=ServerSizingError(
        "The largest that fits is [65536] tokens", largest_context_per_user=65536,
    ))

    assert run_serve_start("a/b", ServeOptions(quantization=None), advisor) == 6  # type: ignore[arg-type]

    assert advisor.start_calls == []
    assert "[65536]" in capsys.readouterr().out


def test_tls_needs_both_files(capsys: pytest.CaptureFixture[str]) -> None:
    options = ServeOptions(quantization=None, tls_cert=Path("cert.pem"))

    assert run_serve_start("a/b", options, FakeAdvisor()) == 2  # type: ignore[arg-type]


def test_server_failure_exits_5(capsys: pytest.CaptureFixture[str]) -> None:
    class Failing(FakeAdvisor):
        def start_server(self, plan: ServerPlan, **kwargs: Any) -> ServerStart:
            raise ServerProcessError("Port 8090 is already in use. Port 8091 is free.")

    assert run_serve_start("a/b", ServeOptions(quantization=None), Failing()) == 5  # type: ignore[arg-type]
    assert "8091" in capsys.readouterr().out


def test_stop_and_status(capsys: pytest.CaptureFixture[str]) -> None:
    advisor = FakeAdvisor()

    assert run_serve_stop(advisor) == 0  # type: ignore[arg-type]
    assert advisor.stopped
    assert run_serve_status(advisor) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "Server stopped." in out
    assert "ready" in out and "http://127.0.0.1:8090" in out and "4242" in out


def test_status_when_nothing_runs(capsys: pytest.CaptureFixture[str]) -> None:
    class Idle(FakeAdvisor):
        def server_status(self) -> ServerStatus:
            return ServerStatus(state=None, running=False, healthy=False)

    assert run_serve_status(Idle()) == 1  # type: ignore[arg-type]
    assert "No Jaull server is running." in capsys.readouterr().out


def test_commands_are_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ServeOptions]] = []

    def fake_start(reference: str, options: ServeOptions) -> int:
        seen.append((reference, options))
        return 0

    monkeypatch.setattr("jaull.cli.app.run_serve_start", fake_start)
    monkeypatch.setattr("jaull.cli.app.run_serve_stop", lambda: 0)
    monkeypatch.setattr("jaull.cli.app.run_serve_status", lambda: 1)
    runner = CliRunner()

    result = runner.invoke(app, [
        "serve", "start", "a/b", "-q", "Q4_K_M", "--users", "4", "--context", "65536",
        "--lan", "--port", "8091", "--backend", "CUDA", "--web-ui", "--shared-context",
    ])
    assert result.exit_code == 0, result.output
    reference, options = seen[0]
    assert reference == "a/b"
    assert options == ServeOptions(
        quantization="Q4_K_M", users=4, context_per_user=65536, lan=True, port=8091,
        backend=ComputeBackend.CUDA, web_ui=True, shared_context=True,
    )
    assert runner.invoke(app, ["serve", "stop"]).exit_code == 0
    assert runner.invoke(app, ["serve", "status"]).exit_code == 1


def test_start_defaults_are_local_one_user() -> None:
    options = ServeOptions(quantization=None)

    assert options.lan is False
    assert options.users == 1
