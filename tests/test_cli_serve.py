"""``jaull serve``: the CLI composes the facade and reports; no policy here."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
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
from jaull.domain.hardware import AcceleratorType, AcceleratorVendor, ComputeBackend
from jaull.domain.runtime import (
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
    SelectedAcceleratorRef,
)
from jaull.execution.server_certificate import ServerCertificate
from jaull.execution.server_process import ServerProcessError, ServerState, ServerStatus

GiB = 1024**3

CERTIFICATE = ServerCertificate(
    cert_path=Path("/data/serve/tls/server-cert.pem"),
    key_path=Path("/data/serve/tls/server-key.pem"),
    names=("192.168.1.107", "127.0.0.1", "monk-e", "localhost"),
    expires=datetime(2027, 11, 16, 12, 0, tzinfo=UTC),
    created=True,
    missing=(),
)

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
            flags=[RuntimeFlag(name=name, value=value, source=RuntimeFlagSource.POLICY,
                               explanation="test") for name, value in (
                ("--device", "Vulkan0"), ("--n-gpu-layers", "-1"),
            )],
        ),
        server_binary=Path("/opt/llama/llama-server"),
        server_sha256=sha256(b"fake server").hexdigest(),
        backend_selection=RuntimeBackendSelection(
            selected_backend=ComputeBackend.VULKAN,
            selected_accelerator=SelectedAcceleratorRef(
                name="Intel Arc Pro B60", vendor=AcceleratorVendor.INTEL,
                type=AcceleratorType.DEDICATED,
            ),
            reason=RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE,
        ),
    )


class FakeAdvisor:
    def __init__(self, *, sizing_error: Exception | None = None) -> None:
        self.sizing_error = sizing_error
        self.plan_calls: list[dict[str, Any]] = []
        self.start_calls: list[dict[str, Any]] = []
        self.certificate_calls: list[bool] = []
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

    def create_server_certificate(self, *, renew: bool = False) -> ServerCertificate:
        self.certificate_calls.append(renew)
        return CERTIFICATE

    def detected_server_certificate(self) -> ServerCertificate | None:
        return None

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


def test_plan_server_refuses_a_found_build_that_does_not_run_the_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Audit: without a Jaull build, the build found on the PC was used for any backend.
    from jaull.advisor.service import AdvisorService
    from jaull.application.execution import ExecutionPlanningError
    from jaull.application.serving import ContextLimit
    from jaull.domain.inference import InferenceConfiguration
    from jaull.domain.runtime import LlamaCppInstallation, RuntimeSource

    advisor = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]
    selection = SimpleNamespace(selected_backend=ComputeBackend.CUDA)
    inputs = SimpleNamespace(
        hardware=None, selection=selection, native=None,
        base=InferenceConfiguration(context_length=1),
        estimate=lambda config: None, quick_estimate=lambda config: None,
    )
    monkeypatch.setattr(AdvisorService, "_server_sizing_inputs", lambda self, *args: inputs)
    monkeypatch.setattr("jaull.application.serving.context_limit", lambda **kwargs: ContextLimit(
        users=1, largest_per_user=8192, native=None))
    monkeypatch.setattr("jaull.application.serving.size_server",
                        lambda **kwargs: _plan().sizing)
    monkeypatch.setattr(AdvisorService, "plan_launch", lambda self, **kwargs: _plan().launch)
    monkeypatch.setattr("jaull.runtime.llama_cpp_install.managed_root", lambda: tmp_path)
    found = LlamaCppInstallation(root="/opt", llama_cli="/opt/llama-cli",
                                 source=RuntimeSource.DISCOVERED)
    monkeypatch.setattr(AdvisorService, "_found_llama_cpp_for", lambda self, sel: (found, False))

    with pytest.raises(ExecutionPlanningError, match="does not run cuda"):
        advisor.plan_server(repo_id="a/b", artifact=ARTIFACT, users=1, context_per_user=8192)


def test_discovery_skips_cli_only_builds_and_incompatible_servers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Audit: selection by llama-cli picked a build without llama-server, then found no server.
    from jaull.advisor.service import AdvisorService
    from jaull.domain.runtime import (
        ExecutionReadinessStatus,
        LlamaCppInstallation,
        RuntimeSource,
    )

    def build(name: str, *, server: bool) -> LlamaCppInstallation:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "llama-cli").write_text("")
        if server:
            (folder / "llama-server").write_text("")
        return LlamaCppInstallation(root=str(folder), llama_cli=str(folder / "llama-cli"),
                                    source=RuntimeSource.DISCOVERED)

    builds = [build("cli-only", server=False), build("incompatible", server=True),
              build("complete", server=True)]
    monkeypatch.setattr(AdvisorService, "_runtime_locator",
                        lambda self: SimpleNamespace(discover_llama_cpp=lambda: builds))
    probed: list[str] = []

    def inspect(*, backend: Any, llama_cli_path: str | Path, timeout_seconds: float) -> str:
        probed.append(str(llama_cli_path))
        return str(llama_cli_path)

    def readiness(*, selection: Any, runtime_capability: str) -> object:
        incompatible = runtime_capability == str(tmp_path / "incompatible" / "llama-server")
        return SimpleNamespace(status=ExecutionReadinessStatus.NOT_READY if incompatible
                               else ExecutionReadinessStatus.READY)

    monkeypatch.setattr("jaull.runtime.llama_cpp_capability.inspect_llama_cpp_runtime", inspect)
    monkeypatch.setattr(
        "jaull.runtime.llama_cpp_capability.evaluate_execution_readiness", readiness,
    )
    advisor = AdvisorService(services=SimpleNamespace())  # type: ignore[arg-type]
    selection = SimpleNamespace(selected_backend=ComputeBackend.CPU, alternatives=[])

    found = advisor.found_llama_server(ComputeBackend.CPU, selection)  # type: ignore[arg-type]

    assert found is not None and found.supports_backend
    assert found.path == tmp_path / "complete" / "llama-server"
    assert probed == [str(tmp_path / name / "llama-server")
                      for name in ("incompatible", "complete")]


def test_lan_without_tls_warns_and_slots_are_explained() -> None:
    from dataclasses import replace

    from jaull.presentation.serving import start_lines

    started = ServerStart(
        state=STATE.model_copy(update={"tls": False}), api_key_path=Path("/k"),
        new_api_key=None, start_script=Path("/s"), firewall_rule=None,
        lan_url="http://192.168.1.50:8090",
    )
    text = "\n".join(start_lines(started))
    assert "Warning: Shared without HTTPS" in text and "not separate users" in text

    secure = replace(started, state=STATE.model_copy(update={"tls": True}),
                     lan_url="https://192.168.1.50:8090")
    secure_text = "\n".join(start_lines(secure))
    assert "without HTTPS" not in secure_text and "not separate users" in secure_text
    local = replace(started, lan_url=None)
    assert "not separate users" not in "\n".join(start_lines(local))


def test_no_prompt_cache_flag_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ServeOptions] = []
    monkeypatch.setattr("jaull.cli.app.run_serve_start",
                        lambda reference, options: seen.append(options) or 0)

    result = CliRunner().invoke(app, ["serve", "start", "a/b", "--no-prompt-cache"])

    assert result.exit_code == 0, result.output
    assert seen[0].prompt_cache is False


def test_the_plan_says_how_much_ram_the_prompt_cache_may_take() -> None:
    from dataclasses import replace

    from jaull.presentation.serving import plan_lines

    sized = "\n".join(plan_lines(replace(_plan(), prompt_cache_bytes=2 * GiB,
                                         ram_available_bytes=12 * GiB)))
    assert "Prompt cache in RAM: up to 2.0 GiB (12.0 GiB of RAM free now)." in sized
    off = "\n".join(plan_lines(replace(_plan(), prompt_cache_bytes=0,
                                       ram_available_bytes=3 * GiB)))
    assert "Prompt cache in RAM: off, not enough free RAM" in off


def test_a_shared_pool_warns_that_overflowing_it_stops_every_chat() -> None:
    from dataclasses import replace

    from jaull.application.serving.security import SHARED_POOL_OVERFLOW_WARNING
    from jaull.presentation.serving import plan_lines

    plan = _plan()
    one_pool = replace(plan, sizing=replace(plan.sizing, total_context=131072))
    capped = plan  # 8 people up to 128K each in a 192K pool: it can overflow too.
    roomy = replace(plan, sizing=replace(plan.sizing, total_context=8 * 131072))
    assert SHARED_POOL_OVERFLOW_WARNING in "\n".join(plan_lines(one_pool))
    assert SHARED_POOL_OVERFLOW_WARNING in "\n".join(plan_lines(capped))
    # A pool as large as everyone's full context cannot overflow.
    assert SHARED_POOL_OVERFLOW_WARNING not in "\n".join(plan_lines(roomy))


def test_start_says_where_the_opencode_config_is() -> None:
    from dataclasses import replace

    from jaull.presentation.serving import start_lines

    started = FakeAdvisor().start_server(_plan())
    shared = "\n".join(start_lines(replace(started,
                                            client_config=Path("/data/serve/opencode.json"))))
    assert ("opencode config (it contains the API key): "
            f"{Path('/data/serve/opencode.json')}") in shared
    assert "only works on this PC" not in shared  # the fake start is shared on the LAN

    local = "\n".join(start_lines(replace(started, lan_url=None, firewall_rule=None,
                                           client_config=Path("/data/serve/opencode.json"))))
    assert "only works on this PC" in local


def test_cert_creates_a_certificate_and_says_how_to_use_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jaull.cli.serve import run_serve_cert

    advisor = FakeAdvisor()

    assert run_serve_cert(renew=False, advisor=advisor) == 0  # type: ignore[arg-type]

    out = " ".join(capsys.readouterr().out.split())
    assert advisor.certificate_calls == [False]
    assert "Created a self-signed certificate" in out
    assert str(CERTIFICATE.cert_path) in out and str(CERTIFICATE.key_path) in out
    assert "192.168.1.107, 127.0.0.1, monk-e, localhost" in out
    assert "valid until 2027-11-16" in out
    assert f"--lan --tls-cert {CERTIFICATE.cert_path} --tls-key {CERTIFICATE.key_path}" in out
    # What a self-signed certificate does and does not give.
    assert "browsers and programs do not trust it" in out


def test_cert_that_already_exists_is_kept_and_a_new_address_is_pointed_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dataclasses import replace

    from jaull.cli.serve import run_serve_cert

    class Existing(FakeAdvisor):
        def create_server_certificate(self, *, renew: bool = False) -> ServerCertificate:
            self.certificate_calls.append(renew)
            return replace(CERTIFICATE, created=False, missing=("192.168.1.50",))

    assert run_serve_cert(renew=False, advisor=Existing()) == 0  # type: ignore[arg-type]

    out = " ".join(capsys.readouterr().out.split())
    assert "Already exists" in out and "--renew" in out
    assert "does not cover 192.168.1.50" in out


def test_cert_errors_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    from jaull.cli.serve import run_serve_cert

    class Broken(FakeAdvisor):
        def create_server_certificate(self, *, renew: bool = False) -> ServerCertificate:
            raise ServerProcessError("The certificate cannot be read. Use --renew.")

    assert run_serve_cert(renew=False, advisor=Broken()) == 2  # type: ignore[arg-type]
    assert "--renew" in capsys.readouterr().out


def test_cert_command_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr("jaull.cli.app.run_serve_cert", lambda *, renew: seen.append(renew) or 0)
    runner = CliRunner()

    assert runner.invoke(app, ["serve", "cert"]).exit_code == 0
    assert runner.invoke(app, ["serve", "cert", "--renew"]).exit_code == 0
    assert seen == [False, True]


def test_sharing_on_the_network_uses_https_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    advisor = FakeAdvisor()

    assert run_serve_start("a/b", ServeOptions(quantization=None, lan=True), advisor) == 0  # type: ignore[arg-type]

    start = advisor.start_calls[0]
    assert (start["tls_cert"], start["tls_key"]) == (CERTIFICATE.cert_path, CERTIFICATE.key_path)
    assert advisor.certificate_calls == [False]  # made or kept, never renewed
    out = " ".join(capsys.readouterr().out.split())
    assert "valid until 2027-11-16" in out


def test_https_is_a_choice_it_can_be_asked_for_or_refused() -> None:
    local = FakeAdvisor()
    run_serve_start("a/b", ServeOptions(quantization=None), local)  # type: ignore[arg-type]
    assert local.start_calls[0]["tls_cert"] is None  # only this PC: no certificate by default
    assert local.certificate_calls == []

    asked = FakeAdvisor()
    run_serve_start("a/b", ServeOptions(quantization=None, https=True), asked)  # type: ignore[arg-type]
    assert asked.start_calls[0]["tls_cert"] == CERTIFICATE.cert_path  # even for this PC alone

    refused = FakeAdvisor()
    run_serve_start("a/b", ServeOptions(quantization=None, lan=True, https=False), refused)  # type: ignore[arg-type]
    assert refused.start_calls[0]["tls_cert"] is None  # shared, plain HTTP, with its warning
    assert refused.certificate_calls == []


def test_files_given_by_hand_win_and_cannot_be_refused_at_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    explicit = FakeAdvisor()
    cert, key = Path("/my/cert.pem"), Path("/my/key.pem")
    run_serve_start("a/b", ServeOptions(quantization=None, lan=True, tls_cert=cert,  # type: ignore[arg-type]
                                        tls_key=key), explicit)
    assert (explicit.start_calls[0]["tls_cert"], explicit.start_calls[0]["tls_key"]) == (cert, key)
    assert explicit.certificate_calls == []
    capsys.readouterr()  # what the first start printed is not what is checked below

    contradictory = FakeAdvisor()
    code = run_serve_start("a/b", ServeOptions(quantization=None, https=False, tls_cert=cert,  # type: ignore[arg-type]
                                               tls_key=key), contradictory)
    assert code == 2 and not contradictory.start_calls
    assert "--no-https cannot be combined with --tls-cert" in capsys.readouterr().out


def test_https_flags_are_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ServeOptions] = []
    monkeypatch.setattr("jaull.cli.app.run_serve_start",
                        lambda reference, options: seen.append(options) or 0)

    for flags in ([], ["--https"], ["--no-https"]):
        assert CliRunner().invoke(app, ["serve", "start", "a/b", *flags]).exit_code == 0

    assert [options.https for options in seen] == [None, True, False]


def test_a_shared_start_inside_wsl_warns_instead_of_printing_a_useless_rule() -> None:
    from dataclasses import replace

    from jaull.application.serving.security import WSL_NOT_REACHABLE_WARNING
    from jaull.presentation.serving import start_lines

    started = FakeAdvisor().start_server(_plan())
    out = "\n".join(start_lines(replace(started, wsl=True, firewall_rule=None)))

    assert WSL_NOT_REACHABLE_WARNING in out and "ufw" not in out


def test_sleep_idle_flag_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ServeOptions] = []
    monkeypatch.setattr("jaull.cli.app.run_serve_start",
                        lambda reference, options: seen.append(options) or 0)

    assert CliRunner().invoke(app, ["serve", "start", "a/b", "--sleep-idle", "0"]).exit_code == 0
    assert CliRunner().invoke(app, ["serve", "start", "a/b"]).exit_code == 0

    assert [options.sleep_idle_seconds for options in seen] == [0, None]


def test_the_plan_says_when_the_model_is_released_from_the_gpu() -> None:
    from dataclasses import replace

    from jaull.presentation.serving import plan_lines

    assert "sleep" not in "\n".join(plan_lines(_plan())).lower()
    text = "\n".join(plan_lines(replace(_plan(), sleep_idle_seconds=45)))
    assert "after 45 s without requests" in text


def test_the_plan_warns_when_the_file_is_an_older_published_version() -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    from jaull.domain.artifacts import OutdatedVersion
    from jaull.presentation.serving import plan_lines

    assert "newer version" not in "\n".join(plan_lines(_plan()))
    old = ARTIFACT.model_copy(update={"outdated": OutdatedVersion(
        current_sha256="b" * 64, last_revision="abc",
        still_published_on=datetime(2026, 10, 27, tzinfo=UTC),
    )})
    text = "\n".join(plan_lines(replace(_plan(), artifact=old)))
    assert "Hugging Face has a newer version of this file" in text
    assert "still current on 2026-10-27" in text


def test_start_tells_how_to_make_opencode_trust_the_servers_certificate() -> None:
    from dataclasses import replace

    from jaull.execution.opencode_install import OpencodeInstall
    from jaull.presentation.serving import start_lines

    started = FakeAdvisor().start_server(_plan())
    installed =OpencodeInstall(path=Path("/home/me/.config/opencode/opencode.json"),
                                outcome="added")
    plain = replace(started, opencode=installed)
    assert "NODE_EXTRA_CA_CERTS" not in "\n".join(start_lines(plain))  # no HTTPS, nothing to trust

    argv = ["llama-server", "--ssl-cert-file", "/data/tls/server-cert.pem",
            "--ssl-key-file", "/data/tls/server-key.pem"]
    secure = replace(plain, state=STATE.model_copy(update={"tls": True, "argv": argv}))
    text = "\n".join(start_lines(secure))
    assert "NODE_EXTRA_CA_CERTS=/data/tls/server-cert.pem opencode" in text

    left_alone = replace(secure, opencode=replace(installed, outcome="unreadable"))
    assert "NODE_EXTRA_CA_CERTS" not in "\n".join(start_lines(left_alone))
    assert "NODE_EXTRA_CA_CERTS" not in "\n".join(start_lines(replace(secure, opencode=None)))
    unchanged = replace(secure, opencode=replace(installed, outcome="unchanged"))
    assert "NODE_EXTRA_CA_CERTS=/data/tls/server-cert.pem" in "\n".join(start_lines(unchanged))


def test_load_mode_flag_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ServeOptions] = []
    monkeypatch.setattr("jaull.cli.app.run_serve_start",
                        lambda reference, options: seen.append(options) or 0)

    assert CliRunner().invoke(app, ["serve", "start", "a/b", "--load-mode", "mmap"]).exit_code == 0
    assert CliRunner().invoke(app, ["serve", "start", "a/b"]).exit_code == 0

    assert [options.load_mode for options in seen] == ["mmap", None]


def test_the_plan_says_the_model_is_not_copied_into_ram() -> None:
    from dataclasses import replace

    from jaull.presentation.serving import plan_lines

    assert "-lm" not in "\n".join(plan_lines(_plan()))
    text = "\n".join(plan_lines(replace(_plan(), load_mode="dio")))
    assert "-lm dio" in text and "RAM" in text
    text = "\n".join(plan_lines(replace(_plan(), load_mode="none")))
    assert "-lm none" in text and "without mmap" in text


def test_the_plan_warns_that_mmap_on_the_cpu_can_use_more_ram_than_estimated() -> None:
    # CPU run with -lm mmap used up to 3.2 GB more free RAM than with -lm none for the same
    # buffers (docs/TFG/resultados-cpu-2.md), so the estimate does not cover it.
    from dataclasses import replace

    from jaull.presentation.serving import plan_lines

    def says(plan: object) -> bool:
        return "more RAM than this estimate" in "\n".join(plan_lines(plan))  # type: ignore[arg-type]

    cpu = replace(_plan(), backend=ComputeBackend.CPU)
    assert says(replace(cpu, load_mode="mmap"))
    assert not says(replace(cpu, load_mode="none"))
    assert not says(cpu)
    assert not says(replace(_plan(), load_mode="mmap"))


def test_start_says_where_the_opencode_backup_is() -> None:
    from dataclasses import replace

    from jaull.execution.opencode_install import OpencodeInstall
    from jaull.presentation.serving import start_lines

    started = FakeAdvisor().start_server(_plan())
    installed = OpencodeInstall(path=Path("/c/opencode.json"), outcome="added",
                                backup=Path("/c/opencode.json.jaull-backup"))

    text = "\n".join(start_lines(replace(started, opencode=installed)))

    # A Path prints with the separators of the OS the test runs on (backslashes on Windows).
    assert f"The file as it was before: {installed.backup}" in text


def test_with_the_chat_page_the_address_for_people_comes_first() -> None:
    from dataclasses import replace

    from jaull.presentation.serving import start_lines

    started = FakeAdvisor().start_server(_plan())
    local = f"{started.state.url}"

    with_page = start_lines(replace(started, web_ui=True, lan_url=None))
    assert with_page[0] == f"Ready. Chat page: {local}/  (enter the API key in its settings)"
    assert with_page[1] == f"API for other tools: {local}/v1"
    assert sum("Chat page" in line for line in with_page) == 1  # said once

    shared = start_lines(replace(started, web_ui=True, lan_url="https://192.168.1.5:8090"))
    assert shared[0].startswith("Ready. Chat page: https://192.168.1.5:8090/")
    assert shared[1] == "API for other tools: https://192.168.1.5:8090/v1"
    assert not any(line.startswith("On the local network") for line in shared)

    without_page = start_lines(replace(started, web_ui=False, lan_url=None))
    assert without_page[0] == f"Ready at {local}  (OpenAI API: {local}/v1)"
    assert not any("Chat page" in line for line in without_page)


def test_the_command_that_launched_the_server_is_shown_in_the_details() -> None:
    import re
    from dataclasses import replace

    from jaull.execution.server_process import ServerStatus
    from jaull.presentation.serving import start_lines, status_lines

    argv = ["llama-server", "--model", "/m/a b.gguf", "--port", "8090",
            "--api-key-file", "/data/serve/api-key"]
    started = FakeAdvisor().start_server(_plan())
    started = replace(started, state=STATE.model_copy(update={"argv": argv}))

    text = "\n".join(start_lines(started))
    assert "llama-server --model " in text and "--port 8090" in text
    assert re.search(r"""['"]/m/a b\.gguf['"]""", text)  # an argument with a space is quoted
    assert "--api-key-file /data/serve/api-key" in text  # the path to the key, never the key

    running = "\n".join(status_lines(
        ServerStatus(state=started.state, running=True, healthy=True)))
    assert "--port 8090" in running


def test_every_flag_jaull_gives_llama_server_is_explained_in_plain_words() -> None:
    # A flag added to the server command without a line in the glossary shows up bare in
    # the details people read, so this fails until it is explained.
    from jaull.domain.runtime import RuntimeFlag, RuntimeFlagSource
    from jaull.presentation.serving import FLAG_MEANINGS
    from jaull.runtime.llama_server import ServerLaunch, server_argv

    launch = ServerLaunch(
        binary=Path("/o/llama-server"), model=Path("/m/q.gguf"), parallel=2,
        total_context=65536, context_per_user=32768, host="0.0.0.0", port=8090,
        api_key_file=Path("/d/api-key"), alias="q",
        launch_flags=(
            RuntimeFlag(name="--n-gpu-layers", value="-1", source=RuntimeFlagSource.POLICY,
                        explanation="x"),
            RuntimeFlag(name="--device", value="Vulkan0", source=RuntimeFlagSource.POLICY,
                        explanation="x"),
            RuntimeFlag(name="--flash-attn", value="off", source=RuntimeFlagSource.POLICY,
                        explanation="x"),
        ),
        tls_cert=Path("/t/c.pem"), tls_key=Path("/t/k.pem"), slots=True,
        prompt_cache=False, cache_ram_mib=2048, sleep_idle_seconds=45, load_mode="dio",
    )
    flags = {token for token in server_argv(launch) if token.startswith("-")} - {"-1"}

    assert flags - set(FLAG_MEANINGS) == set()


def test_the_breakdown_shortens_paths_and_pairs_each_flag_with_its_meaning() -> None:
    from jaull.presentation.serving import command_breakdown

    long_model = r"C:\Users\someone\AppData\Local\jaull\models\org\repo\Model-Q4.gguf"
    lines = command_breakdown([
        r"C:\llama-win\llama-server.exe", "--model", long_model, "--n-gpu-layers", "-1",
        "--kv-unified", "--future-flag", "7",
    ])

    text = "\n".join(lines)
    assert "the program that runs the model" in lines[0]
    assert "AppData" not in text and "Model-Q4.gguf" in text  # long paths are shortened
    assert any(line.lstrip().startswith("--n-gpu-layers -1") and "graphics card" in line
               for line in lines)  # "-1" is the flag's value, not another flag
    assert any(line.strip() == "--future-flag 7" for line in lines)  # unknown: shown, no text


def test_a_windows_server_also_gives_opencode_in_wsl_the_path_linux_sees() -> None:
    from jaull.presentation.serving import wsl_run_command

    certificate = r"C:\Users\someone\AppData\Local\jaull\serve\tls\server-cert.pem"
    argv = ["llama-server", "--ssl-cert-file", certificate]
    shared = STATE.model_copy(update={"tls": True, "host": "0.0.0.0", "argv": argv})

    assert wsl_run_command(shared, windows=True) == (
        "NODE_EXTRA_CA_CERTS=/mnt/c/Users/someone/AppData/Local/jaull/serve/tls/server-cert.pem"
        " opencode")
    # Only a Windows server that other systems can reach, and only with HTTPS and a drive path.
    assert wsl_run_command(shared, windows=False) is None
    assert wsl_run_command(shared.model_copy(update={"host": "127.0.0.1"}), windows=True) is None
    assert wsl_run_command(shared.model_copy(update={"tls": False}), windows=True) is None
    assert wsl_run_command(shared.model_copy(update={"argv": [
        "llama-server", "--ssl-cert-file", "/data/tls/server-cert.pem"]}), windows=True) is None
