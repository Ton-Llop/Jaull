"""Scaffolding to walk `uv run jaull` on a simulated Intel Arc machine.

Faked: the Hub (search, model metadata, downloads), the host machine's probes and the
``llama-cli`` / ``llama-bench`` executables (a scripted backend answers their
commands and records every argv). Real: hardware detection, backend selection, the
estimator, Hardware Fit, the launch policy, ranking, readiness, the planner, the
advisor and every TUI screen. Output samples are synthetic, shaped like llama.cpp's
documented logs; none was captured from a B60.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from jaull.advisor.service import AdvisorService
from jaull.benchmarks.storage import BenchmarkStore
from jaull.bootstrap.container import ServiceContainer
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import ExecutionRequest, ExecutionResult
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelConfig
from jaull.estimator.service import estimate_memory
from jaull.runtime.llama_bench_runner import LlamaBenchRunner
from jaull.runtime.llama_cpp_runner import LlamaCppRunner
from jaull.runtime.locator import RuntimeLocator, RuntimeLocatorConfig
from jaull.workflow.progress import HARDWARE_STEPS
from tests._intel_fixtures import B60_NAME, GIB
from tests._workflow_fixtures import FakeHfClient, FakeSearchClient, candidate, gguf_analysis
from tests.test_llama_cpp_capability import _result
from tests.test_tui_recommendation_execution import _pytorch_runtime_capability

REPO = "org/Coder-7B-GGUF"
FILENAME = "coder-7b-q4_k_m.gguf"
LLAMA_7B = ModelConfig(
    model_type="llama",
    max_position_embeddings=32768,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
)

BENCH_TABLE = """
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = {name} (Intel open-source Mesa driver) | uma: 0 | fp16: 1 | warp size: 32
| model | size | params | backend | ngl | dev     | test   | t/s            |
| ----- | ---- | ------ | ------- | --- | ------- | ------ | -------------- |
| llama | 4.0G | 7.0 B  | Vulkan  | -1  | {device} | pp512  | 812.00 ± 10.00 |
| llama | 4.0G | 7.0 B  | Vulkan  | -1  | {device} | tg128  | 41.00 ± 1.00   |
build: abc123
"""


def listing(devices: tuple[tuple[str, str | None], ...]) -> str:
    lines = ["Available devices:"]
    for runtime_id, name in devices:
        suffix = f"{name} (24576 MiB, 23552 MiB free)" if name else "(24576 MiB, 23552 MiB free)"
        lines.append(f"  {runtime_id}: {suffix}")
    return "\n".join(lines) + "\n"


@dataclass
class ScriptedLlama:
    """Stands in for llama-cli and llama-bench: answers by command, records every argv."""

    devices: tuple[tuple[str, str | None], ...] = (("Vulkan0", B60_NAME),)
    commands: list[tuple[str, ...]] = field(default_factory=list)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        command = request.command
        self.commands.append(command)
        executable = Path(command[0]).name
        if "--list-devices" in command:
            return _result(stdout=listing(self.devices))
        if "--version" in command:
            return _result(stdout="version: 1 (abc123)\nbuilt with test")
        if executable.startswith("llama-bench"):
            dev = command[command.index("-dev") + 1] if "-dev" in command else "none"
            return _result(stdout=BENCH_TABLE.format(name=B60_NAME, device=dev))
        device = command[command.index("--device") + 1] if "--device" in command else "Vulkan0"
        return _result(
            stdout="Hello from the Arc.",
            stderr=(
                f"ggml_vulkan: Found 1 Vulkan devices:\n"
                f"ggml_vulkan: 0 = {B60_NAME} (Intel open-source Mesa driver) | uma: 0\n"
                f"load_tensors: offloaded 33/33 layers to GPU\n"
                f"load_tensors:      {device} model buffer size =  4096.00 MiB\n"
                f"llama_context:      {device} KV buffer size =   512.00 MiB\n"
            ),
        )

    def argv_of(self, kind: str) -> list[tuple[str, ...]]:
        """Recorded commands of one program (``llama-cli`` or ``llama-bench``)."""
        return [c for c in self.commands if Path(c[0]).name.startswith(kind)]


@dataclass(frozen=True)
class IntelAdvisor(AdvisorService):
    """The real advisor with its machine boundary replaced: probes, downloads, execution."""

    scripted: ScriptedLlama | None = None
    local_model: Path | None = None

    def _host_execution_backend(self):  # type: ignore[no-untyped-def]
        assert self.scripted is not None
        return self.scripted

    def inspect_llama_cpp_runtime(self, *, backend=None, selection=None, hardware=None):  # type: ignore[no-untyped-def]
        # The public probe builds a HostExecutionBackend when none is given; here the
        # scripted llama.cpp answers instead, so no process is ever spawned.
        return super().inspect_llama_cpp_runtime(
            backend=backend or self.scripted, selection=selection, hardware=hardware
        )

    def inspect_pytorch_runtime(self, *, backend=None, refresh=False):  # type: ignore[no-untyped-def]
        return _pytorch_runtime_capability()

    def resolve_artifact(self, repo_id, quantization=None, revision=None):  # type: ignore[no-untyped-def]
        assert self.local_model is not None
        return ModelArtifact(
            repo_id=repo_id,
            revision="main",
            filename=FILENAME,
            format="gguf",
            quantization=quantization or "Q4_K_M",
            size_bytes=self.local_model.stat().st_size,
            local_path=self.local_model,
            sha256="a" * 64,
            is_downloaded=True,
            is_verified=True,
        )

    def download_artifact(self, artifact, on_progress=None):  # type: ignore[no-untyped-def]
        return artifact

    def verify_artifact(self, artifact, *, full=False):  # type: ignore[no-untyped-def]
        return artifact.model_copy(update={"is_verified": True})


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def intel_advisor(
    tmp_path: Path,
    profile: HardwareProfile,
    *,
    devices: tuple[tuple[str, str | None], ...] = (("Vulkan0", B60_NAME),),
    weights_gib: int = 5,
) -> tuple[IntelAdvisor, ScriptedLlama]:
    suffix = ".exe" if os.name == "nt" else ""
    cli = _executable(tmp_path / "bin" / f"llama-cli{suffix}")
    bench = _executable(tmp_path / "bin" / f"llama-bench{suffix}")
    model = tmp_path / "models" / FILENAME
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF")
    scripted = ScriptedLlama(devices=devices)
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_cli_path=cli, llama_bench_path=bench)
    )

    def detect(**kwargs: object) -> HardwareProfile:
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for key, _ in HARDWARE_STEPS:
                on_step(key)
        return profile

    analysis = gguf_analysis(
        repo_id=REPO, quantizations=("Q4_K_M",), base_bytes=weights_gib * GIB
    ).model_copy(update={"config": LLAMA_7B})
    services = ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=FakeSearchClient(
            default=[candidate(repo_id=REPO, tags=["text-generation", "gguf", "code"])]
        ),
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=lambda repo_id, client=None: analysis,  # type: ignore[arg-type]
        estimate_memory=estimate_memory,  # the real estimator and Hardware Fit
    )
    advisor = IntelAdvisor(
        services=services,
        scripted=scripted,
        local_model=model,
        llama_cli_path=cli,
        llama_bench_path=bench,
        runtime_locator=locator,
        llama_cpp_installation=locator.resolve_llama_cpp(),
        llama_cpp_runner=LlamaCppRunner(backend=scripted, llama_cli_path=cli),
        llama_bench_runner=LlamaBenchRunner(backend=scripted, llama_bench_path=bench),
        benchmark_store=BenchmarkStore(root=tmp_path / "benchmarks"),
    )
    return advisor, scripted
