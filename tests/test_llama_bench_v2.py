"""llama_bench_v2: explicit settings, recorded rule, and a check against what ran.

The JSONL row shape is the one a real probe on the RTX 2060 returned with the v2
flags (build 689e227db, 2026-10-10): n_batch 512, n_ubatch 512, flash_attn 0,
f16 caches, load_mode mmap, 6 threads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jaull.benchmarks.errors import BenchmarkConfigurationError
from jaull.domain.benchmarks import (
    LLAMA_BENCH_V2,
    BenchmarkGpuLayers,
    BenchmarkRequest,
    LlamaBenchProtocol,
    llama_bench_v2_mismatches,
)
from jaull.domain.hardware import ComputeBackend
from jaull.runtime import llama_bench_runner
from jaull.runtime.llama_bench_runner import LlamaBenchRunner
from tests.test_benchmarks import _artifact, _executable, _FakeExecutionBackend, _runtime
from tests.test_llama_bench_protocol import _ROW, BANNER, REAL_MD

V2 = LlamaBenchProtocol(
    build_commit="689e227db", build_number=10357, n_threads=6, n_batch=512, n_ubatch=512,
    cache_type_k="f16", cache_type_v="f16", flash_attn="off", load_mode="mmap",
    warmup="ran", thread_policy="physical_cores",
)


def _v2_output(*, n_batch: int = 512, flash: int = 0) -> str:
    rows = [
        _ROW.format(flash=flash, pp=32, tg=0, ts=2468.89),
        _ROW.format(flash=flash, pp=0, tg=16, ts=271.89),
    ]
    rows = [row.replace('"n_batch": 2048', f'"n_batch": {n_batch}') for row in rows]
    return "\n".join((REAL_MD, BANNER, *rows))


def test_a_run_that_followed_v2_has_no_mismatch() -> None:
    assert llama_bench_v2_mismatches(V2, physical_cores=6) == []


@pytest.mark.parametrize("change,cores,expected", [
    ({"flash_attn": "on"}, 6, "flash_attn='on'"),
    ({"n_batch": 2048}, 6, "n_batch=2048"),
    ({"thread_policy": None}, 6, "thread_policy=None"),
    ({}, 12, "n_threads=6 (expected 12)"),  # Judged by the recording CPU's own cores.
])
def test_each_departure_from_v2_is_named(change, cores, expected) -> None:
    mismatches = llama_bench_v2_mismatches(V2.model_copy(update=change), physical_cores=cores)
    assert any(expected in item for item in mismatches)


def test_no_reported_settings_is_a_mismatch_not_a_pass() -> None:
    assert llama_bench_v2_mismatches(None, physical_cores=6) == [
        "llama-bench reported no settings",
    ]


@pytest.mark.parametrize("cores", [None, 0, -1])
@pytest.mark.parametrize("threads", [None, 6])
def test_unknown_or_invalid_core_count_never_confirms_v2(cores, threads) -> None:
    protocol = V2.model_copy(update={"n_threads": threads})
    assert "Physical core count is unknown or invalid." in llama_bench_v2_mismatches(
        protocol, physical_cores=cores,
    )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str) -> tuple:
    monkeypatch.setattr(llama_bench_runner.psutil, "cpu_count", lambda logical=True: 6)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    backend = _FakeExecutionBackend(stdout=output)
    runner = LlamaBenchRunner(backend=backend, llama_bench_path=_executable(tmp_path))
    observation = runner.run(BenchmarkRequest(
        artifact=_artifact(model), runtime=_runtime(ngl=-1), backend=ComputeBackend.CUDA,
        device="CUDA0", gpu_layers=BenchmarkGpuLayers.full(),
        # The captured table's own workload; Fastest's pp512/tg128 is checked later.
        prefill_sizes=(32,), generation_sizes=(16,), repetitions=2,
    ))
    return observation, backend.requests[0].command


def test_the_runner_issues_v2_and_records_the_thread_rule(tmp_path, monkeypatch) -> None:
    observation, command = _run(tmp_path, monkeypatch, _v2_output())
    assert observation.success and observation.methodology == LLAMA_BENCH_V2
    assert command[command.index("-t") + 1] == "6"
    assert command[command.index("-fa") + 1] == "off"
    assert observation.protocol is not None
    assert observation.protocol.thread_policy == "physical_cores"
    assert llama_bench_v2_mismatches(observation.protocol, physical_cores=6) == []


def test_a_completed_run_that_departed_from_v2_keeps_its_measurements(
    tmp_path, monkeypatch,
) -> None:
    # The measurement is kept; applicability then holds it as a reference only.
    observation, _ = _run(tmp_path, monkeypatch, _v2_output(n_batch=2048))
    assert observation.success and observation.measurements
    assert llama_bench_v2_mismatches(observation.protocol, physical_cores=6) == [
        "n_batch=2048 (expected 512)",
    ]


def test_an_unknown_physical_core_count_refuses_to_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llama_bench_runner.psutil, "cpu_count", lambda logical=True: None)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    runner = LlamaBenchRunner(
        backend=_FakeExecutionBackend(stdout=""), llama_bench_path=_executable(tmp_path),
    )
    with pytest.raises(BenchmarkConfigurationError, match="Physical core count"):
        runner.run(BenchmarkRequest(
            artifact=_artifact(model), runtime=_runtime(ngl=-1), backend=ComputeBackend.CUDA,
            device="CUDA0", gpu_layers=BenchmarkGpuLayers.full(),
        ))
