from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.application.execution import ExecutionPlanningError, plan_launch
from jaull.domain.execution import ExecutionRequest
from jaull.domain.inference import InferenceConfiguration, WeightPrecision
from jaull.domain.runtime import RuntimeName
from jaull.estimator.hardware_fit import _calculate_offload_placement, analyze_components
from jaull.estimator.kv_cache import estimate_kv_cache
from jaull.execution.cancellation import ExecutionCancelledError, cancellation_scope
from jaull.execution.host import HostExecutionBackend
from jaull.runtime.llama_cpp import build
from tests._gguf_fixtures import build_header
from tests.test_estimate_with_enrichment import (
    _gguf_analysis,
    _hardware_with_gpu,
    _StubHfClient,
    _StubRangeClient,
)
from tests.test_kv_cache_hybrid import _qwen36_35b_a3b
from tests.test_memory_estimator import _coder_like_gguf


@pytest.mark.parametrize("precision", [WeightPrecision.INT4, WeightPrecision.INT8])
def test_unmodelled_kv_is_unknown_not_zero(precision: WeightPrecision) -> None:
    result = estimate_kv_cache(_qwen36_35b_a3b(), 131072, 1, precision)
    assert result.component.bytes is None
    assert result.component.confidence.value == "unknown"


def test_non_fp16_gguf_does_not_silently_launch_with_default_cache() -> None:
    estimate = _coder_like_gguf()
    config = estimate.inference_configuration.model_copy(update={
        "kv_cache_dtype": WeightPrecision.FLOAT32,
    })
    estimate = estimate.model_copy(update={"inference_configuration": config})
    runtime = build(estimate, _hardware_with_gpu(24, 32))
    assert runtime.runtime is RuntimeName.UNKNOWN
    assert runtime.command_preview is None
    with pytest.raises(ExecutionPlanningError, match="FP16"):
        plan_launch(runtime=RuntimeName.LLAMA_CPP, estimate=estimate,
                    hardware=_hardware_with_gpu(24, 32))


def test_explicit_ten_percent_is_not_the_default_policy() -> None:
    config = InferenceConfiguration(context_length=4096)
    restored = InferenceConfiguration.model_validate_json(config.model_dump_json())
    assert restored.safety_margin_percent is None
    default = _coder_like_gguf()
    explicit = _coder_like_gguf(safety_margin_percent=10.0)
    assert default.safety_margin is None
    assert explicit.safety_margin is not None
    assert explicit.total_bytes > default.total_bytes


def test_hybrid_suffix_charges_full_attention_layer() -> None:
    cache = estimate_kv_cache(_qwen36_35b_a3b(), 262144, 1, WeightPrecision.FLOAT16)
    assert cache.layer_bytes is not None
    assert sum(cache.layer_bytes) == cache.component.bytes
    result = _calculate_offload_placement(
        weights_bytes=40 * 1024**3, kv_cache_bytes=cache.component.bytes,
        overhead_bytes=0, device_reserve_bytes=0, safety_margin_bytes=0,
        total_transformer_blocks=40, gpu_transformer_blocks=1,
        gpu_weight_bytes=1024**3, kv_layer_bytes=cache.layer_bytes,
    )
    assert result is not None
    assert result.gpu_kv_cache_bytes == 512 * 1024**2
    assert result.gpu_kv_cache_bytes + result.ram_kv_cache_bytes == cache.component.bytes
    fit = analyze_components(
        weights_bytes=30 * 1024**3, kv_cache_bytes=cache.component.bytes,
        overhead_bytes=0, total_transformer_blocks=40, kv_layer_bytes=cache.layer_bytes,
        hardware=_hardware_with_gpu(1, 64),
    )
    # 768 MiB weights + 512 MiB KV cannot occupy a 1 GiB GPU.
    assert fit.mode.value == "cpu_ram"


def test_sweep_reads_header_once_and_caps_at_its_limit(monkeypatch: Any) -> None:
    header = _StubRangeClient(body=build_header({
        "general.architecture": "llama", "llama.context_length": 8192,
        "llama.block_count": 2, "llama.embedding_length": 512,
        "llama.attention.head_count": 8, "llama.attention.head_count_kv": 2,
    }))
    calls = []
    original = header.fetch_range

    def fetch(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(header, "fetch_range", fetch)
    from jaull.estimator.service import estimate_memory

    advisor = AdvisorService.build(
        hf_client=_StubHfClient(), detect_hardware=lambda: _hardware_with_gpu(24, 32),
        inspect_model=lambda *args, **kwargs: _gguf_analysis(), estimate_memory=estimate_memory,
    )
    monkeypatch.setattr(AdvisorService, "_make_range_client", lambda self: header)
    sweep = advisor.sweep_context_for_model(
        _gguf_analysis(), _hardware_with_gpu(24, 32), InferenceConfiguration(context_length=4096),
    )
    assert len(calls) == 1
    assert sweep.native_limit == 8192
    assert max(point.context_length for point in sweep.points) == 8192
    assert sweep.provenance["artifact"]["quantization"] == "Q4_K_M"


def test_cancelling_execution_terminates_owned_process(monkeypatch: Any) -> None:
    processes = []
    original = subprocess.Popen

    def spawn(*args: Any, **kwargs: Any) -> Any:
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with cancellation_scope(lambda: bool(processes)), pytest.raises(ExecutionCancelledError):
        HostExecutionBackend().execute(ExecutionRequest(
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
        ))
    assert len(processes) == 1
    assert processes[0].poll() is not None
