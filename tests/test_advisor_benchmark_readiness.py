from __future__ import annotations

import pytest

from jaull.advisor.service import AdvisorService
from jaull.benchmarks.errors import BenchmarkUnavailableError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import BenchmarkGpuLayers, BenchmarkRequest
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.hardware import ComputeBackend, CpuInfo, HardwareProfile, MemoryInfo
from jaull.domain.runtime import (
    PyTorchBackendCapability,
    PyTorchBackendCapabilityState,
    PyTorchCapabilityReason,
    PyTorchRuntimeCapability,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.workflow.container import ServiceContainer


class _NeverRunBenchmark:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, request: BenchmarkRequest) -> object:
        self.calls += 1
        raise AssertionError("benchmark worker must not run when preflight fails")


def test_quantized_transformers_benchmark_checks_bitsandbytes_before_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _NeverRunBenchmark()
    advisor = AdvisorService(
        services=_services(),
        transformers_benchmark_runner=worker,  # type: ignore[arg-type]
    )
    capability = PyTorchRuntimeCapability(
        runtime_status=PyTorchRuntimeStatus.AVAILABLE,
        bitsandbytes_available=False,
        backend_capabilities=[
            PyTorchBackendCapability(
                backend=ComputeBackend.CUDA,
                state=PyTorchBackendCapabilityState.CONFIRMED,
                reason=PyTorchCapabilityReason.BACKEND_EXPOSED,
            )
        ],
    )
    monkeypatch.setattr(
        AdvisorService,
        "inspect_pytorch_runtime",
        lambda self: capability,
    )
    monkeypatch.setattr(
        AdvisorService,
        "select_runtime_backend",
        lambda self, hardware=None: _cuda_selection(),
    )

    with pytest.raises(BenchmarkUnavailableError, match=r"bitsandbytes.*not installed"):
        advisor.run_benchmark(
            _request(),
            hardware=_hardware(),
            persist=False,
        )

    assert worker.calls == 0


def _request() -> BenchmarkRequest:
    return BenchmarkRequest(
        artifact=ModelArtifact(
            repo_id="org/model",
            revision="main",
            filename="org/model",
            format="safetensors",
        ),
        runtime=RuntimeRecommendation(
            runtime=RuntimeName.TRANSFORMERS,
            flags=[
                RuntimeFlag(
                    name="quantization",
                    value="4bit",
                    source=RuntimeFlagSource.ESTIMATE,
                    explanation="test plan",
                )
            ],
            confidence=EstimationConfidence.HIGH,
        ),
        backend=ComputeBackend.CUDA,
        gpu_layers=BenchmarkGpuLayers.count_layers(0),
    )


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(physical_cores=4),
        memory=MemoryInfo(total_bytes=8 * 1024**3, available_bytes=4 * 1024**3),
    )


def _cuda_selection() -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=ComputeBackend.CUDA,
        reason=RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
    )


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: _hardware(),
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
