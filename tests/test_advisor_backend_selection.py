from __future__ import annotations

from pathlib import Path
from typing import Any

from jaull.advisor.service import AdvisorService
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution import ExecutionObservation, ExecutionRequest, ExecutionResult
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.model import ModelAnalysis, SafetensorsSummary
from jaull.domain.runtime import ExecutionReadinessStatus


class _FakeHfClient:
    def model_info(self, repo_id: str) -> object:
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


class _FakeExecutionBackend:
    def __init__(self, outputs: dict[tuple[str, ...], str]) -> None:
        self.outputs = outputs

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            stdout=self.outputs.get(request.command, "llama-cli test build"),
            stderr="",
            observation=ExecutionObservation(
                success=True,
                duration_seconds=0.01,
                exit_code=0,
            ),
        )


def test_advisor_select_runtime_backend_delegates_to_selector() -> None:
    hardware = HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=16, available_bytes=8),
        accelerators=[
            AcceleratorProfile(
                name="AMD Radeon(TM) Graphics",
                vendor=AcceleratorVendor.AMD,
                type=AcceleratorType.INTEGRATED,
                shared_memory=True,
                backends=[
                    ComputeBackendInfo(
                        backend=ComputeBackend.VULKAN,
                        availability=BackendAvailability.AVAILABLE,
                    )
                ],
            )
        ],
    )
    advisor = AdvisorService.build(
        hf_client=_FakeHfClient(),  # type: ignore[arg-type]
        detect_hardware=lambda: hardware,
        inspect_model=_unused_inspect,
        estimate_memory=_unused_estimate,
    )

    selection = advisor.select_runtime_backend()

    assert selection.selected_backend is ComputeBackend.VULKAN
    assert selection.selected_accelerator is not None
    assert selection.selected_accelerator.name == "AMD Radeon(TM) Graphics"


def test_advisor_evaluates_execution_readiness_with_injected_backend(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llama-cli"
    binary.write_text("fake", encoding="utf-8")
    binary.chmod(0o755)
    hardware = HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(),
        memory=MemoryInfo(total_bytes=16, available_bytes=8),
        accelerators=[
            AcceleratorProfile(
                name="AMD Radeon(TM) Graphics",
                vendor=AcceleratorVendor.AMD,
                type=AcceleratorType.INTEGRATED,
                shared_memory=True,
                backends=[
                    ComputeBackendInfo(
                        backend=ComputeBackend.VULKAN,
                        availability=BackendAvailability.AVAILABLE,
                    )
                ],
            )
        ],
    )
    advisor = AdvisorService.build(
        hf_client=_FakeHfClient(),  # type: ignore[arg-type]
        detect_hardware=lambda: hardware,
        inspect_model=_unused_inspect,
        estimate_memory=_unused_estimate,
        llama_cli_path=binary,
    )

    readiness = advisor.evaluate_execution_readiness(
        backend=_FakeExecutionBackend(
            {
                (str(binary), "--list-devices"): "Available devices:\n"
                "Vulkan0: AMD Radeon(TM) Graphics\n",
            }
        )
    )

    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.selection.selected_backend is ComputeBackend.VULKAN


def _unused_inspect(repo_id: str, client: object | None = None) -> ModelAnalysis:
    del repo_id, client
    raise NotImplementedError


def _unused_estimate(**kwargs: Any) -> MemoryEstimate:
    del kwargs
    raise NotImplementedError
