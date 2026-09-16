from __future__ import annotations

from jaull.domain.artifact_profile import ArtifactConfirmation, ArtifactFormat, ArtifactProfile
from jaull.domain.estimation import EstimationConfidence
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
from jaull.domain.runtime import RuntimeExecutability, RuntimeName, RuntimeRecommendation
from jaull.runtime.executability import assess_runtime


def test_llama_cpp_gguf_accepts_a_detected_vulkan_accelerator() -> None:
    hardware = HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test"),
        memory=MemoryInfo(total_bytes=16, available_bytes=8),
        accelerators=[
            AcceleratorProfile(
                name="AMD Radeon",
                vendor=AcceleratorVendor.AMD,
                type=AcceleratorType.DEDICATED,
                backends=[
                    ComputeBackendInfo(
                        backend=ComputeBackend.VULKAN,
                        availability=BackendAvailability.AVAILABLE,
                    )
                ],
            )
        ],
    )

    assessment = assess_runtime(
        RuntimeRecommendation(
            runtime=RuntimeName.LLAMA_CPP,
            confidence=EstimationConfidence.LOW,
        ),
        hardware,
        ArtifactProfile(
            format=ArtifactFormat.GGUF,
            confirmation=ArtifactConfirmation.CONFIRMED,
        ),
        estimate=None,
    )

    assert assessment.executability is RuntimeExecutability.POTENTIALLY_EXECUTABLE
    assert assessment.reasons[0] == "GGUF + GPU: llama.cpp is designed for this pairing."
