"""Dispatch estimate + hardware into a per-runtime recommendation."""

from __future__ import annotations

from local_ai_check.domain.enums import RepositoryType
from local_ai_check.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.domain.runtime import (
    RuntimeName,
    RuntimeRecommendation,
)
from local_ai_check.runtime import llama_cpp, transformers, vllm


def recommend(
    estimate: MemoryEstimate,
    hardware: HardwareProfile,
    repository_type: RepositoryType,
) -> RuntimeRecommendation:
    if estimate.assessment.status is CompatibilityStatus.INSUFFICIENT:
        return _no_runtime_available()

    if estimate.assessment.status is CompatibilityStatus.UNKNOWN:
        return _unknown_estimate()

    if repository_type is RepositoryType.GGUF:
        return llama_cpp.build(estimate, hardware)

    if repository_type is RepositoryType.TRANSFORMERS:
        primary = transformers.build(estimate, hardware)
        if vllm.is_viable(estimate, hardware):
            primary = primary.model_copy(update={"alternatives": [RuntimeName.VLLM]})
        return primary

    return _unsupported_type(repository_type)


def _no_runtime_available() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.UNKNOWN,
        command_preview=None,
        python_snippet=None,
        confidence=EstimationConfidence.HIGH,
        warnings=[
            "No runtime can run this model on the current hardware: memory is insufficient."
        ],
    )


def _unknown_estimate() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.UNKNOWN,
        command_preview=None,
        python_snippet=None,
        confidence=EstimationConfidence.UNKNOWN,
        warnings=[
            "Not enough information to pick a runtime "
            "(memory estimate is incomplete)."
        ],
    )


def _unsupported_type(repository_type: RepositoryType) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.UNKNOWN,
        command_preview=None,
        python_snippet=None,
        confidence=EstimationConfidence.UNKNOWN,
        warnings=[
            f"Runtime recommendations for repository type {repository_type.value!r} "
            "are out of scope in the current iteration."
        ],
    )


__all__ = ["recommend"]
