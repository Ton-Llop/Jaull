"""Domain objects produced by the RuntimeSelector."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.estimation import EstimationConfidence


class RuntimeName(StrEnum):
    LLAMA_CPP = "llama.cpp"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    UNKNOWN = "unknown"


class RuntimeExecutability(StrEnum):
    """Can the recommended runtime actually load this artifact here?"""

    # Everything the runtime needs is present locally and confirmed.
    EXECUTABLE = "executable"
    # Likely to work but at least one gap: no local binary check, unusual dtype,
    # etc. Safe to suggest, unsafe to promote without a caveat.
    POTENTIALLY_EXECUTABLE = "potentially_executable"
    # Hardware or artifact makes this runtime unusable — e.g. AWQ needing CUDA
    # on a CPU-only box.
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class RuntimeAssessment(BaseModel):
    """The result of validating a runtime against detected hardware."""

    model_config = ConfigDict(frozen=True)

    runtime: RuntimeName
    executability: RuntimeExecutability
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RuntimeFlagSource(StrEnum):
    HARDWARE = "hardware"
    ESTIMATE = "estimate"
    POLICY = "policy"
    USER_INPUT = "user_input"


class RuntimeFlag(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: str
    source: RuntimeFlagSource
    explanation: str


class RuntimeRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: RuntimeName
    command_preview: str | None = None
    python_snippet: str | None = None
    flags: list[RuntimeFlag] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: EstimationConfidence
    alternatives: list[RuntimeName] = Field(default_factory=list)


# Resolve the forward reference in MemoryEstimate.runtime_recommendation now that
# RuntimeRecommendation exists. Importing MemoryEstimate here is safe because
# estimation.py does not import runtime.py at module load time.
def _rebuild_memory_estimate() -> None:
    from jaull.domain.estimation import MemoryEstimate

    MemoryEstimate.model_rebuild(
        _types_namespace={"RuntimeRecommendation": RuntimeRecommendation}
    )


_rebuild_memory_estimate()
