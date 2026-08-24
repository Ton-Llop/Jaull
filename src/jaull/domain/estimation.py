"""Domain models produced by the memory estimator.

Every numeric estimate carries the reasoning it was derived from — the source of the
underlying number, its confidence, and a short human-readable explanation — so the
CLI can render decisions the user can actually trust.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.enums import RepositoryType
from jaull.domain.inference import InferenceConfiguration, TargetDevice, WeightPrecision
from jaull.domain.model import ModelRepositoryInfo

if TYPE_CHECKING:
    from jaull.domain.runtime import RuntimeRecommendation


class EstimateSource(StrEnum):
    EXACT = "exact"                          # Directly measured (e.g. GGUF file size on the Hub).
    METADATA = "metadata"                    # From reliable metadata (safetensors headers).
    GGUF_HEADER = "gguf_header"              # Read from the GGUF file's own KV metadata table.
    BASE_MODEL_CONFIG = "base_model_config"  # From the resolved base repository's config.json.
    DERIVED = "derived"                      # Computed from configuration or file sizes.
    ASSUMED = "assumed"                      # From a documented policy/constant.
    UNKNOWN = "unknown"                      # Not enough information to estimate.


class EstimationConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CompatibilityStatus(StrEnum):
    COMFORTABLE = "comfortable"
    COMPATIBLE = "compatible"
    TIGHT = "tight"
    OFFLOADING_REQUIRED = "offloading_required"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class HardwareFitMode(StrEnum):
    GPU_RESIDENT = "gpu_resident"
    GPU_OFFLOAD = "gpu_offload"
    CPU_RAM = "cpu_ram"
    TOO_LARGE = "too_large"


class HardwareMemoryTopology(StrEnum):
    DISCRETE_MEMORY = "discrete_memory"
    UNIFIED_MEMORY = "unified_memory"


class HardwareFitPlacementMethod(StrEnum):
    LAYERS = "layers"
    ESTIMATED_BYTES = "estimated_bytes"
    NONE = "none"


class MetadataSource(StrEnum):
    MODEL_CARD_METADATA = "model_card_metadata"
    GGUF_METADATA = "gguf_metadata"
    EXPLICIT_REPOSITORY_REFERENCE = "explicit_repository_reference"
    UNRESOLVED = "unresolved"


class ConfigurationSource(StrEnum):
    EXACT_FILE_METADATA = "exact_file_metadata"
    GGUF_HEADER = "gguf_header"
    BASE_MODEL_CONFIG = "base_model_config"
    MODEL_CARD = "model_card"
    DERIVED = "derived"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"


class BaseModelResolution(BaseModel):
    """Explains how (or whether) we identified the base repository of a GGUF variant."""

    model_config = ConfigDict(frozen=True)

    repo_id: str | None
    source: MetadataSource
    confidence: EstimationConfidence
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)


class MemoryComponent(BaseModel):
    """A single line of the memory breakdown."""

    model_config = ConfigDict(frozen=True)

    name: str
    bytes: int | None
    source: EstimateSource
    confidence: EstimationConfidence
    explanation: str


class WeightEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: MemoryComponent
    num_parameters: int | None = None
    bits_per_parameter: float | None = None
    precision: WeightPrecision | None = None
    gguf_variant: str | None = None


class KvCacheEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: MemoryComponent
    layers: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    context_length: int
    batch_size: int
    dtype_bytes: int
    formula: str
    notes: list[str] = Field(default_factory=list)


class RuntimeOverheadEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: MemoryComponent
    base_bytes: int
    weight_fraction: float
    minimum_bytes: int


class CompatibilityAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: CompatibilityStatus
    confidence: EstimationConfidence
    target_device: TargetDevice
    effective_device: TargetDevice
    available_vram_bytes: int | None = None
    available_ram_bytes: int | None = None
    ratio: float | None = None
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HardwareFitResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: HardwareFitMode
    memory_topology: HardwareMemoryTopology
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    device_reserve_bytes: int = 0
    safety_margin_bytes: int = 0
    available_vram_bytes: int | None = None
    available_ram_bytes: int | None = None
    gpu_required_bytes: int | None = None
    gpu_weight_bytes: int = 0
    gpu_overhead_bytes: int = 0
    gpu_safety_margin_bytes: int = 0
    ram_required_bytes: int | None = None
    ram_weight_bytes: int = 0
    ram_overhead_bytes: int = 0
    ram_safety_margin_bytes: int = 0
    gpu_layers: int | None = None
    total_layers: int | None = None
    placement_method: HardwareFitPlacementMethod = HardwareFitPlacementMethod.NONE
    reason: str
    warnings: list[str] = Field(default_factory=list)


class MemoryEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository: ModelRepositoryInfo
    repository_type: RepositoryType
    inference_configuration: InferenceConfiguration
    weights: WeightEstimate
    kv_cache: KvCacheEstimate
    runtime_overhead: RuntimeOverheadEstimate
    device_reserve: MemoryComponent
    safety_margin: MemoryComponent | None
    total_bytes: int | None
    assessment: CompatibilityAssessment
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Additive fields introduced in Fase 3 (GGUF enrichment). Both default to None/empty
    # so consumers written against schema_version=1 keep working unchanged.
    base_model_resolution: BaseModelResolution | None = None
    configuration_sources: dict[str, ConfigurationSource] = Field(default_factory=dict)
    # Architecture shortname (e.g. "llama", "qwen2") when known. Additive.
    architecture: str | None = None
    # Additive in Fase 4 (RuntimeSelector). Also None-by-default for compatibility.
    runtime_recommendation: RuntimeRecommendation | None = None
