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

    @property
    def places_weights_on_gpu(self) -> bool:
        """Whether this placement actually puts model weights in VRAM.

        ``CPU_RAM`` and ``TOO_LARGE`` still report ``gpu_required_bytes`` when a
        GPU exists, but that number is the *hypothetical* resident requirement
        used to explain why the GPU was rejected — not a prediction of what any
        process will allocate. Consumers comparing against a measurement must
        check this first.
        """

        return self.mode in {HardwareFitMode.GPU_RESIDENT, HardwareFitMode.GPU_OFFLOAD}

    @property
    def gpu_physical_bytes(self) -> int | None:
        """VRAM this placement expects a process to actually allocate.

        ``gpu_required_bytes`` is a *capacity budget*: it also carries the
        device reserve (memory deliberately left free for the display server and
        other processes) and the safety margin (policy padding). Neither is ever
        allocated by the inference process, so neither belongs in a comparison
        against a process-attributed measurement.

        Subtracting exactly those two leaves weights, KV cache and runtime
        overhead — the components that model real allocations. This reads the
        placement the analyzer already decided; it does not decide anything.

        ``None`` when the placement puts nothing on the GPU.
        """

        if not self.places_weights_on_gpu or self.gpu_required_bytes is None:
            return None
        return (
            self.gpu_required_bytes
            - self.device_reserve_bytes
            - self.gpu_safety_margin_bytes
        )

    @property
    def ram_physical_bytes(self) -> int | None:
        """System memory this placement expects a process to actually allocate.

        The RAM counterpart of :attr:`gpu_physical_bytes`, and the same
        subtraction: the safety margin is policy, never an allocation. The
        device reserve is only charged to RAM on unified-memory hardware, where
        the accelerator draws from the same pool, so it is removed only there.
        """

        if self.ram_required_bytes is None:
            return None
        physical = self.ram_required_bytes - self.ram_safety_margin_bytes
        if self.memory_topology is HardwareMemoryTopology.UNIFIED_MEMORY:
            physical -= self.device_reserve_bytes
        return physical


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
    # ``assessment`` is the compatibility *summary*: one status, one ratio, one
    # effective device. ``hardware_fit`` is the structured placement the summary
    # was derived from — mode, layer split, and the per-pool byte breakdown.
    # Present only when the analyzer ran, which today means an AUTO target with
    # all three memory components known; ``None`` otherwise, so consumers
    # written before this field keep working unchanged.
    hardware_fit: HardwareFitResult | None = None
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
