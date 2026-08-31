"""Domain models produced by the memory estimator.

Every numeric estimate carries the reasoning it was derived from — the source of the
underlying number, its confidence, and a short human-readable explanation — so the
CLI can render decisions the user can actually trust.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

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


class TransformerBlockWeightDecompositionMethod(StrEnum):
    """How model-weight bytes were divided into block and non-block weights."""

    CONFIG_PARAMETER_DECOMPOSITION = "config_parameter_decomposition"
    UNIFORM_WEIGHT_FALLBACK = "uniform_weight_fallback"


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
    TRANSFORMER_BLOCKS = "transformer_blocks"
    ESTIMATED_BYTES = "estimated_bytes"
    NONE = "none"

    # Compatibility alias for callers compiled against the first Hardware Fit
    # contract. New code should use TRANSFORMER_BLOCKS because these are model
    # blocks, not runtime-specific offload units.
    LAYERS = "transformer_blocks"

    @classmethod
    def _missing_(cls, value: object) -> HardwareFitPlacementMethod | None:
        if value == "layers":
            return cls.TRANSFORMER_BLOCKS
        return None


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


class TransformerBlockWeightDecomposition(BaseModel):
    """Estimated split of artifact bytes by architectural responsibility.

    The config-based method projects a parameter fraction onto the artifact's
    total weight bytes. It is not a tensor-level measurement: mixed tensor
    quantization, alignment and format metadata can make the real byte split
    differ. The aggregate split conserves bytes exactly. The per-block value is
    rounded up independently, so multiplying it by the block count can exceed
    the aggregate block bytes by at most ``total_transformer_blocks - 1``.

    This describes weight estimation only. It does not say whether non-block
    weights live in GPU or host memory and does not participate in Hardware Fit
    placement yet.
    """

    model_config = ConfigDict(frozen=True)

    total_weight_bytes: int = Field(ge=0)
    estimated_transformer_block_weight_bytes: int = Field(ge=0)
    estimated_non_block_weight_bytes: int = Field(ge=0)
    estimated_bytes_per_transformer_block: int = Field(ge=0)
    total_transformer_blocks: int = Field(gt=0)
    method: TransformerBlockWeightDecompositionMethod

    @model_validator(mode="after")
    def _weights_are_conserved(self) -> TransformerBlockWeightDecomposition:
        if (
            self.estimated_transformer_block_weight_bytes
            + self.estimated_non_block_weight_bytes
            != self.total_weight_bytes
        ):
            raise ValueError(
                "Transformer-block and non-block weight bytes must sum exactly "
                "to total_weight_bytes."
            )
        return self


class WeightEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: MemoryComponent
    num_parameters: int | None = None
    bits_per_parameter: float | None = None
    precision: WeightPrecision | None = None
    gguf_variant: str | None = None
    transformer_block_decomposition: TransformerBlockWeightDecomposition | None = None


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


class HardwareFitOffloadCandidate(BaseModel):
    """One estimated transformer-block placement considered by Hardware Fit.

    These bytes are capacity-planning budget terms. They are not CUDA
    allocations and they are not backend-specific runtime offload units.

    The two pools reconstruct exactly from their own components::

        gpu_required_bytes == gpu_weight_bytes + gpu_kv_cache_bytes
                              + device_reserve_bytes + gpu_overhead_bytes
                              + gpu_safety_margin_bytes

        ram_required_bytes == ram_weight_bytes + ram_kv_cache_bytes
                              + ram_overhead_bytes + ram_safety_margin_bytes

    ``kv_cache_bytes`` is the *total* KV cache being split, kept alongside the
    two shares so ``gpu_kv_cache_bytes + ram_kv_cache_bytes == kv_cache_bytes``
    is visible in the payload rather than having to be trusted. It is context,
    not a GPU term: only ``gpu_kv_cache_bytes`` belongs in the GPU sum.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    gpu_transformer_blocks: int
    gpu_required_bytes: int
    ram_required_bytes: int
    available_vram_bytes: int
    excess_bytes: int = 0
    headroom_bytes: int = 0
    gpu_weight_bytes: int
    ram_weight_bytes: int
    kv_cache_bytes: int
    # Before KV placement followed the transformer-block split, a candidate
    # carried one ``kv_cache_bytes`` charged entirely to VRAM. Reading that as
    # the GPU share is what it meant at the time.
    gpu_kv_cache_bytes: int = Field(
        default=0,
        validation_alias=AliasChoices("gpu_kv_cache_bytes", "kv_cache_bytes"),
    )
    ram_kv_cache_bytes: int = 0
    device_reserve_bytes: int
    gpu_overhead_bytes: int
    gpu_safety_margin_bytes: int
    ram_overhead_bytes: int = 0
    ram_safety_margin_bytes: int = 0


class HardwareFitOffloadDiagnostics(BaseModel):
    """Boundary that explains why the selected offload point was chosen."""

    model_config = ConfigDict(frozen=True)

    search_ceiling_transformer_blocks: int
    selected: HardwareFitOffloadCandidate
    first_rejected_higher: HardwareFitOffloadCandidate | None = None


class HardwareFitResult(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

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
    # How ``kv_cache_bytes`` is split between the two pools. Always sums back to
    # ``kv_cache_bytes``; a placement that uses no GPU leaves the GPU share at 0.
    gpu_kv_cache_bytes: int = 0
    ram_kv_cache_bytes: int = 0
    gpu_transformer_blocks: int | None = Field(
        default=None,
        validation_alias=AliasChoices("gpu_transformer_blocks", "gpu_layers"),
    )
    total_transformer_blocks: int | None = Field(
        default=None,
        validation_alias=AliasChoices("total_transformer_blocks", "total_layers"),
    )
    placement_method: HardwareFitPlacementMethod = HardwareFitPlacementMethod.NONE
    offload_diagnostics: HardwareFitOffloadDiagnostics | None = None
    reason: str
    warnings: list[str] = Field(default_factory=list)

    @property
    def gpu_layers(self) -> int | None:
        """Deprecated compatibility alias for transformer-block placement.

        llama.cpp and other runtimes may expose their own layer/offload units.
        This value is the runtime-agnostic Hardware Fit transformer-block
        estimate and must not be treated as ``--n-gpu-layers``.
        """

        return self.gpu_transformer_blocks

    @property
    def total_layers(self) -> int | None:
        """Deprecated compatibility alias for total transformer blocks."""

        return self.total_transformer_blocks

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
    # was derived from — mode, transformer-block split, and the per-pool byte
    # breakdown.
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
