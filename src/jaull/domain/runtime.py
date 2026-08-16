"""Domain objects produced by the RuntimeSelector."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.estimation import EstimationConfidence
from jaull.domain.hardware import (
    AcceleratorType,
    AcceleratorVendor,
    ComputeBackend,
    ComputeBackendInfo,
)


class RuntimeName(StrEnum):
    LLAMA_CPP = "llama.cpp"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    UNKNOWN = "unknown"


class RuntimeFamily(StrEnum):
    LLAMA_CPP = "llama_cpp"
    PYTORCH_TRANSFORMERS = "pytorch_transformers"


class RuntimeSource(StrEnum):
    EXPLICIT = "explicit"
    REGISTERED = "registered"
    MANAGED = "managed"
    SIBLING = "sibling"
    PATH = "path"
    DISCOVERED = "discovered"
    CURRENT_ENVIRONMENT = "current_environment"


class RuntimeResolutionStatus(StrEnum):
    FOUND = "found"
    RUNTIME_NOT_FOUND = "runtime_not_found"
    CONFIGURED_RUNTIME_MISSING = "configured_runtime_missing"
    RUNTIME_NOT_EXECUTABLE = "runtime_not_executable"
    RUNTIME_PROBE_FAILED = "runtime_probe_failed"
    REQUESTED_BACKEND_NOT_AVAILABLE = "requested_backend_not_available"


class RuntimeDiscoveryMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str | None = None
    searched: list[str] = Field(default_factory=list)
    message: str | None = None


class LlamaCppInstallation(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: str
    llama_cli: str | None = None
    llama_bench: str | None = None
    source: RuntimeSource
    status: RuntimeResolutionStatus = RuntimeResolutionStatus.FOUND
    discovery: RuntimeDiscoveryMetadata = Field(default_factory=RuntimeDiscoveryMetadata)
    version_text: str | None = None


class PyTorchInstallation(BaseModel):
    model_config = ConfigDict(frozen=True)

    python_executable: str | None = None
    source: RuntimeSource
    status: RuntimeResolutionStatus = RuntimeResolutionStatus.FOUND
    discovery: RuntimeDiscoveryMetadata = Field(default_factory=RuntimeDiscoveryMetadata)


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


class RuntimeBackendSelectionReason(StrEnum):
    NATIVE_BACKEND_AVAILABLE = "native_backend_available"
    VULKAN_BACKEND_AVAILABLE = "vulkan_backend_available"
    CPU_FALLBACK = "cpu_fallback"
    NO_USABLE_ACCELERATOR = "no_usable_accelerator"
    SOFTWARE_RENDERER_IGNORED = "software_renderer_ignored"


class SelectedAcceleratorRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    vendor: AcceleratorVendor
    type: AcceleratorType
    vendor_id: str | None = None
    device_id: str | None = None
    pci_bus_id: str | None = None
    uuid: str | None = None
    dedicated_memory_bytes: int | None = None
    shared_memory: bool = False
    detection_sources: list[str] = Field(default_factory=list)


class RuntimeBackendCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    accelerator: SelectedAcceleratorRef | None = None
    backend_info: ComputeBackendInfo | None = None
    reason: RuntimeBackendSelectionReason


class RuntimeBackendSelection(BaseModel):
    """Preferred compute backend according to observed hardware/environment.

    This does not certify that an installed runtime binary supports the backend.
    For example, ``selected_backend=vulkan`` means Vulkan is the preferred
    target from hardware discovery, not that ``llama-cli`` was built with
    ``GGML_VULKAN``.
    """

    model_config = ConfigDict(frozen=True)

    selected_backend: ComputeBackend
    selected_accelerator: SelectedAcceleratorRef | None = None
    reason: RuntimeBackendSelectionReason
    backend_info: ComputeBackendInfo | None = None
    alternatives: list[RuntimeBackendCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LlamaCppBinaryStatus(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    NOT_EXECUTABLE = "not_executable"
    PROBE_FAILED = "probe_failed"
    UNKNOWN = "unknown"


class LlamaCppBackendCapabilityState(StrEnum):
    CONFIRMED = "confirmed"
    NOT_OBSERVED = "not_observed"
    UNKNOWN = "unknown"


class LlamaCppCapabilityReason(StrEnum):
    RUNTIME_AVAILABLE = "runtime_available"
    RUNTIME_MISSING = "runtime_missing"
    RUNTIME_NOT_EXECUTABLE = "runtime_not_executable"
    PROBE_FAILED = "probe_failed"
    BACKEND_EXPOSED = "backend_exposed"
    BACKEND_NOT_OBSERVED = "backend_not_observed"
    CAPABILITY_UNKNOWN = "capability_unknown"


class LlamaCppRuntimeDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    runtime_id: str
    name: str | None = None
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None


class LlamaCppBackendCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    state: LlamaCppBackendCapabilityState
    devices: list[LlamaCppRuntimeDevice] = Field(default_factory=list)
    reason: LlamaCppCapabilityReason
    source: str | None = None
    detail: str | None = None


class LlamaCppRuntimeCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_family: RuntimeFamily = RuntimeFamily.LLAMA_CPP
    binary_path: str | None = None
    binary_status: LlamaCppBinaryStatus
    version_text: str | None = None
    backend_capabilities: list[LlamaCppBackendCapability] = Field(default_factory=list)
    probe_source: str | None = None
    message: str | None = None


class PyTorchRuntimeStatus(StrEnum):
    AVAILABLE = "available"
    PYTHON_MISSING = "python_missing"
    TORCH_MISSING = "torch_missing"
    TRANSFORMERS_MISSING = "transformers_missing"
    IMPORT_FAILED = "import_failed"
    PROBE_FAILED = "probe_failed"
    UNKNOWN = "unknown"


class PyTorchBackendCapabilityState(StrEnum):
    CONFIRMED = "confirmed"
    NOT_OBSERVED = "not_observed"
    UNKNOWN = "unknown"


class PyTorchCapabilityReason(StrEnum):
    RUNTIME_AVAILABLE = "runtime_available"
    PYTHON_MISSING = "python_missing"
    TORCH_MISSING = "torch_missing"
    TRANSFORMERS_MISSING = "transformers_missing"
    IMPORT_FAILED = "import_failed"
    PROBE_FAILED = "probe_failed"
    BACKEND_EXPOSED = "backend_exposed"
    BACKEND_NOT_OBSERVED = "backend_not_observed"
    CAPABILITY_UNKNOWN = "capability_unknown"


class PyTorchRuntimeDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    runtime_id: str
    index: int | None = None
    name: str | None = None
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None


class PyTorchBackendCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    state: PyTorchBackendCapabilityState
    devices: list[PyTorchRuntimeDevice] = Field(default_factory=list)
    reason: PyTorchCapabilityReason
    source: str | None = None
    detail: str | None = None


class PyTorchRuntimeCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_family: RuntimeFamily = RuntimeFamily.PYTORCH_TRANSFORMERS
    python_executable: str | None = None
    runtime_status: PyTorchRuntimeStatus
    torch_version: str | None = None
    transformers_version: str | None = None
    torch_cuda_version: str | None = None
    torch_hip_version: str | None = None
    backend_capabilities: list[PyTorchBackendCapability] = Field(default_factory=list)
    probe_source: str | None = None
    message: str | None = None
    raw_probe: dict[str, object] | None = None


RuntimeCapability = LlamaCppRuntimeCapability | PyTorchRuntimeCapability
RuntimeBackendCapability = LlamaCppBackendCapability | PyTorchBackendCapability


class ExecutionReadinessStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


class ExecutionReadinessReason(StrEnum):
    RUNTIME_AVAILABLE = "runtime_available"
    RUNTIME_MISSING = "runtime_missing"
    RUNTIME_NOT_EXECUTABLE = "runtime_not_executable"
    RUNTIME_IMPORT_FAILED = "runtime_import_failed"
    PROBE_FAILED = "probe_failed"
    SELECTED_BACKEND_EXPOSED = "selected_backend_exposed"
    SELECTED_BACKEND_NOT_EXPOSED = "selected_backend_not_exposed"
    CAPABILITY_UNKNOWN = "capability_unknown"


class ExecutionReadiness(BaseModel):
    """Preflight decision for attempting execution with a selected backend.

    ``READY`` means Jaull has not found a blocking mismatch between the
    hardware backend selection and the observed runtime capability. It is not
    proof that inference will succeed; the definitive execution outcome remains
    ``ExecutionObservation.success``.
    """

    model_config = ConfigDict(frozen=True)

    status: ExecutionReadinessStatus
    reason: ExecutionReadinessReason
    selection: RuntimeBackendSelection
    runtime_capability: RuntimeCapability
    selected_backend_capability: RuntimeBackendCapability | None = None
    message: str | None = None


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
