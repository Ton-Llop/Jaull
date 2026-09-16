from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CpuInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None


class MemoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_bytes: int
    available_bytes: int


class StorageInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    mountpoint: str
    total_bytes: int
    available_bytes: int


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    OTHER = "other"
    UNKNOWN = "unknown"


class AcceleratorType(StrEnum):
    DEDICATED = "dedicated"
    INTEGRATED = "integrated"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


class ComputeBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    VULKAN = "vulkan"
    HIP = "hip"


class BackendAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class BackendAvailabilityReason(StrEnum):
    PROBE_AVAILABLE = "probe_available"
    PROBE_FAILED = "probe_failed"
    NOT_CHECKED = "not_checked"
    VENDOR_NOT_SUPPORTED = "vendor_not_supported"
    SOFTWARE_RENDERER_ONLY = "software_renderer_only"
    RUNTIME_NOT_INSTALLED = "runtime_not_installed"
    DEVICE_NOT_VISIBLE = "device_not_visible"
    NOT_IMPLEMENTED = "not_implemented"


class ComputeBackendInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: ComputeBackend
    availability: BackendAvailability
    reason: BackendAvailabilityReason | None = None
    source: str | None = None
    detail: str | None = None
    api_version: str | None = None
    device_name: str | None = None
    driver_name: str | None = None
    software_renderer: bool = False


class AcceleratorProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    vendor: AcceleratorVendor = AcceleratorVendor.UNKNOWN
    type: AcceleratorType = AcceleratorType.UNKNOWN
    vendor_id: str | None = None
    device_id: str | None = None
    pci_bus_id: str | None = None
    uuid: str | None = None
    dedicated_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    shared_memory: bool = False
    detection_sources: list[str] = Field(default_factory=list)
    backends: list[ComputeBackendInfo] = Field(default_factory=list)


class GpuInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    vram_total_bytes: int
    vram_available_bytes: int
    driver_version: str | None = None
    cuda_version: str | None = None


class HardwareProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    os: str
    os_version: str | None = None
    arch: str
    cpu: CpuInfo
    memory: MemoryInfo
    storage: list[StorageInfo] = Field(default_factory=list)
    gpus: list[GpuInfo] = Field(default_factory=list)
    accelerators: list[AcceleratorProfile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def planning_accelerator_memory_bytes(hardware: HardwareProfile) -> int | None:
    """Return the largest discrete accelerator capacity for planning.

    ``gpus`` is the legacy NVIDIA/NVML view. ``accelerators`` is the
    backend-neutral view and carries Vulkan/HIP devices. They describe the
    same physical resource when both are present, so selecting the maximum
    preserves Jaull's existing single-device policy without combining pools.
    Shared-memory accelerators are deliberately excluded: their capacity is
    system RAM, not a second VRAM pool.
    """

    capacities = [gpu.vram_total_bytes for gpu in hardware.gpus]
    capacities.extend(
        accelerator.dedicated_memory_bytes
        for accelerator in hardware.accelerators
        if accelerator.type is AcceleratorType.DEDICATED
        and not accelerator.shared_memory
        and accelerator.dedicated_memory_bytes is not None
    )
    return max(capacities, default=None)


def available_accelerator_memory_bytes(hardware: HardwareProfile) -> int | None:
    """Return the largest observed discrete accelerator availability.

    Capacity is not substituted for availability. A detector that only knows
    total VRAM can participate in planning, but cannot claim a run-now budget
    for Hardware Fit analysis.
    """

    availability = [gpu.vram_available_bytes for gpu in hardware.gpus]
    availability.extend(
        accelerator.available_memory_bytes
        for accelerator in hardware.accelerators
        if accelerator.type is AcceleratorType.DEDICATED
        and not accelerator.shared_memory
        and accelerator.available_memory_bytes is not None
    )
    return max(availability, default=None)


def has_usable_accelerator(hardware: HardwareProfile) -> bool:
    """Whether hardware detection exposes a non-software compute backend.

    ``gpus`` remains a valid NVIDIA signal for legacy serialized profiles that
    predate the generic ``accelerators`` mirror.
    """

    return bool(hardware.gpus) or any(
        accelerator.type is not AcceleratorType.SOFTWARE
        and any(
            backend.availability is BackendAvailability.AVAILABLE
            for backend in accelerator.backends
        )
        for accelerator in hardware.accelerators
    )
