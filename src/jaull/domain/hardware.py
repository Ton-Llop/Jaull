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
