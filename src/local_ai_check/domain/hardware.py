from __future__ import annotations

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
    warnings: list[str] = Field(default_factory=list)
