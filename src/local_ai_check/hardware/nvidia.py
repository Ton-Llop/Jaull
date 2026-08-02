"""NVIDIA GPU detection via NVML.

The NVML library (`pynvml`, provided by the `nvidia-ml-py` package) is optional at runtime:
if the driver or the library is not present the detector returns an empty list of GPUs
together with an explanatory warning, rather than raising.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Protocol, cast

from local_ai_check.domain.hardware import GpuInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NvidiaProbe:
    gpus: list[GpuInfo]
    warnings: list[str]
    driver_version: str | None
    cuda_version: str | None


class _NvmlMemoryInfo(Protocol):
    total: int
    free: int


class NvmlProvider(Protocol):
    """Thin protocol over the parts of pynvml we call — enables mocking in tests."""

    def nvmlInit(self) -> None: ...
    def nvmlShutdown(self) -> None: ...
    def nvmlSystemGetDriverVersion(self) -> str | bytes: ...
    def nvmlSystemGetCudaDriverVersion(self) -> int: ...
    def nvmlDeviceGetCount(self) -> int: ...
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...
    def nvmlDeviceGetName(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _NvmlMemoryInfo: ...


def _load_pynvml() -> NvmlProvider | None:
    try:
        import pynvml
    except ImportError:
        return None
    return cast(NvmlProvider, pynvml)


def _decode(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _format_cuda_version(raw: int) -> str:
    # NVML returns e.g. 12040 for CUDA 12.4
    major = raw // 1000
    minor = (raw % 1000) // 10
    return f"{major}.{minor}"


def detect_nvidia_gpus(provider: NvmlProvider | None = None) -> NvidiaProbe:
    nvml = provider if provider is not None else _load_pynvml()
    if nvml is None:
        return NvidiaProbe(
            gpus=[],
            warnings=["NVML library is not installed; skipping NVIDIA GPU detection."],
            driver_version=None,
            cuda_version=None,
        )

    warnings: list[str] = []
    try:
        nvml.nvmlInit()
    except Exception as exc:  # NVMLError is not a subclass of a specific stdlib class
        logger.debug("nvmlInit failed", exc_info=exc)
        return NvidiaProbe(
            gpus=[],
            warnings=[
                "NVIDIA driver not detected (NVML initialization failed); "
                "continuing with CPU information."
            ],
            driver_version=None,
            cuda_version=None,
        )

    try:
        try:
            driver = _decode(nvml.nvmlSystemGetDriverVersion())
        except Exception:
            driver = None
            warnings.append("Could not read NVIDIA driver version.")

        try:
            cuda = _format_cuda_version(nvml.nvmlSystemGetCudaDriverVersion())
        except Exception:
            cuda = None
            warnings.append("Could not read CUDA driver version.")

        gpus: list[GpuInfo] = []
        try:
            count = nvml.nvmlDeviceGetCount()
        except Exception as exc:
            logger.debug("nvmlDeviceGetCount failed", exc_info=exc)
            warnings.append("Could not enumerate NVIDIA devices.")
            count = 0

        for i in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = _decode(nvml.nvmlDeviceGetName(handle))
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append(
                    GpuInfo(
                        name=name,
                        vram_total_bytes=int(mem.total),
                        vram_available_bytes=int(mem.free),
                        driver_version=driver,
                        cuda_version=cuda,
                    )
                )
            except Exception as exc:
                logger.debug("Failed to probe GPU %d", i, exc_info=exc)
                warnings.append(f"Could not read information for GPU index {i}.")

        return NvidiaProbe(
            gpus=gpus, warnings=warnings, driver_version=driver, cuda_version=cuda
        )
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()
