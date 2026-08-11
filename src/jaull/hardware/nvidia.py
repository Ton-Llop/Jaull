"""NVIDIA GPU detection via NVML.

The NVML library (`pynvml`, provided by the `nvidia-ml-py` package) is optional at runtime:
if the driver or the library is not present the detector returns an empty list of GPUs
together with an explanatory warning, rather than raising.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol, cast

from jaull.domain.hardware import GpuInfo

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


class _NvmlProcessInfo(Protocol):
    pid: int
    usedGpuMemory: int


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


class NvidiaProcessMemorySampler:
    """Sample NVIDIA device memory attributed to one process PID.

    NVML process accounting is optional and driver-dependent. This sampler
    therefore returns ``None`` whenever process memory cannot be attributed
    reliably, rather than making command execution fail.
    """

    def __init__(self, provider: NvmlProvider | None = None) -> None:
        self._nvml = provider if provider is not None else _load_pynvml()
        self._handles: list[object] = []
        self._active = False
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlInit()
            count = self._nvml.nvmlDeviceGetCount()
            self._handles = [
                self._nvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(count)
            ]
            self._active = True
        except Exception as exc:
            logger.debug("NVML process sampler initialization failed", exc_info=exc)
            self.close()

    def sample_pid_bytes(self, pid: int) -> int | None:
        if not self._active or self._nvml is None:
            return None

        total = 0
        found = False
        for handle in self._handles:
            device_peak: int | None = None
            for process in self._running_processes(handle):
                try:
                    if int(process.pid) != pid:
                        continue
                    used = int(process.usedGpuMemory)
                except Exception:
                    continue
                if used < 0:
                    continue
                device_peak = used if device_peak is None else max(device_peak, used)
                found = True
            if device_peak is not None:
                total += device_peak
        return total if found else None

    def close(self) -> None:
        if self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
        self._active = False
        self._handles = []

    def _running_processes(self, handle: object) -> list[_NvmlProcessInfo]:
        processes: list[_NvmlProcessInfo] = []
        for names in (
            (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            ),
            (
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v2",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ),
        ):
            processes.extend(self._first_process_query_result(handle, names))
        return processes

    def _first_process_query_result(
        self, handle: object, names: tuple[str, ...]
    ) -> list[_NvmlProcessInfo]:
        for name in names:
            method = getattr(cast(Any, self._nvml), name, None)
            if method is None:
                continue
            try:
                raw = method(handle)
            except Exception as exc:
                logger.debug("NVML process query %s failed", name, exc_info=exc)
                continue
            return cast(list[_NvmlProcessInfo], raw)
        return []
