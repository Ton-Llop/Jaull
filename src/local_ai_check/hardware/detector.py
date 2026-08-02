from __future__ import annotations

import platform

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.hardware.cpu import detect_cpu
from local_ai_check.hardware.memory import detect_memory
from local_ai_check.hardware.nvidia import NvmlProvider, detect_nvidia_gpus
from local_ai_check.hardware.storage import detect_storage


def detect_hardware(nvml_provider: NvmlProvider | None = None) -> HardwareProfile:
    warnings: list[str] = []

    cpu = detect_cpu()
    memory = detect_memory()
    storage = detect_storage()
    if not storage:
        warnings.append("No storage partitions could be enumerated.")

    nvidia = detect_nvidia_gpus(nvml_provider)
    warnings.extend(nvidia.warnings)

    return HardwareProfile(
        os=_os_display_name(),
        os_version=platform.version(),
        arch=platform.machine() or "unknown",
        cpu=cpu,
        memory=memory,
        storage=storage,
        gpus=nvidia.gpus,
        warnings=warnings,
    )


def _os_display_name() -> str:
    system = platform.system()
    release = platform.release()
    if system and release:
        return f"{system} {release}"
    return system or "unknown"
