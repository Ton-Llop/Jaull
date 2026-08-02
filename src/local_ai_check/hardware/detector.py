from __future__ import annotations

import platform
from collections.abc import Callable

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.hardware.cpu import detect_cpu
from local_ai_check.hardware.memory import detect_memory
from local_ai_check.hardware.nvidia import NvmlProvider, detect_nvidia_gpus
from local_ai_check.hardware.storage import detect_storage

# Keys reported to ``on_step``, in the order they complete. The guided UI uses
# these to advance a progress list as each real probe returns — there is no
# artificial pacing, a step turns green exactly when its probe finishes.
STEP_OS = "os"
STEP_CPU = "cpu"
STEP_GPU = "gpu"
STEP_STORAGE = "storage"
STEP_PROFILE = "profile"


def detect_hardware(
    nvml_provider: NvmlProvider | None = None,
    on_step: Callable[[str], None] | None = None,
) -> HardwareProfile:
    warnings: list[str] = []
    notify = on_step if on_step is not None else _ignore

    os_name = _os_display_name()
    notify(STEP_OS)

    cpu = detect_cpu()
    memory = detect_memory()
    notify(STEP_CPU)

    nvidia = detect_nvidia_gpus(nvml_provider)
    warnings.extend(nvidia.warnings)
    notify(STEP_GPU)

    storage = detect_storage()
    if not storage:
        warnings.append("No storage partitions could be enumerated.")
    notify(STEP_STORAGE)

    profile = HardwareProfile(
        os=os_name,
        os_version=platform.version(),
        arch=platform.machine() or "unknown",
        cpu=cpu,
        memory=memory,
        storage=storage,
        gpus=nvidia.gpus,
        warnings=warnings,
    )
    notify(STEP_PROFILE)
    return profile


def _ignore(step: str) -> None:
    del step


def _os_display_name() -> str:
    system = platform.system()
    release = platform.release()
    if system and release:
        return f"{system} {release}"
    return system or "unknown"
