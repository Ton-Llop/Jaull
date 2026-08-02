from __future__ import annotations

import platform

import psutil

from local_ai_check.domain.hardware import CpuInfo


def detect_cpu() -> CpuInfo:
    model = _cpu_model_name()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    return CpuInfo(model=model, physical_cores=physical, logical_cores=logical)


def _cpu_model_name() -> str | None:
    # platform.processor() returns useful info on Windows; on Linux it is often the arch string.
    proc = platform.processor()
    if proc and proc != platform.machine():
        return proc

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return proc or None

    return proc or None
