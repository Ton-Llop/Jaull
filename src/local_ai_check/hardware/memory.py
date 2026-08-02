from __future__ import annotations

import psutil

from local_ai_check.domain.hardware import MemoryInfo


def detect_memory() -> MemoryInfo:
    vm = psutil.virtual_memory()
    return MemoryInfo(total_bytes=int(vm.total), available_bytes=int(vm.available))
