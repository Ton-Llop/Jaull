from __future__ import annotations

import psutil

from jaull.domain.hardware import StorageInfo

# WSL exposes many synthetic mounts from the Windows host / Docker Desktop that add
# noise without informing capacity planning. Skip them by prefix.
_SKIP_PREFIXES = ("/mnt/wsl", "/mnt/wslg", "/snap", "/run")


def detect_storage() -> list[StorageInfo]:
    results: list[StorageInfo] = []
    seen: set[str] = set()
    for part in psutil.disk_partitions(all=False):
        mp = part.mountpoint
        if not mp or mp in seen:
            continue
        if any(mp.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        try:
            usage = psutil.disk_usage(mp)
        except (PermissionError, OSError):
            continue
        seen.add(mp)
        results.append(
            StorageInfo(
                mountpoint=mp,
                total_bytes=int(usage.total),
                available_bytes=int(usage.free),
            )
        )
    return results
