"""Which official llama.cpp archive Jaull installs on this machine.

The catalog is pinned (``runtime/policies.py``): one upstream build, the archive
for each supported OS and backend, and the SHA-256 the upstream release
publishes. A combination missing from it is refused, never guessed.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from jaull.domain.hardware import ComputeBackend
from jaull.runtime import policies

_OS_NAMES = {"windows": "windows", "linux": "linux"}
_ARCH_NAMES = {"amd64": "x86_64", "x86_64": "x86_64"}


class UnsupportedPlatformError(ValueError):
    """No pinned llama.cpp archive exists for this OS, architecture and backend."""


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    sha256: str
    url: str


@dataclass(frozen=True)
class LlamaCppPackage:
    build: str
    os_name: str
    arch: str
    backend: ComputeBackend
    assets: tuple[ReleaseAsset, ...]

    @property
    def archive(self) -> ReleaseAsset:
        return self.assets[0]

    @property
    def directory_name(self) -> str:
        return f"{self.build}-{self.os_name}-{self.arch}-{self.backend.value}"


def current_platform() -> tuple[str, str]:
    """The (os, arch) pair the catalog is keyed by, for this machine."""
    system = platform.system().casefold()
    machine = platform.machine().casefold()
    return _OS_NAMES.get(system, system), _ARCH_NAMES.get(machine, machine)


def package_for(os_name: str, arch: str, backend: ComputeBackend) -> LlamaCppPackage:
    entries = policies.LLAMA_CPP_RELEASE_ASSETS.get((os_name, arch, backend.value))
    if entries is None:
        supported = ", ".join("/".join(key) for key in sorted(policies.LLAMA_CPP_RELEASE_ASSETS))
        raise UnsupportedPlatformError(
            f"No pinned llama.cpp build for {os_name}/{arch}/{backend.value}. "
            f"Supported: {supported}."
        )
    build = policies.LLAMA_CPP_PINNED_BUILD
    return LlamaCppPackage(
        build=build,
        os_name=os_name,
        arch=arch,
        backend=backend,
        assets=tuple(
            ReleaseAsset(
                name=name, sha256=sha256,
                url=f"{policies.LLAMA_CPP_RELEASE_URL}/{build}/{name}",
            )
            for name, sha256 in entries
        ),
    )


__all__ = [
    "LlamaCppPackage",
    "ReleaseAsset",
    "UnsupportedPlatformError",
    "current_platform",
    "package_for",
]
