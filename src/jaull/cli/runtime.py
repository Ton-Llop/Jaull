"""``jaull runtime ...``: runtimes installed and managed by Jaull."""

from __future__ import annotations

import sys

from jaull.advisor.service import AdvisorService
from jaull.domain.hardware import ComputeBackend
from jaull.runtime.llama_cpp_install import LlamaCppInstallError
from jaull.runtime.llama_cpp_release import UnsupportedPlatformError


def run_runtime_install(
    backend: ComputeBackend, *, latest: bool = False, advisor: AdvisorService | None = None,
) -> int:
    resolved = advisor or AdvisorService.default()
    try:
        installation = (
            resolved.install_llama_cpp(backend, latest=True) if latest
            else resolved.install_llama_cpp(backend)
        )
    except (UnsupportedPlatformError, LlamaCppInstallError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    caveat = " (newest upstream build, not validated with Jaull)" if latest else ""
    sys.stdout.write(
        f"llama.cpp {installation.version_text} ({backend.value}) ready{caveat}: "
        f"{installation.llama_cli}\n"
    )
    return 0


__all__ = ["run_runtime_install"]
