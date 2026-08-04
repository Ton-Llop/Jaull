"""Collect environment diagnostics (Python, network, HF, NVML, cache, deps)."""

from __future__ import annotations

import importlib
import os
import platform
import socket
import tempfile
from pathlib import Path

from jaull.domain.enums import DiagnosticStatus
from jaull.domain.model import DiagnosticResult
from jaull.hardware.nvidia import detect_nvidia_gpus

_REQUIRED_DEPENDENCIES = (
    "typer",
    "rich",
    "pydantic",
    "huggingface_hub",
    "psutil",
)


def collect_diagnostics() -> list[DiagnosticResult]:
    return [
        _check_python(),
        _check_internet(),
        _check_huggingface(),
        _check_nvml(),
        _check_nvidia_gpu(),
        _check_cache_writable(),
        *_check_dependencies(),
    ]


def _check_python() -> DiagnosticResult:
    # ``requires-python = ">=3.12"`` gates installation, so if this code runs
    # the version constraint is already satisfied.
    return DiagnosticResult(
        name="Python version",
        status=DiagnosticStatus.OK,
        detail=platform.python_version(),
    )


def _check_internet() -> DiagnosticResult:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=3):
            return DiagnosticResult(
                name="Internet reachable",
                status=DiagnosticStatus.OK,
                detail="ok",
            )
    except OSError as exc:
        return DiagnosticResult(
            name="Internet reachable",
            status=DiagnosticStatus.WARN,
            detail=f"unreachable ({exc.__class__.__name__})",
        )


def _check_huggingface() -> DiagnosticResult:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=5):
            return DiagnosticResult(
                name="Hugging Face API reachable",
                status=DiagnosticStatus.OK,
                detail="huggingface.co:443",
            )
    except OSError as exc:
        return DiagnosticResult(
            name="Hugging Face API reachable",
            status=DiagnosticStatus.WARN,
            detail=f"unreachable ({exc.__class__.__name__})",
        )


def _check_nvml() -> DiagnosticResult:
    try:
        importlib.import_module("pynvml")
    except ImportError:
        return DiagnosticResult(
            name="NVML library",
            status=DiagnosticStatus.WARN,
            detail="pynvml not installed (GPU detection disabled)",
        )
    return DiagnosticResult(
        name="NVML library",
        status=DiagnosticStatus.OK,
        detail="installed",
    )


def _check_nvidia_gpu() -> DiagnosticResult:
    probe = detect_nvidia_gpus()
    if probe.gpus:
        names = ", ".join(g.name for g in probe.gpus)
        return DiagnosticResult(
            name="NVIDIA GPU",
            status=DiagnosticStatus.OK,
            detail=names,
        )
    return DiagnosticResult(
        name="NVIDIA GPU",
        status=DiagnosticStatus.WARN,
        detail="not detected",
    )


def _check_cache_writable() -> DiagnosticResult:
    cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, delete=True):
            pass
    except OSError as exc:
        return DiagnosticResult(
            name="Cache writable",
            status=DiagnosticStatus.FAIL,
            detail=f"{cache} not writable ({exc})",
        )
    return DiagnosticResult(
        name="Cache writable",
        status=DiagnosticStatus.OK,
        detail=str(cache),
    )


def _check_dependencies() -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    for name in _REQUIRED_DEPENDENCIES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            results.append(
                DiagnosticResult(
                    name=f"Dependency: {name}",
                    status=DiagnosticStatus.FAIL,
                    detail="missing",
                )
            )
            continue
        version = getattr(module, "__version__", "unknown")
        results.append(
            DiagnosticResult(
                name=f"Dependency: {name}",
                status=DiagnosticStatus.OK,
                detail=version,
            )
        )
    return results


__all__ = ["collect_diagnostics"]
