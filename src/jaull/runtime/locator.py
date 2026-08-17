"""Central runtime discovery and resolution for local execution backends."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppInstallation,
    PyTorchInstallation,
    RuntimeDiscoveryMetadata,
    RuntimeResolutionStatus,
    RuntimeSource,
)
from jaull.paths import user_data_dir

_LLAMA_CLI = "llama-cli"
_LLAMA_BENCH = "llama-bench"
_REGISTRY_VERSION = 1


@dataclass(frozen=True)
class RuntimeLocatorConfig:
    llama_cli_path: str | Path | None = None
    llama_bench_path: str | Path | None = None
    python_executable: str | Path | None = None


@dataclass(frozen=True)
class RuntimeRegistry:
    path: Path = field(default_factory=lambda: user_data_dir("runtimes.json"))

    def load_llama_cpp(self) -> list[LlamaCppInstallation]:
        payload = self._read()
        raw_items = payload.get("llama_cpp", [])
        if not isinstance(raw_items, list):
            return []
        installations: list[LlamaCppInstallation] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            installation = _llama_from_raw(raw, source=RuntimeSource.REGISTERED)
            if installation is not None and _llama_installation_is_valid(installation):
                installations.append(installation)
        return installations

    def load_pytorch(self) -> list[PyTorchInstallation]:
        payload = self._read()
        raw_items = payload.get("pytorch", [])
        if not isinstance(raw_items, list):
            return []
        installations: list[PyTorchInstallation] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            python = _str_or_none(raw.get("python_executable"))
            if python is None:
                continue
            installation = _pytorch_installation(
                python,
                source=RuntimeSource.REGISTERED,
                detail="registered local runtime",
            )
            if installation.status is RuntimeResolutionStatus.FOUND:
                installations.append(installation)
        return installations

    def register_llama_cpp(self, installation: LlamaCppInstallation) -> None:
        if not _llama_installation_is_valid(installation):
            raise ValueError("Cannot register an invalid llama.cpp installation.")
        payload = self._read()
        existing_llama = payload.get("llama_cpp", [])
        items = (
            [item for item in existing_llama if isinstance(item, dict)]
            if isinstance(existing_llama, list)
            else []
        )
        normalized = _llama_registry_payload(installation)
        items = [
            item
            for item in items
            if _str_or_none(item.get("root")) != normalized["root"]
        ]
        items.append(normalized)
        payload["schema_version"] = _REGISTRY_VERSION
        payload["llama_cpp"] = items
        self._write(payload)

    def register_pytorch(self, installation: PyTorchInstallation) -> None:
        if installation.status is not RuntimeResolutionStatus.FOUND:
            raise ValueError("Cannot register an invalid PyTorch installation.")
        payload = self._read()
        existing_pytorch = payload.get("pytorch", [])
        items = (
            [item for item in existing_pytorch if isinstance(item, dict)]
            if isinstance(existing_pytorch, list)
            else []
        )
        normalized = _pytorch_registry_payload(installation)
        items = [
            item
            for item in items
            if _str_or_none(item.get("python_executable"))
            != normalized["python_executable"]
        ]
        items.append(normalized)
        payload["schema_version"] = _REGISTRY_VERSION
        payload["pytorch"] = items
        self._write(payload)

    def _read(self) -> dict[str, object]:
        if not self.path.is_file():
            return {"schema_version": _REGISTRY_VERSION}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": _REGISTRY_VERSION}
        if not isinstance(raw, dict):
            return {"schema_version": _REGISTRY_VERSION}
        return dict(raw)

    def _write(self, payload: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class RuntimeLocator:
    config: RuntimeLocatorConfig = field(default_factory=RuntimeLocatorConfig)
    registry: RuntimeRegistry | None = field(default_factory=RuntimeRegistry)
    environ: Mapping[str, str] | None = None
    home: Path | None = None

    def discover_llama_cpp(self) -> list[LlamaCppInstallation]:
        explicit = self._explicit_llama_cpp()
        if explicit:
            return explicit
        installations: list[LlamaCppInstallation] = []
        installations.extend(self._managed_llama_cpp())
        installations.extend(self._registered_llama_cpp())
        installations.extend(self._path_llama_cpp())
        installations.extend(self._bounded_llama_cpp_discovery())
        return _dedupe_llama_installations(installations)

    def resolve_llama_cpp(self) -> LlamaCppInstallation:
        explicit = self._explicit_llama_cpp()
        if explicit:
            return explicit[0]
        installations = self.discover_llama_cpp()
        for installation in installations:
            if installation.status is RuntimeResolutionStatus.FOUND:
                return installation
        return LlamaCppInstallation(
            root="",
            source=RuntimeSource.DISCOVERED,
            status=RuntimeResolutionStatus.RUNTIME_NOT_FOUND,
            discovery=RuntimeDiscoveryMetadata(
                detail="llama.cpp runtime not found",
                searched=self._llama_discovery_patterns(),
                message=(
                    "llama.cpp runtime not found. Configure llama-cli/llama-bench, "
                    "register a local runtime, add it to PATH, or place it under "
                    "~/tools/llama.cpp/build-*/bin or ~/llama.cpp/build-*/bin."
                ),
            ),
        )

    def resolve_pytorch(self) -> PyTorchInstallation:
        explicit = self._explicit_pytorch()
        if explicit is not None:
            return explicit
        registered = self._registered_pytorch()
        if registered:
            return registered[0]
        executable = sys.executable
        if executable:
            return _pytorch_installation(
                executable,
                source=RuntimeSource.CURRENT_ENVIRONMENT,
                detail="current Jaull environment",
            )
        return PyTorchInstallation(
            python_executable=None,
            source=RuntimeSource.CURRENT_ENVIRONMENT,
            status=RuntimeResolutionStatus.RUNTIME_NOT_FOUND,
            discovery=RuntimeDiscoveryMetadata(
                detail="current Jaull environment",
                message="Python executable was not found.",
            ),
        )

    def select_llama_cpp(
        self,
        *,
        requested_backend: ComputeBackend,
        readiness_by_cli: Mapping[str, ExecutionReadinessStatus],
    ) -> LlamaCppInstallation:
        fallback: LlamaCppInstallation | None = None
        for installation in self.discover_llama_cpp():
            if installation.status is not RuntimeResolutionStatus.FOUND:
                if fallback is None:
                    fallback = installation
                continue
            if fallback is None:
                fallback = installation
            if installation.llama_cli is None:
                continue
            readiness = readiness_by_cli.get(installation.llama_cli)
            if readiness is ExecutionReadinessStatus.READY:
                return installation
        if fallback is not None:
            if requested_backend is not ComputeBackend.CPU:
                return fallback.model_copy(
                    update={
                        "status": RuntimeResolutionStatus.REQUESTED_BACKEND_NOT_AVAILABLE
                    }
                )
            return fallback
        return self.resolve_llama_cpp()

    def register_llama_cpp(self, installation: LlamaCppInstallation) -> None:
        if self.registry is None:
            raise ValueError("Runtime registry is disabled.")
        self.registry.register_llama_cpp(installation)

    def register_pytorch(self, installation: PyTorchInstallation) -> None:
        if self.registry is None:
            raise ValueError("Runtime registry is disabled.")
        self.registry.register_pytorch(installation)

    def _explicit_llama_cpp(self) -> list[LlamaCppInstallation]:
        cli = self.config.llama_cli_path
        bench = self.config.llama_bench_path
        if cli is None and bench is None:
            return []
        return [
            _llama_installation_from_candidates(
                llama_cli=cli,
                llama_bench=bench,
                source=RuntimeSource.EXPLICIT,
                path_env=self._path_env(),
                detail="explicit configuration",
                configured=True,
            )
        ]

    def _managed_llama_cpp(self) -> list[LlamaCppInstallation]:
        return []

    def _registered_llama_cpp(self) -> list[LlamaCppInstallation]:
        if self.registry is None:
            return []
        return self.registry.load_llama_cpp()

    def _path_llama_cpp(self) -> list[LlamaCppInstallation]:
        cli = _which(_LLAMA_CLI, self._path_env())
        bench = _which(_LLAMA_BENCH, self._path_env())
        installations: list[LlamaCppInstallation] = []
        if cli is not None:
            installations.append(
                _llama_installation_from_candidates(
                    llama_cli=cli,
                    llama_bench=None,
                    source=RuntimeSource.PATH,
                    path_env=self._path_env(),
                    detail="PATH",
                    configured=False,
                )
            )
        if bench is not None:
            installations.append(
                _llama_installation_from_candidates(
                    llama_cli=None,
                    llama_bench=bench,
                    source=RuntimeSource.PATH,
                    path_env=self._path_env(),
                    detail="PATH",
                    configured=False,
                )
            )
        return installations

    def _bounded_llama_cpp_discovery(self) -> list[LlamaCppInstallation]:
        installations: list[LlamaCppInstallation] = []
        for bin_dir in self._bounded_llama_bin_dirs():
            cli = _first_existing_executable(bin_dir, _binary_names(_LLAMA_CLI))
            bench = _first_existing_executable(bin_dir, _binary_names(_LLAMA_BENCH))
            if cli is None and bench is None:
                continue
            installations.append(
                _llama_installation_from_candidates(
                    llama_cli=cli,
                    llama_bench=bench,
                    source=RuntimeSource.DISCOVERED,
                    path_env=self._path_env(),
                    detail="bounded local discovery",
                    configured=False,
                )
            )
        return installations

    def _explicit_pytorch(self) -> PyTorchInstallation | None:
        if self.config.python_executable is None:
            return None
        return _pytorch_installation(
            self.config.python_executable,
            source=RuntimeSource.EXPLICIT,
            detail="explicit configuration",
            path_env=self._path_env(),
            configured=True,
        )

    def _registered_pytorch(self) -> list[PyTorchInstallation]:
        if self.registry is None:
            return []
        return self.registry.load_pytorch()

    def _bounded_llama_bin_dirs(self) -> list[Path]:
        home = self.home or Path.home()
        dirs: list[Path] = []
        for root in (home / "tools" / "llama.cpp", home / "llama.cpp"):
            try:
                matches = sorted(root.glob("build-*/bin"))
            except OSError:
                matches = []
            dirs.extend(path for path in matches if path.is_dir())
        return dirs

    def _llama_discovery_patterns(self) -> list[str]:
        home = self.home or Path.home()
        return [
            str(home / "tools" / "llama.cpp" / "build-*" / "bin"),
            str(home / "llama.cpp" / "build-*" / "bin"),
        ]

    def _path_env(self) -> str | None:
        env = self.environ if self.environ is not None else os.environ
        return env.get("PATH")


def _llama_installation_from_candidates(
    *,
    llama_cli: str | Path | None,
    llama_bench: str | Path | None,
    source: RuntimeSource,
    path_env: str | None,
    detail: str,
    configured: bool,
) -> LlamaCppInstallation:
    cli = _resolve_candidate(llama_cli, path_env=path_env)
    bench = _resolve_candidate(llama_bench, path_env=path_env)

    if cli.status is RuntimeResolutionStatus.FOUND and bench.path is None:
        sibling = _sibling_binary(cli.path, _LLAMA_BENCH)
        if sibling is not None:
            bench = _Candidate(path=sibling, status=RuntimeResolutionStatus.FOUND)
    if bench.status is RuntimeResolutionStatus.FOUND and cli.path is None:
        sibling = _sibling_binary(bench.path, _LLAMA_CLI)
        if sibling is not None:
            cli = _Candidate(path=sibling, status=RuntimeResolutionStatus.FOUND)

    status = _combined_llama_status(cli, bench, configured=configured)
    root = _installation_root(cli.path or bench.path)
    message = cli.message or bench.message
    return LlamaCppInstallation(
        root=str(root) if root is not None else "",
        llama_cli=str(cli.path) if cli.path is not None else None,
        llama_bench=str(bench.path) if bench.path is not None else None,
        source=source,
        status=status,
        discovery=RuntimeDiscoveryMetadata(detail=detail, message=message),
    )


def _combined_llama_status(
    cli: _Candidate,
    bench: _Candidate,
    *,
    configured: bool,
) -> RuntimeResolutionStatus:
    if cli.status is RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE:
        return RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE
    if bench.status is RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE:
        return RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE
    if cli.status is RuntimeResolutionStatus.FOUND or bench.status is RuntimeResolutionStatus.FOUND:
        return RuntimeResolutionStatus.FOUND
    if configured:
        return RuntimeResolutionStatus.CONFIGURED_RUNTIME_MISSING
    return RuntimeResolutionStatus.RUNTIME_NOT_FOUND


def _pytorch_installation(
    python: str | Path,
    *,
    source: RuntimeSource,
    detail: str,
    path_env: str | None = None,
    configured: bool = False,
) -> PyTorchInstallation:
    resolved = _resolve_candidate(python, path_env=path_env)
    status = resolved.status
    if configured and status is RuntimeResolutionStatus.RUNTIME_NOT_FOUND:
        status = RuntimeResolutionStatus.CONFIGURED_RUNTIME_MISSING
    return PyTorchInstallation(
        python_executable=str(resolved.path) if resolved.path is not None else None,
        source=source,
        status=status,
        discovery=RuntimeDiscoveryMetadata(detail=detail, message=resolved.message),
    )


@dataclass(frozen=True)
class _Candidate:
    path: Path | None
    status: RuntimeResolutionStatus
    message: str | None = None


def _resolve_candidate(
    value: str | Path | None,
    *,
    path_env: str | None,
) -> _Candidate:
    if value is None:
        return _Candidate(path=None, status=RuntimeResolutionStatus.RUNTIME_NOT_FOUND)
    raw = str(value)
    candidate = Path(value).expanduser()
    if candidate.parent == Path("."):
        found = _which(raw, path_env)
        if found is None:
            return _Candidate(
                path=None,
                status=RuntimeResolutionStatus.RUNTIME_NOT_FOUND,
                message=f"Executable not found in PATH: {raw!r}.",
            )
        candidate = Path(found)
    if not candidate.exists():
        return _Candidate(
            path=candidate,
            status=RuntimeResolutionStatus.RUNTIME_NOT_FOUND,
            message=f"Executable not found at {candidate}.",
        )
    if candidate.is_dir() or not _is_executable(candidate):
        return _Candidate(
            path=candidate,
            status=RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE,
            message=f"Executable found but not runnable at {candidate}.",
        )
    return _Candidate(path=_normalize_path(candidate), status=RuntimeResolutionStatus.FOUND)


def _which(name: str, path_env: str | None) -> str | None:
    return shutil.which(name, path=path_env)


def _binary_names(base: str) -> tuple[str, ...]:
    if os.name == "nt":
        return (f"{base}.exe", base)
    return (base,)


def _sibling_binary(binary: Path | None, sibling_name: str) -> Path | None:
    if binary is None:
        return None
    return _first_existing_executable(binary.parent, _binary_names(sibling_name))


def _first_existing_executable(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists() and not candidate.is_dir() and _is_executable(candidate):
            return _normalize_path(candidate)
    return None


def _is_executable(path: Path) -> bool:
    if os.name == "nt":
        # Windows has no POSIX executable bit, and several call sites inject a
        # fake execution backend that deliberately uses extensionless temp
        # files. Whether a real file can be spawned is determined by the host
        # execution backend; the locator only rejects directories here.
        return True
    return os.access(path, os.X_OK)


def _normalize_path(path: Path) -> Path:
    # Runtime identity is the path we execute, not necessarily the filesystem
    # target. A virtualenv Python is often a symlink to the base interpreter;
    # dereferencing it would silently leave the virtualenv and lose installed
    # packages such as torch/transformers.
    return Path(os.path.abspath(path.expanduser()))


def _installation_root(binary: Path | None) -> Path | None:
    if binary is None:
        return None
    if binary.parent.name.casefold() == "bin":
        return _normalize_path(binary.parent.parent)
    return _normalize_path(binary.parent)


def _dedupe_llama_installations(
    installations: Sequence[LlamaCppInstallation],
) -> list[LlamaCppInstallation]:
    seen: set[tuple[str | None, str | None]] = set()
    result: list[LlamaCppInstallation] = []
    for installation in installations:
        key = (installation.llama_cli, installation.llama_bench)
        if key in seen:
            continue
        seen.add(key)
        result.append(installation)
    return result


def _llama_installation_is_valid(installation: LlamaCppInstallation) -> bool:
    if installation.status is not RuntimeResolutionStatus.FOUND:
        return False
    if installation.llama_cli is not None:
        cli = _resolve_candidate(installation.llama_cli, path_env=None)
        if cli.status is not RuntimeResolutionStatus.FOUND:
            return False
    if installation.llama_bench is not None:
        bench = _resolve_candidate(installation.llama_bench, path_env=None)
        if bench.status is not RuntimeResolutionStatus.FOUND:
            return False
    return installation.llama_cli is not None or installation.llama_bench is not None


def _llama_registry_payload(installation: LlamaCppInstallation) -> dict[str, str | None]:
    return {
        "root": str(_normalize_path(Path(installation.root))),
        "llama_cli": _normalized_optional(installation.llama_cli),
        "llama_bench": _normalized_optional(installation.llama_bench),
    }


def _pytorch_registry_payload(
    installation: PyTorchInstallation,
) -> dict[str, str | None]:
    return {
        "python_executable": _normalized_optional(installation.python_executable),
    }


def _normalized_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return str(_normalize_path(Path(value)))


def _llama_from_raw(
    raw: Mapping[str, object],
    *,
    source: RuntimeSource,
) -> LlamaCppInstallation | None:
    cli = _str_or_none(raw.get("llama_cli"))
    bench = _str_or_none(raw.get("llama_bench"))
    if cli is None and bench is None:
        return None
    return _llama_installation_from_candidates(
        llama_cli=cli,
        llama_bench=bench,
        source=source,
        path_env=None,
        detail="registered local runtime",
        configured=False,
    )


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "RuntimeLocator",
    "RuntimeLocatorConfig",
    "RuntimeRegistry",
]
