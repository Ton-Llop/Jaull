from __future__ import annotations

import json
import os
from pathlib import Path

from jaull.advisor.service import AdvisorService
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    RuntimeResolutionStatus,
    RuntimeSource,
)
from jaull.runtime.locator import RuntimeLocator, RuntimeLocatorConfig, RuntimeRegistry
from jaull.workflow.container import ServiceContainer


def _norm(path: Path) -> str:
    return str(Path(os.path.abspath(path.expanduser())))


def test_explicit_llama_cli_wins_over_path_and_discovery(tmp_path: Path) -> None:
    explicit = _exe(tmp_path / "explicit" / "bin" / "llama-cli")
    path_cli = _exe(tmp_path / "path" / "llama-cli")
    discovered = _exe(
        tmp_path / "home" / "tools" / "llama.cpp" / "build-cpu" / "bin" / "llama-cli"
    )
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_cli_path=explicit),
        registry=None,
        environ={"PATH": str(path_cli.parent)},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert discovered.is_file()
    assert installation.llama_cli == _norm(explicit)
    assert installation.source is RuntimeSource.EXPLICIT


def test_explicit_missing_path_preserves_configured_missing_semantics(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing" / "llama-cli"
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_cli_path=missing),
        registry=None,
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.status is RuntimeResolutionStatus.CONFIGURED_RUNTIME_MISSING
    assert installation.llama_cli == str(missing)


def test_path_llama_cli_discovery(tmp_path: Path) -> None:
    cli = _exe(tmp_path / "path bin" / "llama-cli")
    locator = RuntimeLocator(
        registry=None,
        environ={"PATH": str(cli.parent)},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.source is RuntimeSource.PATH
    assert installation.llama_cli == _norm(cli)


def test_bounded_local_discovery_finds_build_cpu_without_arbitrary_scan(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    random_cli = _exe(home / "random" / "deep" / "llama-cli")
    build_cli = _exe(home / "tools" / "llama.cpp" / "build-cpu" / "bin" / "llama-cli")
    locator = RuntimeLocator(registry=None, environ={"PATH": ""}, home=home)

    installation = locator.resolve_llama_cpp()

    assert random_cli.is_file()
    assert installation.source is RuntimeSource.DISCOVERED
    assert installation.llama_cli == _norm(build_cli)


def test_sibling_llama_bench_is_associated_with_llama_cli(tmp_path: Path) -> None:
    cli = _exe(tmp_path / "llama.cpp" / "build-cpu" / "bin" / "llama-cli")
    bench = _exe(cli.parent / "llama-bench")
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_cli_path=cli),
        registry=None,
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.llama_cli == _norm(cli)
    assert installation.llama_bench == _norm(bench)
    assert Path(installation.llama_cli).parent == Path(installation.llama_bench).parent


def test_sibling_llama_cli_is_associated_from_llama_bench(tmp_path: Path) -> None:
    bench = _exe(tmp_path / "llama.cpp" / "build-cpu" / "bin" / "llama-bench")
    cli = _exe(bench.parent / "llama-cli")
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_bench_path=bench),
        registry=None,
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.llama_cli == _norm(cli)
    assert installation.llama_bench == _norm(bench)


def test_multiple_llama_cpp_installations_can_coexist(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cpu = _exe(home / "tools" / "llama.cpp" / "build-cpu" / "bin" / "llama-cli")
    cuda = _exe(home / "tools" / "llama.cpp" / "build-cuda" / "bin" / "llama-cli")
    locator = RuntimeLocator(registry=None, environ={"PATH": ""}, home=home)

    installations = locator.discover_llama_cpp()

    assert {item.llama_cli for item in installations} == {
        _norm(cpu),
        _norm(cuda),
    }


def test_cuda_selection_uses_capability_readiness_not_path_name(tmp_path: Path) -> None:
    home = tmp_path / "home"
    misleading = _exe(
        home / "tools" / "llama.cpp" / "build-cuda" / "bin" / "llama-cli"
    )
    actual = _exe(home / "tools" / "llama.cpp" / "build-cpu" / "bin" / "llama-cli")
    locator = RuntimeLocator(registry=None, environ={"PATH": ""}, home=home)

    selected = locator.select_llama_cpp(
        requested_backend=ComputeBackend.CUDA,
        readiness_by_cli={
            _norm(misleading): ExecutionReadinessStatus.NOT_READY,
            _norm(actual): ExecutionReadinessStatus.READY,
        },
    )

    assert selected.llama_cli == _norm(actual)


def test_build_cuda_name_is_not_selected_for_cuda_without_ready_capability(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    misleading = _exe(
        home / "tools" / "llama.cpp" / "build-cuda" / "bin" / "llama-cli"
    )
    locator = RuntimeLocator(registry=None, environ={"PATH": ""}, home=home)

    selected = locator.select_llama_cpp(
        requested_backend=ComputeBackend.CUDA,
        readiness_by_cli={_norm(misleading): ExecutionReadinessStatus.NOT_READY},
    )

    assert selected.status is RuntimeResolutionStatus.REQUESTED_BACKEND_NOT_AVAILABLE


def test_registered_installation_is_loaded_when_valid(tmp_path: Path) -> None:
    cli = _exe(tmp_path / "registered" / "bin" / "llama-cli")
    registry_path = tmp_path / "runtimes.json"
    registry_path.write_text(
        json.dumps({"llama_cpp": [{"root": str(cli.parent.parent), "llama_cli": str(cli)}]}),
        encoding="utf-8",
    )
    locator = RuntimeLocator(
        registry=RuntimeRegistry(registry_path),
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.source is RuntimeSource.REGISTERED
    assert installation.llama_cli == _norm(cli)


def test_registered_installation_that_disappeared_is_ignored(tmp_path: Path) -> None:
    registry_path = tmp_path / "runtimes.json"
    registry_path.write_text(
        json.dumps(
            {
                "llama_cpp": [
                    {
                        "root": str(tmp_path / "missing"),
                        "llama_cli": str(tmp_path / "missing" / "llama-cli"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    locator = RuntimeLocator(
        registry=RuntimeRegistry(registry_path),
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.status is RuntimeResolutionStatus.RUNTIME_NOT_FOUND


def test_explicit_python_is_resolved(tmp_path: Path) -> None:
    python = _exe(tmp_path / "venv with spaces" / "bin" / "python")
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(python_executable=python),
        registry=None,
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_pytorch()

    assert installation.source is RuntimeSource.EXPLICIT
    assert installation.python_executable == _norm(python)


def test_pytorch_resolution_preserves_virtualenv_python_symlink(
    tmp_path: Path,
) -> None:
    system_python = _exe(tmp_path / "system" / "python")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    try:
        venv_python.symlink_to(system_python)
    except OSError:
        return
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(python_executable=venv_python),
        registry=None,
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_pytorch()

    assert installation.python_executable == _norm(venv_python)
    assert installation.python_executable != _norm(system_python)


def test_not_executable_runtime_is_reported(tmp_path: Path) -> None:
    cli = tmp_path / "llama.cpp" / "build-cpu" / "bin" / "llama-cli"
    cli.mkdir(parents=True)
    locator = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_cli_path=cli),
        registry=None,
        environ={"PATH": ""},
        home=tmp_path / "home",
    )

    installation = locator.resolve_llama_cpp()

    assert installation.status is RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE


def test_advisor_uses_same_llama_cpp_installation_for_run_and_benchmark(
    tmp_path: Path,
) -> None:
    cli = _exe(tmp_path / "llama.cpp" / "build-cpu" / "bin" / "llama-cli")
    bench = _exe(cli.parent / "llama-bench")
    advisor = AdvisorService(
        services=_services(),
        runtime_locator=RuntimeLocator(
            config=RuntimeLocatorConfig(llama_cli_path=cli),
            registry=None,
            environ={"PATH": ""},
            home=tmp_path / "home",
        ),
    )

    assert advisor._llama_cpp_runner()._llama_cli == _norm(cli)
    assert advisor._llama_bench_runner()._llama_bench == _norm(bench)


def test_advisor_uses_same_python_for_probe_run_and_benchmark(tmp_path: Path) -> None:
    python = _exe(tmp_path / "venv" / "bin" / "python")
    advisor = AdvisorService(
        services=_services(),
        runtime_locator=RuntimeLocator(
            config=RuntimeLocatorConfig(python_executable=python),
            registry=None,
            environ={"PATH": ""},
            home=tmp_path / "home",
        ),
    )

    assert advisor._resolved_pytorch_installation().python_executable == _norm(python)
    assert advisor._transformers_runner()._python == _norm(python)
    assert advisor._transformers_benchmark_runner()._python == _norm(python)


def test_advisor_preserves_symlinked_python_for_probe_run_and_benchmark(
    tmp_path: Path,
) -> None:
    system_python = _exe(tmp_path / "system" / "python")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    try:
        venv_python.symlink_to(system_python)
    except OSError:
        return
    advisor = AdvisorService(
        services=_services(),
        runtime_locator=RuntimeLocator(
            config=RuntimeLocatorConfig(python_executable=venv_python),
            registry=None,
            environ={"PATH": ""},
            home=tmp_path / "home",
        ),
    )

    assert advisor._resolved_pytorch_installation().python_executable == _norm(
        venv_python
    )
    assert advisor._transformers_runner()._python == _norm(venv_python)
    assert advisor._transformers_benchmark_runner()._python == _norm(venv_python)


def _exe(path: Path) -> Path:
    if os.name == "nt" and path.suffix == "":
        path = path.with_suffix(".exe")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
