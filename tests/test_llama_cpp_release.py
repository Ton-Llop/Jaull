from __future__ import annotations

import re

import pytest

from jaull.domain.hardware import ComputeBackend
from jaull.runtime import policies
from jaull.runtime.llama_cpp_release import (
    UnsupportedPlatformError,
    current_platform,
    package_for,
)


@pytest.mark.parametrize("os_name", ["windows", "linux"])
@pytest.mark.parametrize(
    "backend", [ComputeBackend.VULKAN, ComputeBackend.CUDA, ComputeBackend.CPU],
)
def test_every_supported_combination_has_a_pinned_archive(
    os_name: str, backend: ComputeBackend,
) -> None:
    package = package_for(os_name, "x86_64", backend)
    assert package.build == policies.LLAMA_CPP_PINNED_BUILD
    assert package.archive.name.startswith("llama-")
    for asset in package.assets:
        assert re.fullmatch(r"[0-9a-f]{64}", asset.sha256)
        assert asset.url == f"{policies.LLAMA_CPP_RELEASE_URL}/{package.build}/{asset.name}"


def test_cuda_brings_its_runtime_libraries() -> None:
    package = package_for("windows", "x86_64", ComputeBackend.CUDA)
    assert [asset.name.split("-")[0] for asset in package.assets] == ["llama", "cudart"]


def test_unpinned_combinations_are_refused_not_guessed() -> None:
    with pytest.raises(UnsupportedPlatformError, match="hip"):
        package_for("linux", "x86_64", ComputeBackend.HIP)
    with pytest.raises(UnsupportedPlatformError, match="darwin"):
        package_for("darwin", "arm64", ComputeBackend.CPU)


def test_current_platform_normalises_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    assert current_platform() == ("windows", "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert current_platform() == ("linux", "x86_64")


def test_install_directory_name_is_stable() -> None:
    package = package_for("linux", "x86_64", ComputeBackend.VULKAN)
    assert package.directory_name == f"{policies.LLAMA_CPP_PINNED_BUILD}-linux-x86_64-vulkan"
