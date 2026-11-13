from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeResolutionStatus, RuntimeSource
from jaull.runtime.llama_cpp_install import (
    LlamaCppInstallError,
    download_verified,
    extract_archive,
    install_llama_cpp,
)
from jaull.runtime.llama_cpp_release import LlamaCppPackage, ReleaseAsset


def _client(files: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name not in files:
            return httpx.Response(404)
        return httpx.Response(200, content=files[name])

    return httpx.Client(transport=httpx.MockTransport(handler))


def _asset(name: str, payload: bytes) -> ReleaseAsset:
    return ReleaseAsset(
        name=name, sha256=hashlib.sha256(payload).hexdigest(),
        url=f"https://example.invalid/b1/{name}",
    )


def test_a_matching_download_is_kept(tmp_path: Path) -> None:
    asset = _asset("a.zip", b"payload")
    path = download_verified(_client({"a.zip": b"payload"}), asset, tmp_path)
    assert path.read_bytes() == b"payload"


def test_a_tampered_download_is_deleted_and_refused(tmp_path: Path) -> None:
    asset = _asset("a.zip", b"payload")
    with pytest.raises(LlamaCppInstallError, match="SHA-256"):
        download_verified(_client({"a.zip": b"tampered"}), asset, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_http_error_is_an_install_error(tmp_path: Path) -> None:
    with pytest.raises(LlamaCppInstallError, match="404"):
        download_verified(_client({}), _asset("a.zip", b"x"), tmp_path)
    assert list(tmp_path.iterdir()) == []


# -- extraction --------------------------------------------------------------

def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
    return path


ArchiveMaker = Callable[[Path, dict[str, bytes]], Path]
MAKERS = [(_zip, ".zip"), (_tar, ".tar.gz")]


@pytest.mark.parametrize(("maker", "suffix"), MAKERS)
def test_archives_unpack_into_the_target(
    tmp_path: Path, maker: ArchiveMaker, suffix: str,
) -> None:
    archive = maker(tmp_path / f"a{suffix}", {"build/bin/llama-cli": b"cli"})
    extract_archive(archive, tmp_path / "out")
    assert (tmp_path / "out" / "build" / "bin" / "llama-cli").read_bytes() == b"cli"


@pytest.mark.parametrize(("maker", "suffix"), MAKERS)
def test_members_escaping_the_target_are_refused(
    tmp_path: Path, maker: ArchiveMaker, suffix: str,
) -> None:
    archive = maker(tmp_path / f"a{suffix}", {"../evil": b"x"})
    with pytest.raises(LlamaCppInstallError, match="outside"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil").exists()


def test_unknown_archive_types_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.rar"
    path.write_bytes(b"x")
    with pytest.raises(LlamaCppInstallError, match="archive type"):
        extract_archive(path, tmp_path / "out")


# -- installing a package ----------------------------------------------------

BINARIES = {
    "build/bin/llama-cli": b"c", "build/bin/llama-bench": b"b", "build/bin/llama-server": b"s",
}
ARCHIVE_NAME = "llama-b1-bin-ubuntu-x64.tar.gz"


def _package(tmp_path: Path, members: dict[str, bytes]) -> tuple[LlamaCppPackage, httpx.Client]:
    payload = _tar(tmp_path / "src.tar.gz", members).read_bytes()
    package = LlamaCppPackage(
        build="b1", os_name="linux", arch="x86_64", backend=ComputeBackend.CPU,
        assets=(_asset(ARCHIVE_NAME, payload),),
    )
    return package, _client({ARCHIVE_NAME: payload})


def test_install_unpacks_and_reports_a_managed_installation(tmp_path: Path) -> None:
    package, client = _package(tmp_path, BINARIES)
    installation = install_llama_cpp(package, root=tmp_path / "managed", client=client)
    assert installation.source is RuntimeSource.MANAGED
    assert installation.status is RuntimeResolutionStatus.FOUND
    assert installation.llama_cli is not None and installation.llama_cli.endswith("llama-cli")
    assert installation.version_text == "b1"
    assert (tmp_path / "managed" / package.directory_name).is_dir()
    assert not list((tmp_path / "managed").rglob("*.tar.gz"))  # downloads cleaned up


def test_install_is_idempotent(tmp_path: Path) -> None:
    package, client = _package(tmp_path, BINARIES)
    first = install_llama_cpp(package, root=tmp_path / "managed", client=client)
    offline = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    second = install_llama_cpp(package, root=tmp_path / "managed", client=offline)
    assert second.llama_cli == first.llama_cli


def test_an_archive_without_llama_server_is_refused(tmp_path: Path) -> None:
    package, client = _package(tmp_path, {"build/bin/llama-cli": b"c"})
    with pytest.raises(LlamaCppInstallError, match="llama-server"):
        install_llama_cpp(package, root=tmp_path / "managed", client=client)
    assert not (tmp_path / "managed" / package.directory_name).exists()
    assert [p.name for p in (tmp_path / "managed").iterdir()] == []  # no staging left behind
