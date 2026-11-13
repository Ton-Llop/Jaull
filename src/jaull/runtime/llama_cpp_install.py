"""Install a pinned official llama.cpp build into Jaull's data directory.

Nothing is unpacked before its SHA-256 matches the pinned catalog, and nothing is
unpacked outside the target directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import httpx

from jaull.domain.runtime import (
    LlamaCppInstallation,
    RuntimeDiscoveryMetadata,
    RuntimeResolutionStatus,
    RuntimeSource,
)
from jaull.paths import user_data_dir
from jaull.runtime import policies
from jaull.runtime.llama_cpp_release import LlamaCppPackage, ReleaseAsset

_CHUNK_BYTES = 1024 * 1024
_REQUIRED_BINARIES = ("llama-cli", "llama-server")
_EXECUTABLE_BITS = 0o111


class LlamaCppInstallError(RuntimeError):
    """The pinned llama.cpp build could not be downloaded, verified or unpacked."""


def download_verified(client: httpx.Client, asset: ReleaseAsset, directory: Path) -> Path:
    """Download ``asset`` into ``directory``; keep it only if its SHA-256 matches."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / asset.name
    digest = hashlib.sha256()
    try:
        with client.stream("GET", asset.url, follow_redirects=True) as response:
            if response.status_code != httpx.codes.OK:
                raise LlamaCppInstallError(
                    f"Download of {asset.name} failed: HTTP {response.status_code}."
                )
            with target.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    digest.update(chunk)
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        target.unlink(missing_ok=True)
        raise LlamaCppInstallError(f"Download of {asset.name} failed: {exc}") from exc
    except LlamaCppInstallError:
        target.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != asset.sha256:
        target.unlink(missing_ok=True)
        raise LlamaCppInstallError(
            f"SHA-256 of {asset.name} does not match the pinned value; the file was deleted."
        )
    return target


def extract_archive(archive: Path, target: Path) -> None:
    """Unpack a .zip or .tar.gz into ``target``, refusing any member that escapes it."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                _require_inside(root, member)
            bundle.extractall(root)
        return
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as bundle:
            for info in bundle.getmembers():
                _require_inside(root, info.name)
            # The "data" filter also rejects links and special files that point outside.
            bundle.extractall(root, filter="data")
        return
    raise LlamaCppInstallError(f"Unsupported archive type: {archive.name}.")


def _require_inside(root: Path, member: str) -> None:
    if not (root / member).resolve().is_relative_to(root):
        raise LlamaCppInstallError(f"Archive member {member!r} would unpack outside {root}.")


def managed_root() -> Path:
    """Where Jaull keeps the llama.cpp builds it installs."""
    return user_data_dir("runtimes", policies.LLAMA_CPP_MANAGED_DIRNAME)


def install_llama_cpp(
    package: LlamaCppPackage, *, root: Path, client: httpx.Client,
) -> LlamaCppInstallation:
    """Install ``package`` under ``root``; reuse it if it is already complete there.

    Everything is downloaded and unpacked in a staging directory first and only
    moved into place once the required binaries are present, so a failed or
    interrupted install never leaves a half-populated build behind.
    """
    target = root / package.directory_name
    if target.is_dir():
        existing = installation_in(target, package.build)
        if existing is not None:
            return existing
        shutil.rmtree(target)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=root))
    try:
        unpacked = staging / "unpacked"
        for asset in package.assets:
            archive = download_verified(client, asset, staging / "downloads")
            extract_archive(archive, unpacked)
            archive.unlink()
        missing = [name for name in _REQUIRED_BINARIES if _find_binary(unpacked, name) is None]
        if missing:
            raise LlamaCppInstallError(
                f"{package.archive.name} does not contain {', '.join(missing)}."
            )
        unpacked.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    installation = installation_in(target, package.build)
    if installation is None:
        raise LlamaCppInstallError(f"Installed build at {target} could not be read back.")
    return installation


def installation_in(directory: Path, build: str) -> LlamaCppInstallation | None:
    """The managed installation unpacked in ``directory``, if it is complete."""
    cli = _find_binary(directory, "llama-cli")
    server = _find_binary(directory, "llama-server")
    if cli is None or server is None:
        return None
    bench = _find_binary(directory, "llama-bench")
    if os.name != "nt":
        for binary in (cli, server, bench):
            if binary is not None:
                binary.chmod(binary.stat().st_mode | _EXECUTABLE_BITS)
    return LlamaCppInstallation(
        root=str(cli.parent.resolve()),
        llama_cli=str(cli.resolve()),
        llama_bench=str(bench.resolve()) if bench is not None else None,
        source=RuntimeSource.MANAGED,
        status=RuntimeResolutionStatus.FOUND,
        discovery=RuntimeDiscoveryMetadata(detail=f"installed by Jaull in {directory.name}"),
        version_text=build,
    )


def _find_binary(directory: Path, name: str) -> Path | None:
    names = (f"{name}.exe", name) if os.name == "nt" else (name,)
    for candidate_name in names:
        for candidate in sorted(directory.rglob(candidate_name)):
            if candidate.is_file():
                return candidate
    return None


__all__ = [
    "LlamaCppInstallError",
    "download_verified",
    "extract_archive",
    "install_llama_cpp",
    "installation_in",
    "managed_root",
]
