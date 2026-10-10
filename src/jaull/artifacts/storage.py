"""Local filesystem for downloaded model artifacts.

Owns the choice of root directory (per-OS default under a user-visible path)
and the safe construction of per-artifact paths. Path traversal is rejected
up front: filenames must be a single basename, and the resolved path must
stay under the root.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from jaull.artifacts.errors import ArtifactError, ArtifactVerificationError
from jaull.domain.artifacts import ModelArtifact
from jaull.paths import user_data_dir

_SHA256_SUFFIX = ".sha256"


@dataclass(frozen=True)
class LocalModelFile:
    repo_id: str
    filename: str
    path: Path
    size_bytes: int
    #: The verified digest from the sidecar, or None if it was never verified.
    sha256: str | None


def _default_models_dir() -> Path:
    return user_data_dir("models")


class ArtifactStorage:
    """Filesystem layout for ``<root>/<owner>/<repo>/<filename>``."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _default_models_dir()).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, artifact: ModelArtifact) -> Path:
        return self._safe_path(artifact.repo_id, artifact.filename)

    def exists(self, artifact: ModelArtifact) -> bool:
        return self.path_for(artifact).is_file()

    def save_sha256(self, path: Path, hex_digest: str) -> None:
        sidecar = self._sidecar(path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(hex_digest.strip() + "\n", encoding="ascii")

    def clear_sha256(self, path: Path) -> None:
        self._sidecar(path).unlink(missing_ok=True)

    def load_sha256(self, path: Path) -> str | None:
        sidecar = self._sidecar(path)
        if not sidecar.is_file():
            return None
        try:
            digest = sidecar.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ArtifactVerificationError(f"Could not read SHA-256 sidecar {sidecar}.") from exc
        if not digest:
            return None
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ArtifactVerificationError(f"Invalid SHA-256 sidecar {sidecar}.")
        return digest.lower()

    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def local_files(self) -> list[LocalModelFile]:
        """Every downloaded file under ``<owner>/<repo>/``, largest first."""
        if not self._root.is_dir():
            return []
        files = []
        for path in self._root.glob("*/*/*"):
            if not path.is_file() or path.name.endswith(_SHA256_SUFFIX):
                continue
            owner, repo = path.parent.parent.name, path.parent.name
            try:
                digest = self.load_sha256(path)
            except ArtifactVerificationError:
                digest = None
            files.append(LocalModelFile(
                f"{owner}/{repo}", path.name, path, path.stat().st_size, digest,
            ))
        return sorted(files, key=lambda item: (-item.size_bytes, item.repo_id, item.filename))

    def delete(self, repo_id: str, filename: str) -> int:
        """Remove one downloaded file with its sidecar and download metadata.

        Returns the bytes freed. The path goes through the same traversal checks
        as a download, so nothing outside the models root can be removed.
        """
        path = self._safe_path(repo_id, filename)
        freed = path.stat().st_size if path.is_file() else 0
        path.unlink(missing_ok=True)
        self._sidecar(path).unlink(missing_ok=True)
        download = path.parent / ".cache" / "huggingface" / "download"
        for leftover in (download / f"{filename}.metadata", download / f"{filename}.lock"):
            leftover.unlink(missing_ok=True)
        # A repository folder with no model left holds only download bookkeeping.
        if not any(item.is_file() and not item.name.endswith(_SHA256_SUFFIX)
                   for item in path.parent.iterdir()):
            shutil.rmtree(path.parent, ignore_errors=True)
            with contextlib.suppress(OSError):
                path.parent.parent.rmdir()  # Only if the owner folder is now empty.
        return freed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _safe_path(self, repo_id: str, filename: str) -> Path:
        owner, _, repo = repo_id.partition("/")
        if not owner or not repo or "/" in repo:
            raise ArtifactError(
                f"Invalid repo_id {repo_id!r}: expected 'owner/repo'."
            )
        self._reject_traversal(owner)
        self._reject_traversal(repo)
        self._reject_filename(filename)

        candidate = (self._root / owner / repo / filename).resolve()
        root_resolved = self._root.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ArtifactError(
                f"Refusing path outside storage root: {candidate}."
            ) from exc
        return candidate

    @staticmethod
    def _reject_traversal(segment: str) -> None:
        if segment in {"", ".", ".."} or "\\" in segment or "/" in segment:
            raise ArtifactError(f"Invalid path segment: {segment!r}.")
        if "\x00" in segment:
            raise ArtifactError("Path segment contains a NUL byte.")

    @staticmethod
    def _reject_filename(filename: str) -> None:
        if not filename or filename in {".", ".."}:
            raise ArtifactError(f"Invalid filename: {filename!r}.")
        if "/" in filename or "\\" in filename:
            raise ArtifactError(
                f"Filename must be a single basename, got {filename!r}."
            )
        if "\x00" in filename:
            raise ArtifactError("Filename contains a NUL byte.")
        if Path(filename).is_absolute():
            raise ArtifactError(f"Filename must be relative, got {filename!r}.")

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_name(path.name + _SHA256_SUFFIX)


__all__ = ["ArtifactStorage", "LocalModelFile"]
