"""Local filesystem for downloaded model artifacts.

Owns the choice of root directory (per-OS default under a user-visible path)
and the safe construction of per-artifact paths. Path traversal is rejected
up front: filenames must be a single basename, and the resolved path must
stay under the root.
"""

from __future__ import annotations

import re
from pathlib import Path

from jaull.artifacts.errors import ArtifactError, ArtifactVerificationError
from jaull.domain.artifacts import ModelArtifact
from jaull.paths import user_data_dir

_SHA256_SUFFIX = ".sha256"


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


__all__ = ["ArtifactStorage"]
