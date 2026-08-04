"""Application-layer service that turns abstract recommendations into files.

Composes:
- an ``ArtifactResolverProtocol`` (typically Hugging Face) that maps
  ``repo_id + quantization`` to a concrete ``ModelArtifact``;
- an ``ArtifactStorage`` that owns the on-disk layout and SHA-256 sidecars;
- a downloader callable (defaults to ``huggingface_hub.hf_hub_download``,
  injected so tests never touch the network).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from jaull.artifacts.errors import (
    ArtifactDownloadError,
    ArtifactNotFoundError,
    ArtifactVerificationError,
)
from jaull.artifacts.ports import ArtifactResolverProtocol
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.exceptions import JaullError

DownloaderFn = Callable[..., str]
ProgressCallback = Callable[[int, int], None]

_HASH_BLOCK_BYTES = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class ArtifactService:
    resolver: ArtifactResolverProtocol
    storage: ArtifactStorage
    downloader: DownloaderFn = field(default=hf_hub_download)

    def resolve(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        artifact = self.resolver.resolve(
            repo_id, quantization=quantization, revision=revision
        )
        return self._promote_if_present(artifact)

    def download(
        self,
        artifact: ModelArtifact,
        progress_callback: ProgressCallback | None = None,
    ) -> ModelArtifact:
        # ``progress_callback`` is accepted for forward compatibility but is
        # not wired to ``hf_hub_download`` yet — that helper renders its own
        # tqdm bar. The next phase (Runner/UI) will replace this with a
        # streaming download that reports byte-level progress.
        del progress_callback

        target = self.storage.path_for(artifact)
        self.storage.ensure_parent(target)

        try:
            local_str = self.downloader(
                repo_id=artifact.repo_id,
                filename=artifact.filename,
                revision=artifact.revision or None,
                local_dir=str(target.parent),
            )
        except (EntryNotFoundError, RepositoryNotFoundError) as exc:
            raise ArtifactNotFoundError(
                f"{artifact.repo_id}/{artifact.filename} not found on Hugging Face."
            ) from exc
        except GatedRepoError as exc:
            raise ArtifactDownloadError(
                f"{artifact.repo_id} is gated. Set HF_TOKEN and request access."
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise ArtifactDownloadError(
                f"HTTP {status} while downloading {artifact.filename!r} from "
                f"{artifact.repo_id}."
            ) from exc
        except JaullError:
            raise
        except OSError as exc:
            raise ArtifactDownloadError(
                f"I/O error while downloading {artifact.filename!r}: {exc}"
            ) from exc

        local_path = Path(local_str).resolve()
        digest = _sha256_of(local_path)
        self.storage.save_sha256(local_path, digest)

        return artifact.model_copy(
            update={
                "local_path": local_path,
                "sha256": digest,
                "is_downloaded": True,
                "is_verified": False,
            }
        )

    def verify(
        self,
        artifact: ModelArtifact,
        *,
        full: bool = False,
    ) -> ModelArtifact:
        path = artifact.local_path or self.storage.path_for(artifact)
        if not path.is_file():
            raise ArtifactVerificationError(
                f"Artifact file missing: {path}."
            )

        actual_size = path.stat().st_size
        if artifact.size_bytes is not None and actual_size != artifact.size_bytes:
            raise ArtifactVerificationError(
                f"Size mismatch for {artifact.filename}: "
                f"expected {artifact.size_bytes}, got {actual_size}."
            )

        stored_hex = self.storage.load_sha256(path)
        if stored_hex is None:
            raise ArtifactVerificationError(
                f"Missing SHA-256 sidecar for {artifact.filename}."
            )

        if full:
            fresh = _sha256_of(path)
            if fresh != stored_hex:
                raise ArtifactVerificationError(
                    f"SHA-256 mismatch for {artifact.filename}: "
                    f"sidecar {stored_hex}, disk {fresh}."
                )

        return artifact.model_copy(
            update={
                "local_path": path,
                "sha256": stored_hex,
                "is_downloaded": True,
                "is_verified": True,
            }
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _promote_if_present(self, artifact: ModelArtifact) -> ModelArtifact:
        path = self.storage.path_for(artifact)
        if not path.is_file():
            return artifact
        update: dict[str, Any] = {"local_path": path, "is_downloaded": True}
        stored = self.storage.load_sha256(path)
        if stored is not None:
            update["sha256"] = stored
        return artifact.model_copy(update=update)


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_HASH_BLOCK_BYTES)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


__all__ = ["ArtifactService", "DownloaderFn", "ProgressCallback"]
