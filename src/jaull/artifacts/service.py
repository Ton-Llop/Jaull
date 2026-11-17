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
import logging
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

from jaull.artifacts import policies
from jaull.artifacts.errors import (
    ArtifactDownloadError,
    ArtifactNotFoundError,
    ArtifactVerificationError,
)
from jaull.artifacts.ports import ArtifactHistoryProtocol, ArtifactResolverProtocol
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact, OutdatedVersion
from jaull.exceptions import JaullError

DownloaderFn = Callable[..., str]
ProgressCallback = Callable[[int, int], None]

_HASH_BLOCK_BYTES = 1024 * 1024  # 1 MiB
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactService:
    resolver: ArtifactResolverProtocol
    storage: ArtifactStorage
    downloader: DownloaderFn = field(default=hf_hub_download)
    #: Tells an older published version of a file from bytes nobody published.
    history: ArtifactHistoryProtocol | None = None

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

    def is_downloaded(self, artifact: ModelArtifact) -> bool:
        return self.storage.exists(artifact)

    def download(
        self,
        artifact: ModelArtifact,
        progress_callback: ProgressCallback | None = None,
    ) -> ModelArtifact:
        target = self.storage.path_for(artifact)
        # Only a listener gets the progress class, so a downloader that knows nothing about it
        # (the tests' fakes, an older huggingface_hub) is called exactly as before.
        extra = (
            {"tqdm_class": _byte_progress_class(progress_callback)}
            if progress_callback is not None else {}
        )

        try:
            self.storage.ensure_parent(target)
            # A previous digest must not certify bytes replaced by a failed download.
            self.storage.clear_sha256(target)
            local_str = self.downloader(
                repo_id=artifact.repo_id,
                filename=artifact.filename,
                revision=artifact.revision or None,
                local_dir=str(target.parent),
                **extra,
            )
            local_path = Path(local_str).resolve()
            digest = _sha256_of(local_path)
            if artifact.sha256 is not None and digest != artifact.sha256.lower():
                raise ArtifactVerificationError(
                    f"SHA-256 mismatch for {artifact.filename}: "
                    f"expected {artifact.sha256}, downloaded {digest}."
                )
            self.storage.save_sha256(local_path, digest)
        except GatedRepoError as exc:
            raise ArtifactDownloadError(
                f"{artifact.repo_id} is gated. Set HF_TOKEN and request access."
            ) from exc
        except (EntryNotFoundError, RepositoryNotFoundError) as exc:
            raise ArtifactNotFoundError(
                f"{artifact.repo_id}/{artifact.filename} not found on Hugging Face."
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
        outdated = artifact.outdated
        if artifact.sha256 is not None and stored_hex != artifact.sha256.lower():
            outdated = self._older_version(artifact, stored_hex)

        if full:
            try:
                fresh = _sha256_of(path)
            except OSError as exc:
                raise ArtifactVerificationError(f"Could not read artifact {path}: {exc}") from exc
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
                "outdated": outdated,
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
            if artifact.sha256 is not None and stored != artifact.sha256.lower():
                update["outdated"] = self._older_version(artifact, stored)
            update["sha256"] = stored
        return artifact.model_copy(update=update)

    def _older_version(self, artifact: ModelArtifact, local_sha256: str) -> OutdatedVersion:
        """The local file differs from what is published now: say whether it was published.

        A digest that some recent commit of the repository still had is a genuine older
        version, so it may be used (and the person is told). Anything else, or a history
        that cannot be read, is refused as before.
        """
        assert artifact.sha256 is not None
        mismatch = (
            f"SHA-256 mismatch for {artifact.filename}: "
            f"expected {artifact.sha256}, sidecar {local_sha256}."
        )
        if self.history is None:
            raise ArtifactVerificationError(mismatch)
        limit = policies.ARTIFACT_HISTORY_COMMIT_LIMIT
        try:
            found = self.history.find_version(
                artifact.repo_id, artifact.filename, local_sha256, limit=limit,
            )
        except (HfHubHTTPError, OSError, JaullError) as exc:
            raise ArtifactVerificationError(
                f"{mismatch} The published versions could not be checked."
            ) from exc
        if found is None:
            raise ArtifactVerificationError(
                f"{mismatch} It matches none of the last {limit} published versions."
            )
        return OutdatedVersion(
            current_sha256=artifact.sha256, last_revision=found.revision,
            still_published_on=found.date,
        )


def _byte_progress_class(callback: ProgressCallback) -> type:
    """A `tqdm` stand-in for `hf_hub_download` that reports bytes instead of drawing a bar.

    The library draws its own bar on the terminal, which is useless (and garbles the screen)
    inside the TUI. It is built disabled, so nothing is drawn, and counts the bytes itself.
    """
    from tqdm.auto import tqdm  # type: ignore[import-untyped]

    class _ByteProgress(tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._counts_bytes = kwargs.get("unit") == "B"
            self._received = int(kwargs.get("initial") or 0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            if not self._counts_bytes or not n:
                return None
            self._received += int(n)
            try:
                callback(self._received, int(self.total or 0))
            except Exception:  # a listener must never break the download
                logger.debug("download progress listener failed", exc_info=True)
            return None

    return _ByteProgress


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
