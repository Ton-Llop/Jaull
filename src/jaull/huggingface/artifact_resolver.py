"""Hugging Face implementation of :class:`ArtifactResolverProtocol`.

Uses ``HfClientProtocol.model_info`` to get the sibling list + commit sha,
then reuses the shared classifier and variant selector so the quantization
matching stays consistent with what the estimator already does.
"""

from __future__ import annotations

from jaull.artifacts.errors import (
    ArtifactFormatNotSupportedError,
    ArtifactNotFoundError,
)
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.enums import RepositoryType
from jaull.domain.model import ModelFile
from jaull.estimator.gguf_selection import select_variant
from jaull.huggingface.classifiers import classify_repository
from jaull.huggingface.client import HfClientProtocol

_GGUF_FORMAT = "gguf"


class HuggingFaceArtifactResolver:
    """Resolve a concrete GGUF file for a given repo/quantization."""

    def __init__(self, hf_client: HfClientProtocol) -> None:
        self._client = hf_client

    def resolve(
        self,
        repo_id: str,
        *,
        quantization: str | None,
        revision: str | None = None,
    ) -> ModelArtifact:
        info = self._client.model_info(repo_id)
        files = self._siblings_to_files(info)
        if not files:
            raise ArtifactNotFoundError(
                f"Repository {repo_id!r} has no listed files."
            )

        classification = classify_repository(files)
        if RepositoryType.GGUF not in classification.detected_types:
            raise ArtifactFormatNotSupportedError(
                f"Repository {repo_id!r} is not a GGUF repository. "
                "Only GGUF is supported in this phase."
            )

        choice = select_variant(classification.gguf_variants, quantization)
        variant = choice.variant
        if len(variant.files) != 1:
            raise ArtifactFormatNotSupportedError(
                f"GGUF variant {variant.quantization!r} in {repo_id!r} is "
                f"multipart ({len(variant.files)} files). Multipart GGUF is "
                "not supported in this phase."
            )

        file = variant.files[0]
        resolved_revision = revision or self._extract_sha(info) or "main"

        return ModelArtifact(
            repo_id=repo_id,
            revision=resolved_revision,
            filename=file.path,
            format=_GGUF_FORMAT,
            quantization=variant.quantization,
            size_bytes=file.size_bytes,
        )

    @staticmethod
    def _siblings_to_files(info: object) -> list[ModelFile]:
        siblings = getattr(info, "siblings", None) or []
        files: list[ModelFile] = []
        for sibling in siblings:
            path = getattr(sibling, "rfilename", None)
            if not path:
                continue
            size = getattr(sibling, "size", None)
            lfs = getattr(sibling, "lfs", None) is not None
            files.append(ModelFile(path=path, size_bytes=size, lfs=lfs))
        return files

    @staticmethod
    def _extract_sha(info: object) -> str | None:
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha:
            return sha
        return None


__all__ = ["HuggingFaceArtifactResolver"]
