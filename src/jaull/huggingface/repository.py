"""Orchestrates fetching metadata, classifying and running the matching analyzer."""

from __future__ import annotations

from datetime import UTC, datetime

from huggingface_hub.hf_api import ModelInfo

from jaull.analyzers.registry import get_analyzer
from jaull.domain.model import (
    ModelAnalysis,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.huggingface.classifiers import classify_repository
from jaull.huggingface.client import HfClient, HfClientProtocol


def inspect_model(repo_id: str, client: HfClientProtocol | None = None) -> ModelAnalysis:
    hf_client = client if client is not None else HfClient()
    info = hf_client.model_info(repo_id)

    repo = _build_repository_info(info, repo_id=repo_id)
    files = _build_files(info)
    classification = classify_repository(files)

    analyzer = get_analyzer(classification.primary_type)
    result = analyzer.analyze(repo, files, classification, hf_client)

    total_size = _compute_total_size(files, classification)

    return ModelAnalysis(
        repo=repo,
        files=files,
        classification=classification,
        config=result.config,
        relevant_files=result.relevant_files,
        total_size_bytes=total_size,
        warnings=result.warnings,
    )


def _build_repository_info(info: ModelInfo, repo_id: str) -> ModelRepositoryInfo:
    tags = list(getattr(info, "tags", None) or [])
    card_data = getattr(info, "card_data", None) or {}
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()

    license_value = None
    if isinstance(card_data, dict):
        license_value = card_data.get("license")
        if isinstance(license_value, list):
            license_value = license_value[0] if license_value else None

    gated = bool(getattr(info, "gated", False))
    private = bool(getattr(info, "private", False))

    last_modified = getattr(info, "last_modified", None)
    if isinstance(last_modified, str):
        try:
            last_modified = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        except ValueError:
            last_modified = None
    if isinstance(last_modified, datetime) and last_modified.tzinfo is None:
        last_modified = last_modified.replace(tzinfo=UTC)

    return ModelRepositoryInfo(
        repo_id=repo_id,
        author=getattr(info, "author", None),
        private=private,
        gated=gated,
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        last_modified=last_modified,
        license=str(license_value) if license_value else None,
        tags=tags,
        pipeline_tag=getattr(info, "pipeline_tag", None),
        library_name=getattr(info, "library_name", None),
    )


def _build_files(info: ModelInfo) -> list[ModelFile]:
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


def _compute_total_size(
    files: list[ModelFile], classification: RepositoryClassification
) -> int | None:
    # This is the "download the whole repository" figure: sum of every unique file.
    # For GGUF repos it therefore includes every quantization variant. The per-variant
    # size that the estimator consumes lives separately in
    # RepositoryClassification.gguf_variants[*].total_bytes.
    del classification
    total = sum(f.size_bytes for f in files if f.size_bytes)
    return total or None
