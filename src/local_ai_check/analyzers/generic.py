from __future__ import annotations

import json
import logging
from pathlib import PurePosixPath

from local_ai_check.analyzers.base import AnalyzerResult
from local_ai_check.domain.enums import RepositoryType
from local_ai_check.domain.model import (
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from local_ai_check.exceptions import ConfigurationNotFoundError, LocalAiCheckError
from local_ai_check.huggingface.client import HfClientProtocol

logger = logging.getLogger(__name__)

_ALWAYS_RELEVANT = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model_index.json",
    "adapter_config.json",
    "README.md",
}


class GenericAnalyzer:
    """Fallback analyzer used for adapter/unknown repositories.

    Lists commonly relevant files and, when :class:`RepositoryType.ADAPTER` is present,
    attempts to fetch ``adapter_config.json``.
    """

    def analyze(
        self,
        repo: ModelRepositoryInfo,
        files: list[ModelFile],
        classification: RepositoryClassification,
        client: HfClientProtocol,
    ) -> AnalyzerResult:
        relevant = collect_relevant_files(files)
        warnings: list[str] = []

        if RepositoryType.ADAPTER in classification.detected_types:
            try:
                path = client.download_small_file(repo.repo_id, "adapter_config.json")
                data = json.loads(path.read_text(encoding="utf-8"))
                base = data.get("base_model_name_or_path")
                if base:
                    warnings.append(f"Adapter targets base model: {base}")
            except ConfigurationNotFoundError:
                warnings.append("adapter_config.json declared but could not be downloaded.")
            except LocalAiCheckError as exc:
                warnings.append(f"Could not read adapter_config.json: {exc}")
            except (OSError, ValueError) as exc:
                logger.debug("adapter_config parse failed", exc_info=exc)
                warnings.append("adapter_config.json could not be parsed.")

        return AnalyzerResult(config=None, relevant_files=relevant, warnings=warnings)


def collect_relevant_files(files: list[ModelFile]) -> list[str]:
    names_present = [f.path for f in files]
    top_level = {PurePosixPath(f.path).name for f in files}
    relevant = sorted(
        p for p in names_present if PurePosixPath(p).name in _ALWAYS_RELEVANT and _is_top_level(p)
    )
    # Include common weight and index files without duplicating.
    weights_and_indices = sorted(
        p
        for p in names_present
        if p not in relevant
        and any(
            PurePosixPath(p).name.lower().endswith(suffix)
            for suffix in (
                ".safetensors",
                ".safetensors.index.json",
                ".bin.index.json",
                ".gguf",
                ".onnx",
            )
        )
    )
    return relevant + weights_and_indices if top_level else []


def _is_top_level(path: str) -> bool:
    return "/" not in path
