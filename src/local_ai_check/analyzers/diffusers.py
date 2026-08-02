from __future__ import annotations

from local_ai_check.analyzers.base import AnalyzerResult
from local_ai_check.analyzers.generic import collect_relevant_files
from local_ai_check.domain.model import (
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from local_ai_check.huggingface.client import HfClientProtocol


class DiffusersAnalyzer:
    """Minimal analyzer for Diffusers pipelines.

    Deeper parsing of ``model_index.json`` and per-subfolder configs is deferred
    to a future iteration.
    """

    def analyze(
        self,
        repo: ModelRepositoryInfo,
        files: list[ModelFile],
        classification: RepositoryClassification,
        client: HfClientProtocol,
    ) -> AnalyzerResult:
        del repo, classification, client
        return AnalyzerResult(
            config=None,
            relevant_files=collect_relevant_files(files),
            warnings=[
                "Diffusers analyzer is minimal in this iteration; "
                "sub-component configs are not parsed."
            ],
        )
