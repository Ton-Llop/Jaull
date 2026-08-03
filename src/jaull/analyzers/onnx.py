from __future__ import annotations

from jaull.analyzers.base import AnalyzerResult
from jaull.analyzers.generic import collect_relevant_files
from jaull.domain.model import (
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.huggingface.client import HfClientProtocol


class OnnxAnalyzer:
    """Minimal analyzer for ONNX repositories.

    Reading ONNX metadata (opsets, IR version) requires downloading the graph,
    which is out of scope for the first iteration.
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
                "ONNX analyzer is minimal in this iteration; "
                "opsets and IR version are not inspected."
            ],
        )
