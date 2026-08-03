from __future__ import annotations

from jaull.analyzers.base import AnalyzerResult
from jaull.analyzers.generic import collect_relevant_files
from jaull.domain.model import (
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.huggingface.client import HfClientProtocol


class GgufAnalyzer:
    """GGUF repositories often contain alternative quantizations that MUST NOT be summed.

    Classification already groups files into variants; this analyzer only adds a
    warning enumerating them so the user knows what is on offer.

    "GGUF: Generalized Graphical Universal Format".
    Un GGUF és un format d'arxiu binari disenyat per emmagatzemar models d'aprenentatge automàtic,
    especialment models de llenguatge gran (LLM). GGUF significa.
    """

    def analyze(
        self,
        repo: ModelRepositoryInfo,
        files: list[ModelFile],
        classification: RepositoryClassification,
        client: HfClientProtocol,
    ) -> AnalyzerResult:
        del repo, client
        warnings: list[str] = []

        if len(classification.gguf_variants) > 1:
            quants = ", ".join(v.quantization for v in classification.gguf_variants)
            warnings.append(
                f"Repository contains {len(classification.gguf_variants)} GGUF "
                f"quantization variants ({quants}); only one is loaded at a time."
            )

        return AnalyzerResult(
            config=None,
            relevant_files=collect_relevant_files(files),
            warnings=warnings,
        )
