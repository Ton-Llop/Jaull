from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from local_ai_check.domain.model import (
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from local_ai_check.huggingface.client import HfClientProtocol


@dataclass
class AnalyzerResult:
    config: ModelConfig | None = None
    relevant_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RepositoryAnalyzer(Protocol):
    """Contract for per-type analyzers.

    An analyzer receives everything already known about a repository plus the HF client
    (in case it needs to fetch small metadata files) and returns extra insights that get
    merged into the final :class:`ModelAnalysis`.
    """

    def analyze(
        self,
        repo: ModelRepositoryInfo,
        files: list[ModelFile],
        classification: RepositoryClassification,
        client: HfClientProtocol,
    ) -> AnalyzerResult: ...
