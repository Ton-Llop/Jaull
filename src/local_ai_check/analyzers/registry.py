from __future__ import annotations

from local_ai_check.analyzers.base import RepositoryAnalyzer
from local_ai_check.analyzers.diffusers import DiffusersAnalyzer
from local_ai_check.analyzers.generic import GenericAnalyzer
from local_ai_check.analyzers.gguf import GgufAnalyzer
from local_ai_check.analyzers.onnx import OnnxAnalyzer
from local_ai_check.analyzers.transformers import TransformersAnalyzer
from local_ai_check.domain.enums import RepositoryType

_ANALYZERS: dict[RepositoryType, RepositoryAnalyzer] = {
    RepositoryType.TRANSFORMERS: TransformersAnalyzer(),
    RepositoryType.GGUF: GgufAnalyzer(),
    RepositoryType.DIFFUSERS: DiffusersAnalyzer(),
    RepositoryType.ONNX: OnnxAnalyzer(),
    RepositoryType.ADAPTER: GenericAnalyzer(),
    RepositoryType.UNKNOWN: GenericAnalyzer(),
}


def get_analyzer(repo_type: RepositoryType) -> RepositoryAnalyzer:
    return _ANALYZERS.get(repo_type, GenericAnalyzer())
