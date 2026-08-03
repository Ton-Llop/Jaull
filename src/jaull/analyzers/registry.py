from __future__ import annotations

from jaull.analyzers.base import RepositoryAnalyzer
from jaull.analyzers.diffusers import DiffusersAnalyzer
from jaull.analyzers.generic import GenericAnalyzer
from jaull.analyzers.gguf import GgufAnalyzer
from jaull.analyzers.onnx import OnnxAnalyzer
from jaull.analyzers.transformers import TransformersAnalyzer
from jaull.domain.enums import RepositoryType

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
