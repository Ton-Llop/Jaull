"""Production composition root for guided recommendation workflows."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from jaull.discovery.search_client import ModelSearchClient
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.huggingface.client import HfClientProtocol
from jaull.ports.cache import ModelAnalysisCacheProtocol
from jaull.recommendation.capability import (
    CapabilityAnalyzer,
    MetadataCapabilityAnalyzer,
)

DetectHardwareFn = Callable[..., HardwareProfile]
InspectModelFn = Callable[..., ModelAnalysis]
EstimateMemoryFn = Callable[..., MemoryEstimate]


@dataclass(frozen=True)
class ServiceContainer:
    """Everything the guided workflow needs from the outside world."""

    hf_client: HfClientProtocol
    search_client: ModelSearchClient
    detect_hardware: DetectHardwareFn
    inspect_model: InspectModelFn
    estimate_memory: EstimateMemoryFn
    capability_analyzer: CapabilityAnalyzer = field(
        default_factory=MetadataCapabilityAnalyzer
    )
    range_client_factory: Callable[[], object] | None = field(default=None)
    model_analysis_cache: ModelAnalysisCacheProtocol | None = field(default=None)

    @classmethod
    def default(cls) -> ServiceContainer:
        """Build the production container."""
        from jaull.adapters.cache.model_analysis_cache import ModelAnalysisCache
        from jaull.discovery.search_client import HfSearchClient
        from jaull.estimator.service import estimate_memory
        from jaull.hardware.detector import detect_hardware
        from jaull.huggingface.client import HfClient
        from jaull.huggingface.repository import inspect_model
        from jaull.metadata.range_reader import HttpxRangeClient

        _load_dotenv()

        return cls(
            hf_client=HfClient(),
            search_client=HfSearchClient(),
            detect_hardware=detect_hardware,
            inspect_model=inspect_model,
            estimate_memory=estimate_memory,
            capability_analyzer=MetadataCapabilityAnalyzer(),
            range_client_factory=HttpxRangeClient,
            model_analysis_cache=ModelAnalysisCache(),
        )


def _load_dotenv() -> None:
    dotenv = Path.cwd() / ".env"
    if not dotenv.is_file():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _parse_dotenv_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line.removeprefix("export ").strip()
    key, value = line.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    return key, _strip_dotenv_value(value.strip())


def _strip_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


__all__ = [
    "DetectHardwareFn",
    "EstimateMemoryFn",
    "InspectModelFn",
    "ServiceContainer",
]
