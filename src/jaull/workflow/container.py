"""A deliberately small service container.

Not a dependency-injection framework — just a frozen dataclass of callables that
the orchestrator and the guided screens receive instead of constructing
``HfClient()`` themselves. That is enough to make every new screen testable with
fakes, without adding a library or a lifecycle to reason about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jaull.discovery.search_client import ModelSearchClient
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.huggingface.client import HfClientProtocol
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

    @classmethod
    def default(cls) -> ServiceContainer:
        """Build the production container.

        Imports are local so that constructing a test container never drags in
        ``huggingface_hub`` or opens a network-capable client.
        """
        from jaull.discovery.search_client import HfSearchClient
        from jaull.estimator.service import estimate_memory
        from jaull.hardware.detector import detect_hardware
        from jaull.huggingface.client import HfClient
        from jaull.huggingface.repository import inspect_model
        from jaull.metadata.range_reader import HttpxRangeClient

        return cls(
            hf_client=HfClient(),
            search_client=HfSearchClient(),
            detect_hardware=detect_hardware,
            inspect_model=inspect_model,
            estimate_memory=estimate_memory,
            capability_analyzer=MetadataCapabilityAnalyzer(),
            range_client_factory=HttpxRangeClient,
        )


__all__ = ["ServiceContainer"]
