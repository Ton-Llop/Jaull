"""Provenance-tracked parameter counts.

The composite recommendation logic needs to know *how* a parameter count was
obtained. A 7B derived from ``safetensors`` metadata is a measurement; a 7B
derived from the repo id is a guess. Treating them the same is what leads to
reports that quietly under-report a Qwen 2.5 7B as "4.3B".
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from jaull.domain.estimation import EstimationConfidence


class ParameterCountSource(StrEnum):
    SAFETENSORS_METADATA = "safetensors_metadata"
    MODEL_CONFIG = "model_config"
    GGUF_METADATA = "gguf_metadata"
    NAME_INFERENCE = "name_inference"
    UNKNOWN = "unknown"


class ParameterCount(BaseModel):
    """A count with the evidence that produced it."""

    model_config = ConfigDict(frozen=True)

    count: int | None
    source: ParameterCountSource
    confidence: EstimationConfidence

    @property
    def is_known(self) -> bool:
        return self.count is not None and self.count > 0


__all__ = ["ParameterCount", "ParameterCountSource"]
