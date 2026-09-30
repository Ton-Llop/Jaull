"""Configuration for a single inference-memory estimation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TargetDevice(StrEnum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


class WeightPrecision(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"


class InferenceConfiguration(BaseModel):
    """User-controllable knobs for a memory estimate."""

    model_config = ConfigDict(frozen=True)

    context_length: int = Field(gt=0)
    # Independent sequences per batch, each retaining its own context/KV cache.
    # Not llama.cpp's --batch-size (tokens submitted per llama_decode call).
    batch_size: int = Field(default=1, gt=0)
    target_device: TargetDevice = TargetDevice.AUTO
    precision: WeightPrecision | None = None
    quantization: str | None = None
    kv_cache_dtype: WeightPrecision = WeightPrecision.FLOAT16
    safety_margin_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    device_reserve_bytes: int = Field(default=0, ge=0)
    # Number of independent inference sessions the estimate should size for. Each
    # session keeps its own KV cache, so N users multiply the KV footprint by N —
    # this is an orthogonal axis to ``batch_size`` (sequences per batch).
    concurrent_users: int = Field(default=1, ge=1)
