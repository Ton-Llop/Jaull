"""Compatibility shim for the production service container."""

from __future__ import annotations

from jaull.bootstrap.container import (
    DetectHardwareFn,
    EstimateMemoryFn,
    InspectModelFn,
    ServiceContainer,
)

__all__ = [
    "DetectHardwareFn",
    "EstimateMemoryFn",
    "InspectModelFn",
    "ServiceContainer",
]
