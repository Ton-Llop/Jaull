"""Lightweight timings and counters for application workflows."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class PerformanceTelemetry:
    durations: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_duration(name, time.perf_counter() - started)

    def add_duration(self, name: str, seconds: float) -> None:
        with self._lock:
            self.durations[name] = self.durations.get(name, 0.0) + seconds

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            durations = dict(self.durations)
            counters = dict(self.counters)
        payload: dict[str, float | int] = {}
        for key, value in durations.items():
            payload[f"duration.{key}"] = value
        for key, value in counters.items():
            payload[f"count.{key}"] = value
        payload["duration.total"] = sum(durations.values())
        return payload


__all__ = ["PerformanceTelemetry"]
