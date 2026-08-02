"""In-memory memoisation for a single guided run.

The same repository frequently arrives from several queries, and a candidate's
config.json is read once per quantization rung on the ladder. Caching those
within the run avoids repeating network calls; the cache dies with the run, so
there is no database, no invalidation policy and no stale-data class of bug.
"""

from __future__ import annotations

from collections.abc import Callable


class RunCache[K, V]:
    """A dict with a ``get_or_compute`` and hit/miss counters for tests."""

    def __init__(self) -> None:
        self._entries: dict[K, V] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: K, factory: Callable[[], V]) -> V:
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        value = factory()
        self._entries[key] = value
        return value

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


__all__ = ["RunCache"]
