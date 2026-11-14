"""Speculative-decoding drafters: GGUFs that only run next to a separate target model.

One definition, used by repository classification (file and folder names),
metadata enrichment (the GGUF header), candidate filtering (Hub tags) and
recommendation (the resolved architecture). None of these is a standalone text
model, so Jaull never recommends or serves one as the main model.
"""

from __future__ import annotations

# ``general.architecture`` / ``model_type`` values of drafter models.
DRAFTER_ARCHITECTURES = frozenset({"eagle3", "dflash"})
# File and folder name tokens that mark a drafter beside the main weights.
DRAFTER_NAME_TOKENS = frozenset({"draft", "eagle3"})
# Hub tags a repository uses to declare that it is a drafter.
DRAFTER_HUB_TAGS = frozenset({"draft-model"})


def is_drafter_architecture(architecture: str | None) -> bool:
    return (architecture or "").casefold() in DRAFTER_ARCHITECTURES


__all__ = [
    "DRAFTER_ARCHITECTURES",
    "DRAFTER_HUB_TAGS",
    "DRAFTER_NAME_TOKENS",
    "is_drafter_architecture",
]
