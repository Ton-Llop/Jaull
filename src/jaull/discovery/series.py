"""Group candidates by *series* (same family + different parameter sizes).

`grouping.collapse_families` already fuses the same repo across formats
(safetensors vs GGUF conversions). `series` is complementary: it groups
``Qwen2.5-0.5B``, ``Qwen2.5-1.5B``, ``Qwen2.5-3B`` and ``Qwen2.5-7B`` together
so the recommender can pick the size that best matches the user priority and
show the others as within-family alternatives.

The signal is intentionally weak — series-key + parameter count. If the repo
name does not carry those, the candidate is its own series (never merged with
a lookalike).
"""

from __future__ import annotations

import re

from jaull.domain import families
from jaull.domain.candidates import EvaluatedCandidate

# Anything after the family token that looks like "-<digits>[.<digits>]B"
# is the size marker; strip it to derive the series key.
_SIZE_RE = re.compile(r"[-_](\d+(?:\.\d+)?)[bB](?![a-zA-Z])")


def series_key(evaluated: EvaluatedCandidate) -> str:
    """Return an identifier that groups siblings of the same family together.

    ``meta-llama/Meta-Llama-3.1-8B-Instruct`` → ``meta-llama/llama-3.1:instruct``.
    Two candidates in the same series share this key; two candidates from
    unrelated repos never do.
    """
    repo = evaluated.candidate.repo_id
    owner = repo.split("/", 1)[0].lower() if "/" in repo else ""
    family = families.detect_family(evaluated.candidate)
    if family == "unknown":
        # No merging when we cannot recognise the family. The series is the
        # repo itself so it never accidentally collides with anything else.
        return f"solo:{repo.lower()}"
    variant = _variant_tag(repo)
    variant_suffix = f":{variant}" if variant else ""
    return f"{owner}/{family}{variant_suffix}"


def _variant_tag(repo_id: str) -> str | None:
    """Return a canonical variant tag so semantically different builds do not
    end up in the same series bucket.

    Rules:

    - ``instruct``/``chat``/``it``/``sft`` collapse to ``instruct`` — same
      family of "fine-tuned to follow instructions".
    - ``base`` is its own bucket (base pretrained ≠ instruct-tuned).
    - Format suffixes (``gguf``, ``awq``, ``gptq``, ``hf``, ``onnx``,
      ``quantized``) are intentionally not variant tags: the same functional
      model shipped in a different format belongs in the same series so the
      ladder can promote the best size regardless of packaging.

    Functional specialisations (``coder``, ``vision``, ``embedding``) already
    live in the family key produced by ``families.detect_family`` and never
    reach this function — so a Coder-Instruct never collides with a general
    Instruct at the family level.
    """
    name = repo_id.split("/")[-1].lower()
    if any(word in name for word in ("instruct", "chat", "-it-", "-it2-")):
        return "instruct"
    if name.endswith("-it"):
        return "instruct"
    if "-sft" in name or name.endswith("-sft"):
        return "instruct"
    if "-base" in name or name.endswith("-base"):
        return "base"
    return None


def parameter_label(params: int | None) -> str:
    if params is None or params <= 0:
        return "unknown size"
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B".replace(".0B", "B")
    if params >= 1_000_000:
        return f"{params / 1_000_000:.0f}M"
    return f"{params}"


def group_by_series(
    evaluated: list[EvaluatedCandidate],
) -> dict[str, list[EvaluatedCandidate]]:
    """Bucket candidates by series, preserving first-seen order per bucket."""
    buckets: dict[str, list[EvaluatedCandidate]] = {}
    for item in evaluated:
        buckets.setdefault(series_key(item), []).append(item)
    return buckets


__all__ = [
    "group_by_series",
    "parameter_label",
    "series_key",
]
