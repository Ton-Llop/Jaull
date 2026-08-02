"""Translate plain-language wizard answers into normalised technical requirements.

This is the only place where "Programming and code" becomes ``coding`` and
"Long documents" becomes a token count. Keeping it out of the Textual screens is
what makes the mapping testable without a terminal.
"""

from __future__ import annotations

import re

from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.workflow import policies
from local_ai_check.workflow.models import (
    CommercialUse,
    UseCase,
    UserAnswers,
    UserRequirements,
)

# Human labels the wizard shows -> ISO 639-1 codes. Hugging Face model cards use
# these codes in their `language` field, so normalising here means the language
# scorer can compare like with like.
_LANGUAGE_CODES: dict[str, str] = {
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    "castellano": "es",
    "catalan": "ca",
    "català": "ca",
    "english": "en",
    "ingles": "en",
    "inglés": "en",
    "portuguese": "pt",
    "portugues": "pt",
    "português": "pt",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "basque": "eu",
    "galician": "gl",
}

# A bare 2- or 3-letter token is assumed to already be a language code.
_CODE_RE = re.compile(r"^[a-z]{2,3}$")

# VRAM below which a server-style runtime (full-precision safetensors on GPU) is
# not a realistic target and quantized GGUF is the sensible default.
_GGUF_FIRST_VRAM_BYTES = 12 * 1024**3


def normalize_language(value: str) -> str | None:
    """Normalise one user-supplied language to a code, or ``None`` if unusable."""
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    # Strip a parenthesised qualifier such as "Spanish (Spain)".
    cleaned = cleaned.split("(")[0].strip()
    if cleaned in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[cleaned]
    if _CODE_RE.match(cleaned):
        return cleaned
    return None


def normalize_languages(values: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(codes, rejected)`` preserving order and dropping duplicates."""
    codes: list[str] = []
    rejected: list[str] = []
    for value in values:
        code = normalize_language(value)
        if code is None:
            if value.strip():
                rejected.append(value.strip())
            continue
        if code not in codes:
            codes.append(code)
    return codes, rejected


def build_requirements(
    answers: UserAnswers, hardware: HardwareProfile | None = None
) -> UserRequirements:
    """Turn wizard answers into the normalised requirements the pipeline consumes."""
    assumptions: list[str] = []

    languages, rejected = normalize_languages(
        [*answers.languages, *answers.other_languages]
    )
    for value in rejected:
        assumptions.append(
            f"Ignored {value!r}: not recognised as a language name or code."
        )
    if not languages:
        languages = ["en"]
        assumptions.append(
            "No language selected; assuming English when comparing model metadata."
        )

    users, concurrency_range = policies.CONCURRENCY_BUCKETS[answers.concurrency.value]
    if users > 1:
        assumptions.append(
            f"Concurrency treated as a ranking signal for {concurrency_range.lower()}; "
            "the tool does not model real throughput or latency."
        )

    context = _resolve_context(answers, assumptions)
    commercial = _resolve_commercial(answers.commercial_use, assumptions)
    formats = _preferred_formats(hardware, assumptions)

    return UserRequirements(
        use_case=answers.use_case,
        priority=answers.priority,
        languages=languages,
        concurrent_users=users,
        concurrency_range=concurrency_range,
        desired_context=context,
        commercial_use_required=commercial,
        pipeline_tag=policies.TEXT_GENERATION_PIPELINE,
        preferred_formats=formats,
        assumptions=assumptions,
    )


def _resolve_context(answers: UserAnswers, assumptions: list[str]) -> int:
    if answers.use_case is not UseCase.DOCUMENT_QA:
        return policies.DEFAULT_CONTEXT_TOKENS
    if answers.document_scale is None:
        assumptions.append(
            "Document scale not answered; assuming the default context window."
        )
        return policies.DEFAULT_CONTEXT_TOKENS
    context = policies.DOCUMENT_CONTEXT_TOKENS[answers.document_scale.value]
    assumptions.append(
        f"Context set to {context} tokens from the document-size answer. This is the "
        "model's context window, not the size of a document collection: a retrieval "
        "system feeds it a few chunks at a time."
    )
    return context


def _resolve_commercial(
    answer: CommercialUse, assumptions: list[str]
) -> bool | None:
    if answer is CommercialUse.YES:
        return True
    if answer is CommercialUse.NO:
        return False
    assumptions.append(
        "Commercial use unspecified; licenses are reported but not used to exclude models."
    )
    return None


def _preferred_formats(
    hardware: HardwareProfile | None, assumptions: list[str]
) -> list[str]:
    """Rank weight formats by how realistic they are on this machine.

    GGUF first on personal hardware: it is the format that runs well on CPU and
    on modest GPUs. Safetensors comes first only when there is enough VRAM for a
    server-style runtime to be a credible target.
    """
    if hardware is None or not hardware.gpus:
        assumptions.append(
            "No NVIDIA GPU detected; preferring quantized GGUF builds, which run on CPU."
        )
        return ["gguf", "safetensors"]

    vram = max(gpu.vram_total_bytes for gpu in hardware.gpus)
    if vram >= _GGUF_FIRST_VRAM_BYTES:
        assumptions.append(
            f"{vram / 1024**3:.0f} GiB of VRAM detected; safetensors builds are a "
            "realistic target alongside GGUF."
        )
        return ["safetensors", "gguf"]

    assumptions.append(
        f"{vram / 1024**3:.0f} GiB of VRAM detected; preferring quantized GGUF builds."
    )
    return ["gguf", "safetensors"]


__all__ = ["build_requirements", "normalize_language", "normalize_languages"]
