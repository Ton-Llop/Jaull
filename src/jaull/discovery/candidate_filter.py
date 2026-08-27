"""Preliminary filtering of search results, before any expensive inspection.

Two rules shape this module:

* Reject only what is genuinely unusable — private repos, the wrong modality, a
  license that contradicts a hard requirement.
* Never reject a model merely for having thin metadata. Incomplete cards are the
  norm on the Hub; those candidates continue with a recorded penalty and a lower
  confidence instead of disappearing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from jaull.domain import licenses
from jaull.domain.candidates import ModelCandidate
from jaull.domain.estimation import (
    EstimationConfidence,
    HardwareFitMode,
    HardwareFitPlacementMethod,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.policies import TEXT_GENERATION_PIPELINE
from jaull.domain.requirements import UserRequirements
from jaull.estimator import policies as estimator_policies

# Tags that mean "this is not a plain text-generation model". This commit only
# supports text, so these are rejections rather than penalties.
_MULTIMODAL_TAGS: frozenset[str] = frozenset(
    {
        "image-to-text",
        "text-to-image",
        "text-to-speech",
        "text-to-video",
        "automatic-speech-recognition",
        "audio-classification",
        "image-classification",
        "visual-question-answering",
        "image-text-to-text",
        "video-text-to-text",
        "multimodal",
        "vision",
    }
)

# An adapter needs a base model to mean anything; without one there is nothing
# to estimate.
_ADAPTER_TAGS: frozenset[str] = frozenset({"peft", "lora", "adapter", "adapter-transformers"})

_GIB = 1024**3

# Cheap bytes-per-parameter assumptions for shortlist ordering only. They are
# deliberately separate from final MemoryEstimate results: the shortlist only
# decides which repos are worth inspecting, then the estimator reads real
# artifacts/configs and replaces these hints.
_COARSE_BYTES_PER_PARAM: dict[str, float] = {
    "q2": 0.35,
    "q3": 0.45,
    "q4": 0.60,
    "q5": 0.72,
    "q6": 0.86,
    "q8": 1.05,
    "int4": 0.60,
    "awq": 0.60,
    "gptq": 0.60,
    "int8": 1.05,
    "float16": 2.0,
    "bfloat16": 2.0,
}

_COARSE_DEFAULT_QUANTIZATION = estimator_policies.DEFAULT_GGUF_QUANTIZATION.lower()
_COARSE_DEFAULT_BYTES_PER_PARAM = _COARSE_BYTES_PER_PARAM["q4"]
_MIN_GPU_OFFLOAD_WEIGHT_FRACTION = 0.10
_MAX_TOO_LARGE_FALLBACKS = 2

_QUANT_RE = re.compile(r"(?<![a-z0-9])q([2-8])(?:_k(?:_[ms])?|_0)?(?![a-z0-9])")

_SCALE_BUCKETS: tuple[tuple[float, str], ...] = (
    (1.0, "tiny"),
    (4.0, "small"),
    (14.0, "medium"),
    (32.0, "large"),
)

_FIT_BONUS: dict[HardwareFitMode | None, float] = {
    HardwareFitMode.GPU_RESIDENT: 4.0,
    HardwareFitMode.GPU_OFFLOAD: 3.0,
    HardwareFitMode.CPU_RAM: 1.5,
    None: 0.0,
    HardwareFitMode.TOO_LARGE: -9.0,
}


@dataclass
class FilterOutcome:
    """What survived, and why the rest did not."""

    kept: list[ModelCandidate] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def reject(self, repo_id: str, reason: str) -> None:
        self.rejected.append((repo_id, reason))


@dataclass(frozen=True)
class CoarsePlacementHint:
    """Shortlist-only placement hint derived without deep inspection."""

    mode: HardwareFitMode | None
    placement_method: HardwareFitPlacementMethod
    confidence: EstimationConfidence
    estimated_weight_bytes: int | None = None
    estimated_total_bytes: int | None = None
    gpu_required_bytes: int | None = None
    ram_required_bytes: int | None = None
    gpu_weight_bytes: int = 0
    ram_weight_bytes: int = 0
    quantization_hint: str | None = None
    reason: str = ""


def deduplicate(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    """Collapse repeated sightings of a repo, preserving first-seen order.

    A repository showing up in several queries is a signal, not a reason to
    inspect it twice — the merged candidate keeps every query label.
    """
    merged: dict[str, ModelCandidate] = {}
    order: list[str] = []
    for candidate in candidates:
        if not candidate.repo_id:
            continue
        existing = merged.get(candidate.repo_id)
        if existing is None:
            merged[candidate.repo_id] = candidate
            order.append(candidate.repo_id)
        else:
            merged[candidate.repo_id] = existing.merged_with(candidate)
    return [merged[repo_id] for repo_id in order]


def filter_candidates(
    candidates: list[ModelCandidate], requirements: UserRequirements
) -> FilterOutcome:
    """Apply the preliminary rules, returning survivors and rejection reasons."""
    outcome = FilterOutcome()

    for candidate in candidates:
        rejection = _rejection_reason(candidate, requirements)
        if rejection is not None:
            outcome.reject(candidate.repo_id, rejection)
            continue
        outcome.kept.append(_apply_penalties(candidate))

    return outcome


def _rejection_reason(
    candidate: ModelCandidate, requirements: UserRequirements
) -> str | None:
    if candidate.private:
        return "Repository is private."
    if candidate.gated:
        return "Repository is gated and cannot be inspected without access."

    tags = {tag.lower() for tag in candidate.tags}

    if tags & _MULTIMODAL_TAGS:
        return "Model is multimodal; this version only recommends text models."

    pipeline = candidate.pipeline_tag
    if pipeline and pipeline != TEXT_GENERATION_PIPELINE:
        return f"Pipeline {pipeline!r} is not text generation."

    if tags & _ADAPTER_TAGS and not candidate.base_model_repo_id:
        return "Adapter without a resolvable base model."

    if requirements.commercial_use_required:
        category = licenses.classify_license(candidate.license)
        if category is licenses.LicenseCategory.COMMERCIAL_RESTRICTED:
            return (
                f"License {candidate.license!r} is not generally suitable for "
                "commercial use."
            )

    return None


def _apply_penalties(candidate: ModelCandidate) -> ModelCandidate:
    """Keep the candidate but record what makes it less trustworthy."""
    penalties = list(candidate.penalties)
    confidence = candidate.metadata_confidence

    if not candidate.license:
        penalties.append("No license declared in the model card.")
        confidence = _lower(confidence)
    if not candidate.languages:
        penalties.append("No languages declared in the model card.")
        confidence = _lower(confidence)
    if candidate.pipeline_tag is None:
        penalties.append("No pipeline tag declared; task match is inferred from tags.")
        confidence = _lower(confidence)

    if not penalties:
        return candidate
    return candidate.model_copy(
        update={"penalties": penalties, "metadata_confidence": confidence}
    )


# Parameter counts are conventionally encoded in the repository name
# ("Qwen2.5-Coder-7B-Instruct"). Matched case-insensitively on a word boundary
# so "7B" is found but "B7" or a bare "7" is not.
_PARAM_COUNT_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def parameter_count_hint(repo_id: str) -> float | None:
    """Best-effort parameter count in billions, read from the repository name.

    A naming *heuristic*, used only to decide which candidates are worth
    inspecting. It never reaches a reported figure: once a candidate is
    inspected, every number comes from real file sizes and configuration
    metadata. Returns ``None`` when the name says nothing.
    """
    matches = _PARAM_COUNT_RE.findall(repo_id)
    if not matches:
        return None
    try:
        # Largest match wins: "Qwen3-Coder-30B-A3B" is a 30B model.
        return max(float(value) for value in matches)
    except ValueError:  # pragma: no cover - regex already constrains the shape
        return None


def shortlist(
    candidates: list[ModelCandidate],
    requirements: UserRequirements,
    limit: int,
    budget_bytes: int | None = None,
    *,
    hardware: HardwareProfile | None = None,
) -> list[ModelCandidate]:
    """Choose which candidates are worth the cost of deep inspection.

    Deep inspection is the expensive stage, so the budget has to go to the most
    plausible repositories. This is a *cheap* pre-ranking over search metadata
    only — the real, hardware-aware ranking happens later and can still demote
    anything selected here.

    Popularity carries real weight at this stage, and deliberately so: it is the
    only pre-inspection signal separating a maintained release from a
    19-download experiment. It stays a secondary signal in the final ranking,
    where hardware fit dominates.

    When ``budget_bytes`` is known, models whose name implies they cannot
    possibly fit are pushed down. Without this the whole inspection budget goes
    to the most-downloaded 30B repositories on a machine that can only run a
    3B one, and the run returns almost nothing.
    """
    if hardware is not None:
        return _hardware_aware_shortlist(candidates, requirements, limit, hardware)

    preferred = requirements.preferred_formats[:1]
    budget_gib = budget_bytes / 1024**3 if budget_bytes else None

    def key(candidate: ModelCandidate) -> tuple[float, str]:
        score = _cheap_desirability(candidate, preferred)
        score += _size_bonus(candidate, budget_gib)
        return (-score, candidate.repo_id)

    return sorted(candidates, key=key)[:limit]


def coarse_placement_hint(
    candidate: ModelCandidate,
    requirements: UserRequirements,
    hardware: HardwareProfile,
) -> CoarsePlacementHint:
    """Estimate a candidate's coarse execution envelope without inspecting it.

    This is intentionally less precise than ``MemoryEstimate``. It uses only
    search metadata already present on ``ModelCandidate`` plus physical memory
    capacity from the hardware scan, so it cannot increase Hub calls or turn the
    shortlist into deep inspection of every candidate.
    """
    billions = parameter_count_hint(candidate.repo_id)
    if billions is None:
        return CoarsePlacementHint(
            mode=None,
            placement_method=HardwareFitPlacementMethod.NONE,
            confidence=EstimationConfidence.UNKNOWN,
            reason="No parameter-count hint is available before inspection.",
        )

    quantization = _quantization_hint(candidate)
    bytes_per_param = _bytes_per_parameter_hint(quantization)
    weights = max(1, int(billions * 1_000_000_000 * bytes_per_param))
    kv_cache = _coarse_kv_cache_bytes(billions, requirements)
    overhead = _coarse_overhead_bytes(weights)
    ram_total = hardware.memory.total_bytes
    vram_total = _planning_vram_bytes(hardware)
    unified = _has_unified_memory(hardware)

    cpu_total = _with_safety_margin(weights + kv_cache + overhead)
    if vram_total is None or unified:
        if cpu_total <= ram_total:
            return CoarsePlacementHint(
                mode=HardwareFitMode.CPU_RAM,
                placement_method=HardwareFitPlacementMethod.ESTIMATED_BYTES,
                confidence=EstimationConfidence.LOW,
                estimated_weight_bytes=weights,
                estimated_total_bytes=cpu_total,
                ram_required_bytes=cpu_total,
                ram_weight_bytes=weights,
                quantization_hint=quantization,
                reason="Coarse estimate fits system memory.",
            )
        return CoarsePlacementHint(
            mode=HardwareFitMode.TOO_LARGE,
            placement_method=HardwareFitPlacementMethod.NONE,
            confidence=EstimationConfidence.LOW,
            estimated_weight_bytes=weights,
            estimated_total_bytes=cpu_total,
            ram_required_bytes=cpu_total,
            ram_weight_bytes=weights,
            quantization_hint=quantization,
            reason="Coarse estimate exceeds system memory.",
        )

    reserve = estimator_policies.DEVICE_RESERVE_DEFAULT_BYTES
    resident_gpu_required = _with_safety_margin(weights + kv_cache + overhead + reserve)
    if resident_gpu_required <= vram_total:
        return CoarsePlacementHint(
            mode=HardwareFitMode.GPU_RESIDENT,
            placement_method=HardwareFitPlacementMethod.ESTIMATED_BYTES,
            confidence=EstimationConfidence.LOW,
            estimated_weight_bytes=weights,
            estimated_total_bytes=resident_gpu_required,
            gpu_required_bytes=resident_gpu_required,
            gpu_weight_bytes=weights,
            quantization_hint=quantization,
            reason="Coarse estimate fits fully in GPU memory.",
        )

    offload = _coarse_offload_hint(weights, kv_cache, overhead, vram_total, ram_total)
    if offload is not None:
        return replace(offload, quantization_hint=quantization)

    if cpu_total <= ram_total:
        return CoarsePlacementHint(
            mode=HardwareFitMode.CPU_RAM,
            placement_method=HardwareFitPlacementMethod.ESTIMATED_BYTES,
            confidence=EstimationConfidence.LOW,
            estimated_weight_bytes=weights,
            estimated_total_bytes=cpu_total,
            ram_required_bytes=cpu_total,
            ram_weight_bytes=weights,
            quantization_hint=quantization,
            reason="Coarse GPU placement is not viable, but CPU RAM is plausible.",
        )

    return CoarsePlacementHint(
        mode=HardwareFitMode.TOO_LARGE,
        placement_method=HardwareFitPlacementMethod.NONE,
        confidence=EstimationConfidence.LOW,
        estimated_weight_bytes=weights,
        estimated_total_bytes=cpu_total,
        gpu_required_bytes=resident_gpu_required,
        ram_required_bytes=cpu_total,
        ram_weight_bytes=weights,
        quantization_hint=quantization,
        reason="No coarse GPU or CPU placement fits the planning budget.",
    )


def _hardware_aware_shortlist(
    candidates: list[ModelCandidate],
    requirements: UserRequirements,
    limit: int,
    hardware: HardwareProfile,
) -> list[ModelCandidate]:
    if limit <= 0:
        return []

    preferred = requirements.preferred_formats[:1]
    entries: list[_ShortlistEntry] = []
    for candidate in candidates:
        hint = coarse_placement_hint(candidate, requirements, hardware)
        entries.append(
            _ShortlistEntry(
                candidate=candidate,
                hint=hint,
                score=_cheap_desirability(candidate, preferred) + _FIT_BONUS[hint.mode],
            )
        )
    entries.sort(key=lambda entry: (-entry.score, entry.candidate.repo_id))
    too_large_entries = [
        entry for entry in entries if entry.hint.mode is HardwareFitMode.TOO_LARGE
    ]
    plausible_entries = [
        entry for entry in entries if entry.hint.mode is not HardwareFitMode.TOO_LARGE
    ]

    selected: list[_ShortlistEntry] = []
    selected_redundancy: set[tuple[str, str, HardwareFitMode | None]] = set()
    selected_families: set[str] = set()

    buckets = _group_shortlist_buckets(plausible_entries, hardware)
    for bucket in buckets:
        bucket.sort(key=lambda entry: (-entry.score, entry.candidate.repo_id))

    while len(selected) < limit:
        before = len(selected)
        for bucket in buckets:
            if len(selected) >= limit:
                break
            picked = _pop_bucket_candidate(
                bucket, selected_redundancy, selected_families
            )
            if picked is None:
                continue
            _remember_selection(picked, selected, selected_redundancy, selected_families)
        if len(selected) == before:
            break

    remaining = [entry for bucket in buckets for entry in bucket]
    remaining.sort(key=lambda entry: (-entry.score, entry.candidate.repo_id))
    for entry in remaining:
        if len(selected) >= limit:
            break
        _remember_selection(entry, selected, selected_redundancy, selected_families)

    too_large_entries.sort(key=lambda entry: (-entry.score, entry.candidate.repo_id))
    too_large_budget = min(_MAX_TOO_LARGE_FALLBACKS, limit - len(selected))
    if selected:
        too_large_entries = too_large_entries[:too_large_budget]
    else:
        too_large_entries = too_large_entries[: min(limit, _MAX_TOO_LARGE_FALLBACKS)]
    for entry in too_large_entries:
        if len(selected) >= limit:
            break
        _remember_selection(entry, selected, selected_redundancy, selected_families)

    return [entry.candidate for entry in selected[:limit]]


@dataclass(frozen=True)
class _ShortlistEntry:
    candidate: ModelCandidate
    hint: CoarsePlacementHint
    score: float


def _cheap_desirability(
    candidate: ModelCandidate, preferred_formats: list[str]
) -> float:
    score = math.log1p(max(0, candidate.downloads))
    score += 0.5 * math.log1p(max(0, candidate.likes))
    tags = {tag.lower() for tag in candidate.tags}
    if preferred_formats and preferred_formats[0] in tags:
        score += 2.0
    score += 0.75 * (len(candidate.source_queries) - 1)
    score -= 0.5 * len(candidate.penalties)
    return score


def _coarse_offload_hint(
    weights: int,
    kv_cache: int,
    overhead: int,
    vram_total: int,
    ram_total: int,
) -> CoarsePlacementHint | None:
    reserve = estimator_policies.DEVICE_RESERVE_DEFAULT_BYTES
    gpu_fixed = kv_cache + reserve
    if _with_safety_margin(gpu_fixed) >= vram_total:
        return None

    gpu_weight_bytes = _max_gpu_weight_bytes(weights, kv_cache, overhead, vram_total)
    ram_weight_bytes = weights - gpu_weight_bytes
    if (
        ram_weight_bytes <= 0
        or gpu_weight_bytes / weights < _MIN_GPU_OFFLOAD_WEIGHT_FRACTION
    ):
        return None

    gpu_overhead = _split_proportionally(overhead, gpu_weight_bytes, weights)
    ram_overhead = overhead - gpu_overhead
    gpu_required = _with_safety_margin(gpu_fixed + gpu_weight_bytes + gpu_overhead)
    ram_required = _with_safety_margin(ram_weight_bytes + ram_overhead)
    if gpu_required <= vram_total and ram_required <= ram_total:
        return CoarsePlacementHint(
            mode=HardwareFitMode.GPU_OFFLOAD,
            placement_method=HardwareFitPlacementMethod.ESTIMATED_BYTES,
            confidence=EstimationConfidence.LOW,
            estimated_weight_bytes=weights,
            estimated_total_bytes=gpu_required + ram_required,
            gpu_required_bytes=gpu_required,
            ram_required_bytes=ram_required,
            gpu_weight_bytes=gpu_weight_bytes,
            ram_weight_bytes=ram_weight_bytes,
            reason="Coarse byte placement fits GPU weights plus host spillover.",
        )
    return None


def _group_shortlist_buckets(
    entries: list[_ShortlistEntry], hardware: HardwareProfile
) -> list[list[_ShortlistEntry]]:
    buckets: dict[tuple[int, int], list[_ShortlistEntry]] = {}
    for entry in entries:
        key = (
            _mode_order(entry.hint.mode, hardware),
            _scale_order(parameter_count_hint(entry.candidate.repo_id)),
        )
        buckets.setdefault(key, []).append(entry)
    return [buckets[key] for key in sorted(buckets)]


def _pop_bucket_candidate(
    bucket: list[_ShortlistEntry],
    selected_redundancy: set[tuple[str, str, HardwareFitMode | None]],
    selected_families: set[str],
) -> _ShortlistEntry | None:
    if not bucket:
        return None
    best_index = 0
    for index, entry in enumerate(bucket):
        if (
            _redundancy_key(entry) not in selected_redundancy
            and _family_key(entry.candidate) not in selected_families
        ):
            best_index = index
            break
    else:
        for index, entry in enumerate(bucket):
            if _redundancy_key(entry) not in selected_redundancy:
                best_index = index
                break
    return bucket.pop(best_index)


def _remember_selection(
    entry: _ShortlistEntry,
    selected: list[_ShortlistEntry],
    selected_redundancy: set[tuple[str, str, HardwareFitMode | None]],
    selected_families: set[str],
) -> None:
    selected.append(entry)
    selected_redundancy.add(_redundancy_key(entry))
    selected_families.add(_family_key(entry.candidate))


def _mode_order(mode: HardwareFitMode | None, hardware: HardwareProfile) -> int:
    if _planning_vram_bytes(hardware) is None or _has_unified_memory(hardware):
        order: dict[HardwareFitMode | None, int] = {
            HardwareFitMode.CPU_RAM: 0,
            None: 1,
            HardwareFitMode.GPU_RESIDENT: 2,
            HardwareFitMode.GPU_OFFLOAD: 3,
            HardwareFitMode.TOO_LARGE: 4,
        }
    else:
        order = {
            HardwareFitMode.GPU_RESIDENT: 0,
            HardwareFitMode.GPU_OFFLOAD: 1,
            HardwareFitMode.CPU_RAM: 2,
            None: 3,
            HardwareFitMode.TOO_LARGE: 4,
        }
    return order[mode]


def _scale_order(billions: float | None) -> int:
    bucket = _scale_bucket(billions)
    order = {
        "tiny": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "huge": 4,
        "unknown": 5,
    }
    return order[bucket]


def _redundancy_key(
    entry: _ShortlistEntry,
) -> tuple[str, str, HardwareFitMode | None]:
    return (
        _family_key(entry.candidate),
        _scale_bucket(parameter_count_hint(entry.candidate.repo_id)),
        entry.hint.mode,
    )


def _family_key(candidate: ModelCandidate) -> str:
    model_name = candidate.repo_id.rsplit("/", maxsplit=1)[-1].lower()
    normalized = re.sub(r"[-_.]+", " ", model_name)
    tokens = [
        token
        for token in normalized.split()
        if token
        and token
        not in {
            "instruct",
            "chat",
            "assistant",
            "gguf",
            "awq",
            "gptq",
            "safetensors",
            "model",
        }
        and parameter_count_hint(token) is None
        and not re.fullmatch(r"q[2-8](?:_k(?:_[ms])?|_0)?", token)
    ]
    return tokens[0] if tokens else candidate.repo_id.split("/", maxsplit=1)[0].lower()


def _scale_bucket(billions: float | None) -> str:
    if billions is None:
        return "unknown"
    for upper, label in _SCALE_BUCKETS:
        if billions <= upper:
            return label
    return "huge"


def _quantization_hint(candidate: ModelCandidate) -> str:
    haystack = " ".join([candidate.repo_id, *candidate.tags]).lower()
    if "awq" in haystack:
        return "awq"
    if "gptq" in haystack:
        return "gptq"
    if "int8" in haystack or "8-bit" in haystack:
        return "int8"
    if "int4" in haystack or "4-bit" in haystack:
        return "int4"
    if "bfloat16" in haystack or "bf16" in haystack:
        return "bfloat16"
    if "float16" in haystack or "fp16" in haystack:
        return "float16"
    if match := _QUANT_RE.search(haystack):
        return f"q{match.group(1)}"
    return _COARSE_DEFAULT_QUANTIZATION


def _bytes_per_parameter_hint(quantization: str) -> float:
    if quantization.startswith("q") and len(quantization) >= 2:
        return _COARSE_BYTES_PER_PARAM.get(quantization[:2], _COARSE_DEFAULT_BYTES_PER_PARAM)
    return _COARSE_BYTES_PER_PARAM.get(quantization, _COARSE_DEFAULT_BYTES_PER_PARAM)


def _coarse_kv_cache_bytes(billions: float, requirements: UserRequirements) -> int:
    context_factor = max(1.0, requirements.desired_context / 4096)
    user_factor = max(1, requirements.concurrent_users)
    gib = max(0.15, billions * 0.035 * context_factor * user_factor)
    return int(gib * _GIB)


def _coarse_overhead_bytes(weights: int) -> int:
    return max(
        estimator_policies.OVERHEAD_MIN_BYTES,
        int(
            estimator_policies.OVERHEAD_BASE_BYTES
            + weights * estimator_policies.OVERHEAD_WEIGHT_FRACTION
        ),
    )


def _with_safety_margin(bytes_required: int) -> int:
    return math.ceil(
        bytes_required
        * (1 + estimator_policies.SAFETY_MARGIN_DEFAULT_PERCENT / 100)
    )


def _split_proportionally(total: int, part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return min(total, max(0, int(total * (part / whole))))


def _max_gpu_weight_bytes(
    weights: int,
    kv_cache: int,
    overhead: int,
    vram_total: int,
) -> int:
    low = 0
    high = weights
    reserve = estimator_policies.DEVICE_RESERVE_DEFAULT_BYTES
    while low < high:
        candidate = (low + high + 1) // 2
        gpu_overhead = _split_proportionally(overhead, candidate, weights)
        gpu_required = _with_safety_margin(kv_cache + reserve + candidate + gpu_overhead)
        if gpu_required <= vram_total:
            low = candidate
        else:
            high = candidate - 1
    return low


def _planning_vram_bytes(hardware: HardwareProfile) -> int | None:
    if not hardware.gpus:
        return None
    return max(gpu.vram_total_bytes for gpu in hardware.gpus)


def _has_unified_memory(hardware: HardwareProfile) -> bool:
    return any(accelerator.shared_memory for accelerator in hardware.accelerators)


def _size_bonus(candidate: ModelCandidate, budget_gib: float | None) -> float:
    """Reward names that suggest the model fits; penalise ones that cannot."""
    if budget_gib is None:
        return 0.0
    billions = parameter_count_hint(candidate.repo_id)
    if billions is None:
        return 0.0
    # ~0.6 GiB per billion parameters at 4-bit, the smallest quantization we
    # would realistically recommend. Anything above that cannot fit at all.
    smallest_plausible_gib = billions * 0.6
    if smallest_plausible_gib > budget_gib:
        return -6.0
    if smallest_plausible_gib > budget_gib * 0.7:
        return -1.5
    return 1.5


_CONFIDENCE_LADDER: tuple[EstimationConfidence, ...] = (
    EstimationConfidence.HIGH,
    EstimationConfidence.MEDIUM,
    EstimationConfidence.LOW,
    EstimationConfidence.UNKNOWN,
)


def _lower(confidence: EstimationConfidence) -> EstimationConfidence:
    index = _CONFIDENCE_LADDER.index(confidence)
    return _CONFIDENCE_LADDER[min(index + 1, len(_CONFIDENCE_LADDER) - 1)]


__all__ = [
    "CoarsePlacementHint",
    "FilterOutcome",
    "coarse_placement_hint",
    "deduplicate",
    "filter_candidates",
    "parameter_count_hint",
    "shortlist",
]
