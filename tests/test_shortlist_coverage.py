"""Bounded exploration must compare plausible alternatives, not only popular copies."""

import json
import math

import pytest

from jaull.discovery.candidate_filter import coarse_placement_hint, filter_candidates, shortlist
from jaull.domain.candidates import ModelCandidate
from jaull.domain.estimation import HardwareFitMode
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import RecommendationPriority, UseCase, UserRequirements
from jaull.recommendation.scoring import targets_another_task
from tests._workflow_fixtures import candidate, hardware
from tests.test_discovery_search import _requirements, _saved_chat_candidates


def test_audit_preserves_inputs_and_replays_the_same_selection_and_decisions():
    req = _requirements(UseCase.GENERAL_CHAT)
    profile = hardware(vram_gib=24, ram_gib=14)
    pool = [*_saved_chat_candidates(), candidate(
        "other/Alternative-24B-Instruct-GGUF", tags=["gguf", "instruct"],
        downloads=40_000, likes=0,
    ).model_copy(update={"penalties": ["incomplete metadata"]})]
    audit = {}
    picked = shortlist(pool, req, 12, hardware=profile,
                       targets_other_task=lambda item: targets_another_task(item, req), audit=audit)
    assert picked == shortlist(pool, req, 12, hardware=profile,
                               targets_other_task=lambda item: targets_another_task(item, req))
    saved = json.loads(json.dumps(audit))
    rows = saved["candidates"]
    assert rows[pool[-1].repo_id]["input"] == pool[-1].model_dump(mode="json")
    for row in rows.values():
        assert sum(row["score_components"].values()) + row["fit_bonus"] == row["score"]
    assert any(step["stage"] == "coverage" for step in saved["steps"])
    assert any("exploration_score_floor" in step for step in saved["steps"])
    replay = {}
    actual = shortlist(
        [ModelCandidate.model_validate(row["input"]) for row in rows.values()],
        UserRequirements.model_validate(saved["requirements"]), saved["limit"],
        hardware=HardwareProfile.model_validate(saved["hardware"]),
        targets_other_task=lambda item: rows[item.repo_id]["other_task"], audit=replay,
    )
    assert actual == picked
    assert json.loads(json.dumps(replay)) == saved


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("family", ["Mistral-Small", "Alternative-Assistant"])
def test_a_24b_alternative_is_inspected_before_the_pool_fills_with_small_models(reverse, family):
    req = _requirements(UseCase.GENERAL_CHAT)
    alternative = candidate(
        f"other/{family}-24B-Instruct-GGUF", tags=["gguf", "instruct"],
        downloads=44118, likes=20,
    )
    # A size-unknown search hit with no declared text task or runnable GGUF is
    # not guaranteed an exploration slot merely because it is very popular.
    unknown = candidate("other/Unspecified", tags=[], pipeline_tag=None, downloads=2_000_000)
    pool = [*_saved_chat_candidates(), alternative, unknown]
    if reverse:
        pool.reverse()
    picked = shortlist(
        pool, req, 12, hardware=hardware(vram_gib=24, ram_gib=14),
        targets_other_task=lambda item: targets_another_task(item, req),
    )
    ids = [item.repo_id for item in picked]
    assert alternative.repo_id in ids
    assert unknown.repo_id not in ids
    assert len(ids) == len(set(ids)) == 12


def test_unknown_candidates_remain_available_when_the_budget_is_not_full():
    req = _requirements(UseCase.GENERAL_CHAT)
    unknown = candidate("other/Unspecified", tags=[], pipeline_tag=None)
    assert shortlist([unknown], req, 12, hardware=hardware())[0] == unknown


@pytest.mark.parametrize(
    "priority", [RecommendationPriority.QUALITY, RecommendationPriority.BALANCED],
)
def test_exploration_does_not_replace_strong_candidates_with_an_obscure_family(priority):
    obscure = candidate("other/Obscure-24B-Instruct-GGUF", tags=["gguf"], downloads=1, likes=0)
    req = _requirements(UseCase.GENERAL_CHAT).model_copy(update={"priority": priority})
    audit = {}
    picked = shortlist(
        [*_saved_chat_candidates(), obscure], req, 12,
        hardware=hardware(vram_gib=24, ram_gib=14),
        targets_other_task=lambda item: targets_another_task(item, req),
        audit=audit,
    )
    assert obscure.repo_id not in {item.repo_id for item in picked}
    attempts = [step for step in audit["steps"]
                if obscure.repo_id in step.get("exploration_group", [])]
    assert attempts
    assert all(obscure.repo_id not in step["eligible_novel_candidates"] for step in attempts)
    row = audit["candidates"][obscure.repo_id]
    score = (row["score_components"]["downloads"]
             if priority is RecommendationPriority.QUALITY
             else sum(row["score_components"].values()))
    assert all(score < step["exploration_score_floor"] for step in attempts)


@pytest.mark.parametrize("family", ["Mistral-Small", "Alternative-Assistant"])
@pytest.mark.parametrize("reverse", [False, True])
def test_test5_popularity_inputs_do_not_block_a_plausible_24b_alternative(family, reverse):
    # Decision-relevant slice of the user's 2026-10-30 test5 audit, not live Hub metadata.
    records = [
        ("ornith-ai/Ornith-1.5-9B-GGUF", 5268636, 613, 1, 0),
        ("ornith-ai/Ornith-1.0-9B-GGUF", 2411236, 698, 1, 0),
        ("unsloth/Qwen3.8-27B-GGUF", 6078631, 5409, 1, 1),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", 656626, 204, 3, 0),
        ("ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF", 1572180, 460, 1, 0),
        ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", 372944, 427, 2, 0),
        ("bartowski/Qwen2.5-32B-Instruct-GGUF", 892605, 93, 2, 0),
        ("bartowski/Qwen2.5-14B-Instruct-GGUF", 167230, 75, 2, 0),
        ("DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-18.4B-GGUF", 38574, 669, 2, 0),
        ("TheBloke/Llama-2-7B-Chat-GGUF", 63657, 517, 1, 0),
        ("LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF", 90711, 221, 1, 0),
        ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 46420, 523, 1, 0),
        ("JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q8_0", 138822, 9, 2, 0),
        ("bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF", 72877, 31, 1, 1),
        (f"MaziyarPanahi/{family}-24B-Instruct-2501-GGUF", 40019, 12, 1, 0),
    ]
    pool = [candidate(repo, tags=["gguf"], downloads=downloads, likes=likes,
                      queries=[f"q{i}" for i in range(queries)]).model_copy(update={
                          "penalties": ["missing metadata"] * penalties,
                      }) for repo, downloads, likes, queries, penalties in records]
    alternative = pool[-1]
    req = _requirements(UseCase.GENERAL_CHAT).model_copy(update={
        "priority": RecommendationPriority.QUALITY,
        "preferred_formats": ["gguf", "safetensors"],
    })
    audit = {}
    picked = shortlist(list(reversed(pool)) if reverse else pool, req, 12,
                       hardware=hardware(vram_gib=24, ram_gib=14), audit=audit)
    assert alternative.repo_id in {item.repo_id for item in picked}
    assert len(picked) == 12
    row = audit["candidates"][alternative.repo_id]
    # The old compound floor in the actual report rejected this same input.
    assert row["score"] - row["fit_bonus"] == pytest.approx(13.879609286868494)
    assert row["score"] - row["fit_bonus"] < 15.365305636921851
    step = next(step for step in audit["steps"] if step["repo_id"] == alternative.repo_id)
    assert step["stage"] == "exploration"
    assert step["exploration_score_axis"] == "log_downloads"
    assert math.log1p(alternative.downloads) >= step["exploration_score_floor"]


@pytest.mark.parametrize("library,tags", [
    ("diffusion-single-file", ["gguf", "i2v"]),
    ("Diffusers", ["gguf"]),
    (None, ["gguf", "diffusion-single-file"]),
])
def test_diffusion_metadata_is_filtered_even_with_a_gguf_tag(library, tags):
    item = candidate("org/Unknown", tags=tags, pipeline_tag=None).model_copy(update={
        "library_name": library,
    })
    outcome = filter_candidates([item], _requirements(UseCase.GENERAL_CHAT))
    assert outcome.kept == []
    assert outcome.rejected[0][0] == item.repo_id
    assert "diffusion" in outcome.rejected[0][1]


def test_sparse_gguf_metadata_is_not_rejected_without_a_non_text_signal():
    item = candidate("org/Unknown", tags=["gguf"], pipeline_tag=None)
    assert filter_candidates([item], _requirements(UseCase.GENERAL_CHAT)).kept


@pytest.mark.parametrize("reverse", [False, True])
def test_quality_compares_a_range_of_large_models_despite_popular_small_candidates(reverse):
    req = _requirements(UseCase.GENERAL_CHAT).model_copy(update={
        "priority": RecommendationPriority.QUALITY,
    })
    large = [candidate(f"org/Assistant-{size}B-Instruct-GGUF", tags=["gguf", "instruct"],
                       downloads=100_000, likes=100) for size in (14, 16, 20, 24, 27, 30)]
    small = [candidate(f"org/Popular{index}-9B-Instruct-GGUF", tags=["gguf", "instruct"],
                       downloads=1_000_000, likes=500) for index in range(20)]
    pool = [*large, *small]
    picked = shortlist(list(reversed(pool)) if reverse else pool, req, 12,
                       hardware=hardware(vram_gib=24, ram_gib=14))
    ids = {item.repo_id for item in picked}
    assert {item.repo_id for item in large} <= ids
    assert len(picked) == 12


def test_generic_gguf_precision_tags_do_not_force_one_published_variant():
    req = _requirements(UseCase.GENERAL_CHAT)
    profile = hardware(vram_gib=24, ram_gib=14)
    generic = candidate("org/Assistant-24B-Instruct-GGUF", tags=["gguf", "fp16", "Q8_0"])
    hint = coarse_placement_hint(generic, req, profile)
    assert hint.quantization_hint == "q4_k_m"
    assert hint.mode is HardwareFitMode.GPU_RESIDENT
    explicit = generic.model_copy(update={"repo_id": "org/Assistant-24B-Q8_0-GGUF"})
    assert coarse_placement_hint(explicit, req, profile).quantization_hint == "q8"


@pytest.mark.parametrize("reverse", [False, True])
def test_quality_inspects_coarse_offload_alternatives_before_popular_small_models(reverse):
    req = _requirements(UseCase.GENERAL_CHAT).model_copy(update={
        "priority": RecommendationPriority.QUALITY,
    })
    profile = hardware(vram_gib=24, ram_gib=14)
    alternative = candidate("org/Alternative-24B-Q6_K-GGUF", tags=["gguf", "instruct"],
                            downloads=100_000, likes=100)
    assert coarse_placement_hint(alternative, req, profile).mode is HardwareFitMode.GPU_OFFLOAD
    pool = [*_saved_chat_candidates(), alternative]
    picked = shortlist(list(reversed(pool)) if reverse else pool, req, 12, hardware=profile,
                       targets_other_task=lambda item: targets_another_task(item, req))
    assert alternative.repo_id in {item.repo_id for item in picked}
    assert len(picked) == 12
