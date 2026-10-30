"""Small-machine coverage must not undo the 24 GiB shortlist policy."""

from __future__ import annotations

import pytest

from jaull.application.recommendation import policies
from jaull.application.requirements import build_requirements
from jaull.discovery.candidate_filter import largest_fitting_billions, shortlist, size_bands
from jaull.discovery.query_builder import build_queries
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.recommendation.scoring import targets_another_task
from jaull.workflow import orchestrator
from jaull.workflow.models import WorkflowStep
from tests._workflow_fixtures import (
    GIB,
    FakeSearchClient,
    answers,
    candidate,
    gguf_analysis,
    hardware,
)
from tests.test_discovery_search import _chat_shortlist, _saved_chat_candidates
from tests.test_workflow_orchestrator import _container


def _laptop(integrated: bool) -> HardwareProfile:
    profile = hardware(ram_gib=8, vram_gib=None).model_copy(update={
        "memory": MemoryInfo(total_bytes=int(7.5 * GIB), available_bytes=6 * GIB),
    })
    if integrated:
        profile = profile.model_copy(update={"accelerators": [AcceleratorProfile(
            name="AMD Radeon Graphics",
            vendor=AcceleratorVendor.AMD,
            type=AcceleratorType.INTEGRATED,
            shared_memory=True,
            backends=[ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=BackendAvailability.AVAILABLE,
            )],
        )]})
    return profile


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("use_case", [UseCase.CODING, UseCase.GENERAL_CHAT])
def test_small_laptop_inspects_small_models_even_when_many_7b_models_look_plausible(
    integrated: bool, use_case: UseCase,
) -> None:
    profile = _laptop(integrated)
    requirements = build_requirements(
        answers(use_case=use_case, priority=RecommendationPriority.QUALITY), profile,
    )
    small = [
        candidate(f"org/Assistant-{size}B-Instruct-GGUF", tags=["gguf", "instruct"],
                  downloads=100_000)
        for size in (0.5, 1.5, 3)
    ]
    large = [
        candidate(f"org/AssistantVersion{i}-7B-Instruct-GGUF", tags=["gguf", "instruct"],
                  downloads=1_000_000 + i)
        for i in range(policies.MAX_DEEP_INSPECTION)
    ]
    pool = [*small, *large]
    picked = shortlist(pool, requirements, policies.MAX_DEEP_INSPECTION, hardware=profile)
    names = [item.repo_id for item in picked]

    assert {item.repo_id for item in small} <= set(names)  # 0.5B, 1.5B and 3B
    assert any(item.repo_id in names for item in large)
    assert len(picked) == policies.MAX_DEEP_INSPECTION
    assert names == [item.repo_id for item in shortlist(
        list(reversed(pool)), requirements, policies.MAX_DEEP_INSPECTION, hardware=profile,
    )]
    # Small coverage must not displace the only plausible larger candidate when
    # a caller explicitly restricts inspection to a single slot.
    only = shortlist(pool, requirements, 1, hardware=profile)
    assert only[0].repo_id in {item.repo_id for item in large}


@pytest.mark.parametrize("integrated", [False, True])
def test_laptop_size_queries_cover_small_ranges_without_unbounding_search(
    integrated: bool,
) -> None:
    profile = _laptop(integrated)
    requirements = build_requirements(answers(priority=RecommendationPriority.QUALITY), profile)
    ceiling = largest_fitting_billions(profile, requirements)
    assert ceiling is not None
    queries = build_queries(requirements, max_billions=ceiling)
    named = [q for q in queries if q.max_billions is not None and q.search is not None]
    open_ = [q for q in queries if q.max_billions is not None and q.search is None]

    assert [(q.min_billions, q.max_billions) for q in named] == [
        (0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, ceiling),
    ]
    assert [(q.min_billions, q.max_billions) for q in open_] == [
        (q.min_billions, q.max_billions) for q in named
    ]
    assert all(q.limit == policies.SEARCH_RESULTS_PER_QUERY for q in queries)
    assert len({q.cache_key() for q in queries}) == len(queries)


def test_the_saved_24_gib_chat_shortlist_explores_large_groups_before_filling() -> None:
    expected = [
        "ornith-ai/Ornith-1.5-9B-GGUF",
        "unsloth/Qwen3.8-27B-GGUF",
        "ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF",
        "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "tencent/Hy-MT2-7B-GGUF",
        "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q8_0",
        "openthaigpt/openthaigpt1.5-7b-instruct",
        "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF",
    ]
    # Even extremely popular small candidates must not consume the B60's
    # existing inspection slots for larger models.
    pool = [*_saved_chat_candidates(), *[
        candidate(f"org/Chat-{size}B-Instruct-GGUF", downloads=100_000_000,
                  tags=["gguf", "chat"])
        for size in (0.5, 1.5, 3)
    ]]
    assert _chat_shortlist(pool=pool) == expected  # legacy GPU profile

    profile = hardware(ram_gib=16, vram_gib=None).model_copy(update={
        "accelerators": [*_laptop(True).accelerators, AcceleratorProfile(
            name="Intel Arc Pro B60",
            vendor=AcceleratorVendor.INTEL,
            type=AcceleratorType.DEDICATED,
            dedicated_memory_bytes=24 * GIB,
            available_memory_bytes=23 * GIB,
            backends=[ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=BackendAvailability.AVAILABLE,
            )],
        )],
    })
    requirements = build_requirements(answers(use_case=UseCase.GENERAL_CHAT), profile)
    picked = shortlist(
        pool, requirements, policies.MAX_DEEP_INSPECTION, hardware=profile,
        targets_other_task=lambda item: targets_another_task(item, requirements),
    )
    assert [item.repo_id for item in picked] == expected
    ceiling = largest_fitting_billions(profile, requirements)
    assert ceiling is not None and ceiling > 14
    assert all(low >= 4 for low, _high in size_bands(ceiling))


@pytest.mark.parametrize(("ceiling", "expected"), [
    (0.5, [(0.0, 0.5)]),
    (1.0, [(0.0, 1.0)]),
    (2.0, [(0.0, 1.0), (1.0, 2.0)]),
    (3.0, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]),
    (4.0, [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]),
    (14.0, [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 14.0)]),
    (32.0, [(4.0, 14.0), (14.0, 32.0)]),
])
def test_size_bands_stay_within_the_planning_ceiling(
    ceiling: float, expected: list[tuple[float, float]],
) -> None:
    assert size_bands(ceiling) == expected


def test_small_alternatives_reach_results_when_coarse_7b_estimates_were_optimistic() -> None:
    profile = _laptop(True)
    large = [f"org/CoderVersion{i}-7B-Instruct-GGUF"
             for i in range(policies.MAX_DEEP_INSPECTION)]
    small = ["org/MiniCoder-0.5B-Instruct-GGUF", "org/SmallCoder-1.5B-Instruct-GGUF",
             "org/MediumCoder-3B-Instruct-GGUF"]
    search = FakeSearchClient(default=[
        candidate(repo, tags=["gguf", "coder", "instruct"], downloads=1_000_000 - index)
        for index, repo in enumerate([*large, *small])
    ])
    analyses = {
        repo: gguf_analysis(
            repo, quantizations=("Q4_K_M",),
            base_bytes=9 * GIB if repo in large[1:] else (4 * GIB if repo in large else GIB),
        )
        for repo in [*large, *small]
    }
    state = orchestrator.run_workflow(
        answers(priority=RecommendationPriority.QUALITY), profile,
        _container(search, analyses=analyses, vram_budget=int(7.5 * GIB), profile=profile),
    )

    assert state.current_step is WorkflowStep.COMPLETED
    assert set(small) <= {item.repo_id for item in state.evaluated_candidates}
    assert set(small) <= {item.repo_id for item in state.recommendations}
    assert large[0] in {item.repo_id for item in state.recommendations}
    assert len(state.evaluated_candidates) <= policies.MAX_DEEP_INSPECTION
