from __future__ import annotations

import json
import threading
from pathlib import Path

from jaull.domain.candidates import SearchQuery
from jaull.domain.estimation import CompatibilityStatus
from jaull.exceptions import (
    HuggingFaceUnavailableError,
    ModelNotFoundError,
)
from jaull.recommendation.report import (
    REPORT_SCHEMA_VERSION,
    report_to_dict,
    report_to_json,
    report_to_markdown,
)
from jaull.workflow import orchestrator, policies
from jaull.workflow.container import ServiceContainer
from jaull.workflow.model_analysis_cache import ModelAnalysisCache
from jaull.workflow.models import StepStatus, WorkflowStep
from jaull.workflow.progress import HARDWARE_STEPS
from tests._workflow_fixtures import (
    GIB,
    FakeHfClient,
    FakeSearchClient,
    answers,
    candidate,
    gguf_analysis,
    hardware,
    size_driven_estimator,
    transformers_analysis,
)


def _container(
    search: FakeSearchClient,
    analyses: dict[str, object] | None = None,
    vram_budget: int = 24 * GIB,
    inspect_error: Exception | None = None,
    profile: object | None = None,
) -> ServiceContainer:
    estimator = size_driven_estimator(vram_budget=vram_budget)
    store = analyses or {}

    def inspect_model(repo_id: str, client: object = None) -> object:
        if inspect_error is not None and repo_id == "org/broken":
            raise inspect_error
        return store.get(repo_id, transformers_analysis(repo_id=repo_id))

    def estimate_memory(**kwargs: object) -> object:
        return estimator(kwargs["analysis"], kwargs["inference_cfg"])

    def detect(**kwargs: object) -> object:
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for key, _ in HARDWARE_STEPS:
                on_step(key)
        return profile if profile is not None else hardware()

    return ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=estimate_memory,  # type: ignore[arg-type]
        range_client_factory=None,
    )


def _search_with(*repo_ids: str) -> FakeSearchClient:
    return FakeSearchClient(
        default=[
            candidate(repo_id=repo_id, tags=["text-generation", "code"])
            for repo_id in repo_ids
        ]
    )


# ---------------------------------------------------------------------------
# Hardware step
# ---------------------------------------------------------------------------
def test_hardware_scan_reports_every_step_as_done() -> None:
    snapshots = []
    services = _container(FakeSearchClient())
    profile = orchestrator.scan_hardware(services, on_progress=snapshots.append)

    assert profile.cpu.model
    final = snapshots[-1]
    assert all(step.status is StepStatus.DONE for step in final.steps)
    assert len(final.steps) == len(HARDWARE_STEPS)


def test_hardware_scan_without_a_gpu_still_succeeds() -> None:
    """A missing NVIDIA GPU is a warning, never a fatal error."""
    services = _container(FakeSearchClient(), profile=hardware(vram_gib=None))
    profile = orchestrator.scan_hardware(services)
    assert profile.gpus == []
    assert profile.warnings


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_full_successful_run() -> None:
    services = _container(_search_with("org/Coder-7B", "org/Coder-13B"))
    state = orchestrator.run_workflow(answers(), hardware(), services)

    assert state.current_step is WorkflowStep.COMPLETED
    assert state.recommendations
    assert state.primary is not None
    assert state.requirements is not None
    assert state.search_queries
    assert len(state.recommendations) <= policies.MAX_RECOMMENDATIONS


def test_run_without_a_gpu_still_produces_recommendations() -> None:
    profile = hardware(vram_gib=None)
    services = _container(_search_with("org/Coder-7B"), vram_budget=32 * GIB)
    state = orchestrator.run_workflow(answers(), profile, services)
    assert state.current_step is WorkflowStep.COMPLETED
    assert state.warnings  # the "no NVIDIA GPU" warning is carried through


def test_gguf_repositories_are_evaluated_through_the_ladder() -> None:
    services = _container(
        _search_with("org/Coder-GGUF"),
        analyses={"org/Coder-GGUF": gguf_analysis(repo_id="org/Coder-GGUF")},
    )
    state = orchestrator.run_workflow(answers(), hardware(), services)
    assert state.recommendations
    config = state.recommendations[0].evaluated.selected_configuration
    assert config is not None
    assert config.quantization


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def test_one_broken_candidate_does_not_end_the_run() -> None:
    services = _container(
        _search_with("org/broken", "org/Coder-7B"),
        inspect_error=ModelNotFoundError("gone"),
    )
    state = orchestrator.run_workflow(answers(), hardware(), services)

    assert state.current_step is WorkflowStep.COMPLETED
    assert state.recommendations
    assert any(item.failed for item in state.evaluated_candidates)
    assert "org/broken" not in {r.repo_id for r in state.recommendations}


def test_total_hugging_face_failure_is_recoverable_not_fatal() -> None:
    search = FakeSearchClient(raises=HuggingFaceUnavailableError("hub down"))
    state = orchestrator.run_workflow(answers(), hardware(), _container(search))

    assert state.current_step is WorkflowStep.FAILED
    assert state.errors
    assert "hub down" in state.errors[0]


def test_a_single_failing_query_does_not_abort_the_search() -> None:
    search = FakeSearchClient(
        default=[candidate(repo_id="org/Coder-7B", tags=["text-generation", "code"])],
        raises=HuggingFaceUnavailableError("one query failed"),
        raise_on={"coding:coder instruct"},
    )
    state = orchestrator.run_workflow(answers(), hardware(), _container(search))
    assert state.current_step is WorkflowStep.COMPLETED
    assert any("failed" in w for w in state.warnings)


def test_no_candidates_explains_why() -> None:
    state = orchestrator.run_workflow(
        answers(), hardware(), _container(FakeSearchClient(default=[]))
    )
    assert state.current_step is WorkflowStep.COMPLETED
    assert state.recommendations == []
    assert state.no_results_reason


def test_nothing_compatible_explains_what_was_missing() -> None:
    services = _container(_search_with("org/Giant-70B"), vram_budget=1 * GIB)
    state = orchestrator.run_workflow(answers(), hardware(), services)
    assert state.recommendations == []
    assert any("No fully compatible" in line for line in state.no_results_reason)


# ---------------------------------------------------------------------------
# Cancellation, limits and caching
# ---------------------------------------------------------------------------
def test_cancellation_stops_the_run() -> None:
    services = _container(_search_with("org/a", "org/b"))
    state = orchestrator.run_workflow(
        answers(), hardware(), services, is_cancelled=lambda: True
    )
    assert state.current_step is WorkflowStep.FAILED
    assert state.errors == ["Search cancelled."]


def test_deep_inspection_is_capped() -> None:
    inspected: list[str] = []
    many = [f"org/model-{index}" for index in range(60)]
    search = FakeSearchClient(
        default=[
            candidate(repo_id=repo, tags=["text-generation", "code"]) for repo in many
        ]
    )
    services = _container(search)

    original = services.inspect_model

    def counting(repo_id: str, client: object = None) -> object:
        inspected.append(repo_id)
        return original(repo_id, client)

    services = ServiceContainer(
        hf_client=services.hf_client,
        search_client=services.search_client,
        detect_hardware=services.detect_hardware,
        inspect_model=counting,  # type: ignore[arg-type]
        estimate_memory=services.estimate_memory,
    )

    state = orchestrator.run_workflow(answers(), hardware(), services)
    assert len(set(inspected)) <= policies.MAX_DEEP_INSPECTION
    assert len(state.candidates) <= policies.MAX_UNIQUE_CANDIDATES


def test_hardware_aware_shortlist_preserves_offload_without_extra_inspection() -> None:
    inspected: list[str] = []
    offload_repo = "Qwen/Qwen2.5-14B-Q4_K_M-GGUF"
    impossible_repo = "Huge/Huge-70B-Q8_0-GGUF"
    candidates = [
        candidate(repo_id=f"Qwen/Qwen2.5-{index}B-Q4_K_M-GGUF", tags=["gguf", "Q4_K_M"])
        for index in range(1, 12)
    ]
    candidates.extend(
        [
            candidate(repo_id=offload_repo, tags=["gguf", "Q4_K_M"]),
            candidate(repo_id=impossible_repo, tags=["gguf", "Q8_0"]),
        ]
    )
    services = _container(FakeSearchClient(default=candidates))
    original = services.inspect_model

    def counting(repo_id: str, client: object = None) -> object:
        inspected.append(repo_id)
        return original(repo_id, client)

    services = ServiceContainer(
        hf_client=services.hf_client,
        search_client=services.search_client,
        detect_hardware=services.detect_hardware,
        inspect_model=counting,  # type: ignore[arg-type]
        estimate_memory=services.estimate_memory,
    )

    orchestrator.run_workflow(answers(), hardware(vram_gib=8, ram_gib=32), services)
    inspected_unique = set(inspected)
    assert len(inspected_unique) <= policies.MAX_DEEP_INSPECTION
    assert offload_repo in inspected_unique
    assert impossible_repo not in inspected_unique


def test_the_same_repository_is_inspected_once_per_run() -> None:
    """The cache is what stops the quantization ladder re-fetching metadata."""
    calls: list[str] = []
    search = FakeSearchClient(
        default=[candidate(repo_id="org/Coder-GGUF", tags=["text-generation", "code"])]
    )
    analysis = gguf_analysis(repo_id="org/Coder-GGUF")

    def inspect_model(repo_id: str, client: object = None) -> object:
        calls.append(repo_id)
        return analysis

    services = ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=lambda **_: hardware(),  # type: ignore[arg-type]
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=lambda **kwargs: size_driven_estimator(24 * GIB)(  # type: ignore[arg-type]
            kwargs["analysis"], kwargs["inference_cfg"]
        ),
    )

    orchestrator.run_workflow(answers(), hardware(), services)
    # The ladder tries several quantizations, but inspection happens once.
    assert calls.count("org/Coder-GGUF") == 1


def test_persistent_cache_avoids_second_run_inspection(tmp_path: Path) -> None:
    calls: list[str] = []
    repo_id = "org/Coder-7B"
    search = FakeSearchClient(
        default=[
            candidate(
                repo_id=repo_id,
                tags=["text-generation", "code"],
            ).model_copy(update={"revision_hint": "rev1"})
        ]
    )

    def inspect_model(repo_id: str, client: object = None) -> object:
        del client
        calls.append(repo_id)
        return transformers_analysis(repo_id=repo_id)

    first = _container(search)
    first = ServiceContainer(
        hf_client=first.hf_client,
        search_client=first.search_client,
        detect_hardware=first.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=first.estimate_memory,
        model_analysis_cache=ModelAnalysisCache(root=tmp_path),
    )
    cold = orchestrator.run_workflow(answers(), hardware(), first)

    second = ServiceContainer(
        hf_client=first.hf_client,
        search_client=search,
        detect_hardware=first.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=first.estimate_memory,
        model_analysis_cache=ModelAnalysisCache(root=tmp_path),
    )
    warm = orchestrator.run_workflow(answers(), hardware(), second)

    assert calls == [repo_id]
    assert cold.recommendations[0].repo_id == warm.recommendations[0].repo_id
    assert cold.telemetry["count.persistent_cache_misses"] == 1
    assert warm.telemetry["count.persistent_cache_hits"] == 1
    assert warm.telemetry.get("count.deep_inspections", 0) == 0


def test_changed_repository_revision_invalidates_only_that_repo(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    repos = ["org/A-7B", "org/B-7B"]
    cold_candidates = [
        candidate(repo_id=repo, tags=["text-generation", "code"]).model_copy(
            update={"revision_hint": "rev1"}
        )
        for repo in repos
    ]
    warm_candidates = [
        cold_candidates[0],
        cold_candidates[1].model_copy(update={"revision_hint": "rev2"}),
    ]

    def inspect_model(repo_id: str, client: object = None) -> object:
        del client
        calls.append(repo_id)
        return transformers_analysis(repo_id=repo_id)

    base = _container(FakeSearchClient(default=cold_candidates))
    cold = ServiceContainer(
        hf_client=base.hf_client,
        search_client=base.search_client,
        detect_hardware=base.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=base.estimate_memory,
        model_analysis_cache=ModelAnalysisCache(root=tmp_path),
    )
    orchestrator.run_workflow(answers(), hardware(), cold)
    calls.clear()

    warm = ServiceContainer(
        hf_client=base.hf_client,
        search_client=FakeSearchClient(default=warm_candidates),
        detect_hardware=base.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=base.estimate_memory,
        model_analysis_cache=ModelAnalysisCache(root=tmp_path),
    )
    orchestrator.run_workflow(answers(), hardware(), warm)

    assert calls == ["org/B-7B"]


def test_persistent_cache_does_not_cache_estimates(tmp_path: Path) -> None:
    estimate_calls = 0
    repo_id = "org/Coder-7B"
    search = FakeSearchClient(
        default=[
            candidate(repo_id=repo_id, tags=["text-generation", "code"]).model_copy(
                update={"revision_hint": "rev1"}
            )
        ]
    )
    estimator = size_driven_estimator(vram_budget=24 * GIB)

    def estimate_memory(**kwargs: object) -> object:
        nonlocal estimate_calls
        estimate_calls += 1
        return estimator(kwargs["analysis"], kwargs["inference_cfg"])

    first = ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=lambda **_: hardware(),  # type: ignore[arg-type]
        inspect_model=lambda repo_id, client=None: transformers_analysis(repo_id),  # type: ignore[arg-type]
        estimate_memory=estimate_memory,  # type: ignore[arg-type]
        model_analysis_cache=ModelAnalysisCache(root=tmp_path),
    )
    orchestrator.run_workflow(answers(), hardware(), first)
    cold_estimates = estimate_calls

    second = ServiceContainer(
        hf_client=first.hf_client,
        search_client=search,
        detect_hardware=first.detect_hardware,
        inspect_model=first.inspect_model,
        estimate_memory=estimate_memory,  # type: ignore[arg-type]
        model_analysis_cache=ModelAnalysisCache(root=tmp_path),
    )
    orchestrator.run_workflow(answers(), hardware(), second)

    assert estimate_calls > cold_estimates


def test_duplicate_search_results_are_inspected_once() -> None:
    repeated = [
        candidate(repo_id="org/same", tags=["text-generation", "code"], queries=[f"q{i}"])
        for i in range(5)
    ]
    search = FakeSearchClient(default=repeated)
    state = orchestrator.run_workflow(answers(), hardware(), _container(search))
    assert len(state.candidates) == 1


def test_deep_inspection_uses_bounded_concurrency() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    release = threading.Event()
    reached_limit = threading.Event()
    candidates = [
        candidate(repo_id=f"org/model-{index}", tags=["text-generation", "code"])
        for index in range(policies.MAX_DEEP_INSPECTION)
    ]

    def inspect_model(repo_id: str, client: object = None) -> object:
        nonlocal active, max_active
        del client
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == policies.MAX_CONCURRENT_INSPECTIONS:
                reached_limit.set()
        reached_limit.wait(timeout=1)
        release.set()
        release.wait(timeout=1)
        with lock:
            active -= 1
        return transformers_analysis(repo_id=repo_id)

    base = _container(FakeSearchClient(default=candidates))
    services = ServiceContainer(
        hf_client=base.hf_client,
        search_client=base.search_client,
        detect_hardware=base.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=base.estimate_memory,
    )

    orchestrator.run_workflow(answers(), hardware(), services)

    assert max_active == policies.MAX_CONCURRENT_INSPECTIONS


def test_slow_candidate_does_not_reorder_evaluated_results() -> None:
    release_first = threading.Event()
    second_started = threading.Event()
    candidates = [
        candidate(
            repo_id="org/slow",
            tags=["text-generation", "code"],
            downloads=20_000,
        ),
        candidate(
            repo_id="org/fast",
            tags=["text-generation", "code"],
            downloads=100,
        ),
    ]

    def inspect_model(repo_id: str, client: object = None) -> object:
        del client
        if repo_id == "org/slow":
            second_started.wait(timeout=1)
            release_first.wait(timeout=1)
        else:
            second_started.set()
            release_first.set()
        return transformers_analysis(repo_id=repo_id)

    base = _container(FakeSearchClient(default=candidates))
    services = ServiceContainer(
        hf_client=base.hf_client,
        search_client=base.search_client,
        detect_hardware=base.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=base.estimate_memory,
    )

    state = orchestrator.run_workflow(answers(), hardware(), services)

    assert [item.repo_id for item in state.evaluated_candidates] == [
        "org/slow",
        "org/fast",
    ]


def test_cancellation_cancels_pending_inspections() -> None:
    started = 0
    cancel_now = threading.Event()
    release = threading.Event()
    candidates = [
        candidate(repo_id=f"org/model-{index}", tags=["text-generation", "code"])
        for index in range(policies.MAX_DEEP_INSPECTION)
    ]

    def inspect_model(repo_id: str, client: object = None) -> object:
        nonlocal started
        del repo_id, client
        started += 1
        cancel_now.set()
        release.wait(timeout=1)
        return transformers_analysis()

    base = _container(FakeSearchClient(default=candidates))
    services = ServiceContainer(
        hf_client=base.hf_client,
        search_client=base.search_client,
        detect_hardware=base.detect_hardware,
        inspect_model=inspect_model,  # type: ignore[arg-type]
        estimate_memory=base.estimate_memory,
    )

    state = orchestrator.run_workflow(
        answers(),
        hardware(),
        services,
        is_cancelled=cancel_now.is_set,
    )
    release.set()

    assert state.current_step is WorkflowStep.FAILED
    assert started <= policies.MAX_CONCURRENT_INSPECTIONS


def test_every_query_contributes_to_the_candidate_pool() -> None:
    """The format and language queries must not be starved by the first ones.

    Concatenating results would let the earliest queries fill the whole budget,
    which is exactly how GGUF builds went missing from a real run.
    """
    plentiful = [
        candidate(repo_id=f"org/generic-{index}", tags=["text-generation", "code"])
        for index in range(policies.MAX_UNIQUE_CANDIDATES * 2)
    ]
    search = FakeSearchClient(
        default=plentiful,
        results={
            "coding:gguf": [
                candidate(repo_id="org/Coder-GGUF", tags=["text-generation", "gguf"])
            ],
            "coding:lang-es": [
                candidate(repo_id="org/Coder-ES", tags=["text-generation", "code"])
            ],
        },
    )
    state = orchestrator.run_workflow(answers(), hardware(), _container(search))

    found = {c.repo_id for c in state.candidates}
    assert "org/Coder-GGUF" in found
    assert "org/Coder-ES" in found
    assert len(state.candidates) <= policies.MAX_UNIQUE_CANDIDATES


def test_all_queries_are_issued_even_when_early_ones_are_plentiful() -> None:
    search = FakeSearchClient(
        default=[
            candidate(repo_id=f"org/m-{index}", tags=["text-generation", "code"])
            for index in range(policies.SEARCH_RESULTS_PER_QUERY)
        ]
    )
    orchestrator.run_workflow(answers(), hardware(), _container(search))
    labels = {query.label for query in search.seen}
    assert "coding:gguf" in labels
    assert "coding:trending" in labels


def test_restart_produces_an_independent_run() -> None:
    services = _container(_search_with("org/Coder-7B"))
    first = orchestrator.run_workflow(answers(), hardware(), services)
    second = orchestrator.run_workflow(answers(), hardware(), services)
    assert first.recommendations[0].repo_id == second.recommendations[0].repo_id
    assert first is not second


def test_progress_is_reported_for_every_discovery_step() -> None:
    snapshots = []
    services = _container(_search_with("org/Coder-7B"))
    orchestrator.run_workflow(
        answers(), hardware(), services, on_progress=snapshots.append
    )
    assert snapshots
    keys = {step.key for step in snapshots[-1].steps}
    assert {"queries", "search", "filter", "inspect", "rank"} <= keys


def test_completed_search_progress_does_not_keep_last_query_detail() -> None:
    snapshots = []
    services = _container(_search_with("org/Coder-7B"))
    orchestrator.run_workflow(
        answers(), hardware(), services, on_progress=snapshots.append
    )

    final_search = next(step for step in snapshots[-1].steps if step.key == "search")

    assert final_search.status is StepStatus.DONE
    assert final_search.detail == "1 unique repositories"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_json_export_is_stable_and_complete() -> None:
    services = _container(_search_with("org/Coder-7B"))
    state = orchestrator.run_workflow(answers(), hardware(), services)

    payload = json.loads(report_to_json(state))
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    for key in (
        "timestamp",
        "hardware",
        "user_requirements",
        "search_strategy",
        "evaluated_candidates",
        "recommendations",
        "warnings",
        "assumptions",
    ):
        assert key in payload
    assert payload["recommendations"][0]["reasons"]
    assert payload["recommendations"][0]["score_breakdown"]["weights"]


def test_export_contains_no_credentials() -> None:
    services = _container(_search_with("org/Coder-7B"))
    state = orchestrator.run_workflow(answers(), hardware(), services)
    text = report_to_json(state).lower()
    for forbidden in ("hf_token", "authorization", "bearer", "api_key"):
        assert forbidden not in text


def test_markdown_export_renders_the_primary_recommendation() -> None:
    services = _container(_search_with("org/Coder-7B"))
    state = orchestrator.run_workflow(answers(), hardware(), services)
    markdown = report_to_markdown(state)
    assert "# jaull recommendation report" in markdown
    # The primary heading is now tier-driven; one of the four tier labels must
    # be present, but "Best match" is only used when the estimate is HIGH
    # confidence and comfortable — which is not guaranteed for every fixture.
    assert any(
        heading in markdown
        for heading in ("Best Match", "Recommended", "Closest Option", "Best Effort")
    )
    assert "not legal advice" in markdown


def test_report_of_an_empty_run_still_serialises() -> None:
    state = orchestrator.run_workflow(
        answers(), hardware(), _container(FakeSearchClient(default=[]))
    )
    payload = report_to_dict(state)
    assert payload["recommendations"] == []
    assert payload["no_results_reason"]
    assert "No recommendations" in report_to_markdown(state) or state.no_results_reason


def test_search_queries_are_recorded_for_the_report() -> None:
    search = _search_with("org/Coder-7B")
    state = orchestrator.run_workflow(answers(), hardware(), _container(search))
    assert state.search_queries
    assert all(isinstance(query, SearchQuery) for query in search.seen)


def test_insufficient_candidates_never_reach_the_recommendations() -> None:
    services = _container(_search_with("org/Big-70B"), vram_budget=2 * GIB)
    state = orchestrator.run_workflow(answers(), hardware(), services)
    for rec in state.recommendations:
        assert rec.status is not CompatibilityStatus.INSUFFICIENT
