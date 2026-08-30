"""The structured hardware fit survives to the public estimate contract.

``CompatibilityAssessment`` is a summary: one status, one ratio, one effective
device. Everything the analyzer decided about *placement* — the mode, the
transformer-block split, the per-pool byte breakdown — used to stop at that
boundary, so no consumer above the estimator could see it.

These tests pin the two halves of the fix: the analyzer's result reaches
``MemoryEstimate`` unchanged, and the summary that consumers already depend on
is not altered by carrying it. Placement coverage comes from the scenario
catalogue in ``_hardware_fit_scenarios`` rather than from new fixtures, so a
mode that catalogue stops covering also stops being covered here.
"""

from __future__ import annotations

import pytest

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import (
    HardwareFitMode,
    HardwareFitPlacementMethod,
    HardwareFitResult,
    MemoryEstimate,
)
from jaull.domain.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import (
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.domain.requirements import RecommendationPriority
from jaull.estimator import service
from jaull.estimator.compatibility import assess_components, assess_components_with_fit
from jaull.reporting.estimation import estimate_to_json_dict
from jaull.workflow.ranking import recommend
from jaull.workflow.requirements import build_requirements
from tests._hardware_fit_scenarios import (
    OBSERVED_FIELDS,
    SCENARIOS,
    Scenario,
)
from tests._workflow_fixtures import answers, hardware

GIB = 1024**3


def _scenario(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise AssertionError(f"unknown scenario {name!r}")


def _assess(scenario: Scenario, *, device_target: TargetDevice = TargetDevice.AUTO):
    total = (
        scenario.weights_bytes
        + scenario.kv_cache_bytes
        + scenario.overhead_bytes
        + scenario.device_reserve_bytes
        + scenario.safety_margin_bytes
    )
    return assess_components_with_fit(
        weights_bytes=scenario.weights_bytes,
        kv_cache_bytes=scenario.kv_cache_bytes,
        overhead_bytes=scenario.overhead_bytes,
        device_reserve_bytes=scenario.device_reserve_bytes,
        safety_margin_bytes=scenario.safety_margin_bytes,
        total_bytes=total,
        total_transformer_blocks=scenario.total_transformer_blocks,
        hardware=scenario.machine.profile(),
        device_target=device_target,
    )


# ---------------------------------------------------------------------------
# The fit reaches the caller, and the summary is unchanged by that
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_every_scenario_returns_the_analyzer_result_unchanged(
    scenario: Scenario,
) -> None:
    """What the caller receives must be what the analyzer produced, field for field."""

    expected = scenario.analyze()
    fit = _assess(scenario).fit

    assert fit is not None
    for field in OBSERVED_FIELDS:
        assert getattr(fit, field) == getattr(expected, field), field
    assert fit.reason == expected.reason
    assert fit.warnings == expected.warnings


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_returning_the_fit_does_not_change_the_summary(scenario: Scenario) -> None:
    """The existing single-value API must keep answering exactly as before."""

    total = (
        scenario.weights_bytes
        + scenario.kv_cache_bytes
        + scenario.overhead_bytes
        + scenario.device_reserve_bytes
        + scenario.safety_margin_bytes
    )
    legacy = assess_components(
        weights_bytes=scenario.weights_bytes,
        kv_cache_bytes=scenario.kv_cache_bytes,
        overhead_bytes=scenario.overhead_bytes,
        device_reserve_bytes=scenario.device_reserve_bytes,
        safety_margin_bytes=scenario.safety_margin_bytes,
        total_bytes=total,
        total_transformer_blocks=scenario.total_transformer_blocks,
        hardware=scenario.machine.profile(),
        device_target=TargetDevice.AUTO,
    )

    assert legacy == _assess(scenario).assessment


# ---------------------------------------------------------------------------
# One test per mode, on the detail that mode exists to express
# ---------------------------------------------------------------------------
def test_gpu_resident_keeps_its_mode_and_full_transformer_block_placement() -> None:
    fit = _assess(_scenario("gpu_resident_comfortable")).fit

    assert fit is not None
    assert fit.mode is HardwareFitMode.GPU_RESIDENT
    assert fit.placement_method is HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
    assert fit.gpu_transformer_blocks == fit.total_transformer_blocks
    assert fit.ram_weight_bytes == 0
    assert fit.places_weights_on_gpu


def test_gpu_offload_keeps_transformer_block_split_and_both_pools() -> None:
    """The offload count is model transformer blocks, not runtime units."""

    scenario = _scenario("gpu_offload_transformer_block_split")
    fit = _assess(scenario).fit

    assert fit is not None
    assert fit.mode is HardwareFitMode.GPU_OFFLOAD
    assert fit.placement_method is HardwareFitPlacementMethod.TRANSFORMER_BLOCKS
    assert fit.gpu_transformer_blocks is not None
    assert fit.total_transformer_blocks is not None
    assert 0 < fit.gpu_transformer_blocks < fit.total_transformer_blocks
    # Both pools carry weight, and together they carry all of it.
    assert fit.gpu_weight_bytes > 0
    assert fit.ram_weight_bytes > 0
    assert fit.gpu_weight_bytes + fit.ram_weight_bytes == scenario.weights_bytes


def test_byte_placement_survives_without_a_transformer_block_count() -> None:
    fit = _assess(_scenario("byte_fallback_without_transformer_block_metadata")).fit

    assert fit is not None
    assert fit.placement_method is HardwareFitPlacementMethod.ESTIMATED_BYTES
    assert fit.gpu_transformer_blocks is None


def test_cpu_ram_keeps_the_mode_and_places_no_weights_on_the_gpu() -> None:
    fit = _assess(_scenario("cpu_ram_when_no_gpu_placement_is_viable")).fit

    assert fit is not None
    assert fit.mode is HardwareFitMode.CPU_RAM
    assert fit.gpu_weight_bytes == 0
    assert not fit.places_weights_on_gpu


def test_too_large_survives_as_a_placement_result() -> None:
    """An impossible placement is still a placement answer, not a missing one."""

    fit = _assess(_scenario("too_large_for_both_pools")).fit

    assert fit is not None
    assert fit.mode is HardwareFitMode.TOO_LARGE
    assert not fit.places_weights_on_gpu
    assert fit.reason


def test_no_gpu_reports_transformer_blocks_as_not_applicable() -> None:
    fit = _assess(_scenario("no_gpu_too_large")).fit

    assert fit is not None
    assert fit.gpu_transformer_blocks is None
    assert fit.available_vram_bytes is None


# ---------------------------------------------------------------------------
# When there is deliberately no fit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", [TargetDevice.CPU, TargetDevice.GPU])
def test_a_pinned_device_target_produces_no_fit(device: TargetDevice) -> None:
    """The analyzer only runs for AUTO; pinned targets keep the older path."""

    result = _assess(_scenario("gpu_resident_comfortable"), device_target=device)

    assert result.fit is None
    assert result.assessment.target_device is device


def test_a_missing_memory_component_produces_no_fit() -> None:
    result = assess_components_with_fit(
        weights_bytes=None,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=0,
        safety_margin_bytes=0,
        total_bytes=None,
        total_transformer_blocks=32,
        hardware=_scenario("gpu_resident_comfortable").machine.profile(),
        device_target=TargetDevice.AUTO,
    )

    assert result.fit is None


# ---------------------------------------------------------------------------
# Budget vs allocation: the distinction the VRAM comparison depends on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_physical_bytes_are_the_budget_minus_policy(scenario: Scenario) -> None:
    """Reserve and margin are capacity policy, never a process allocation."""

    fit = scenario.analyze()

    if fit.places_weights_on_gpu:
        assert fit.gpu_required_bytes is not None
        assert fit.gpu_physical_bytes == (
            fit.gpu_required_bytes
            - fit.device_reserve_bytes
            - fit.gpu_safety_margin_bytes
        )
    else:
        assert fit.gpu_physical_bytes is None

    if fit.ram_required_bytes is not None:
        assert fit.ram_physical_bytes is not None
        assert fit.ram_physical_bytes <= fit.ram_required_bytes


def test_gpu_physical_bytes_is_weights_plus_kv_plus_overhead_when_resident() -> None:
    """Spelled out against the components, not just against the subtraction."""

    scenario = _scenario("gpu_resident_comfortable")
    fit = scenario.analyze()

    assert fit.gpu_physical_bytes == (
        scenario.weights_bytes + scenario.kv_cache_bytes + scenario.overhead_bytes
    )


def test_a_rejected_gpu_reports_no_physical_bytes_despite_a_required_figure() -> None:
    """``gpu_required_bytes`` on CPU_RAM is hypothetical, not a prediction.

    It exists to explain why the GPU was rejected. Treating it as an allocation
    forecast would compare a number no process will ever allocate.
    """

    fit = _scenario("cpu_ram_when_no_gpu_placement_is_viable").analyze()

    assert fit.gpu_required_bytes is not None
    assert fit.gpu_physical_bytes is None


# ---------------------------------------------------------------------------
# End to end through the estimator, and out into the JSON report
# ---------------------------------------------------------------------------
def _estimate(*, vram: int | None, context: int = 4096) -> MemoryEstimate:
    """A real estimate through ``estimator.service``, not a hand-built model.

    A Transformers repository with a populated config is what the fit needs: a
    GGUF repo without one produces no KV estimate, so the analyzer correctly
    never runs and there is no placement to propagate.
    """

    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="user/tf"),
        files=[
            ModelFile(path="config.json", size_bytes=1024),
            ModelFile(path="model.safetensors", size_bytes=4 * GIB),
        ],
        classification=RepositoryClassification(
            primary_type=RepositoryType.TRANSFORMERS,
            detected_types={RepositoryType.TRANSFORMERS},
            formats={Format.SAFETENSORS},
            gguf_variants=[],
        ),
        config=ModelConfig(
            model_type="llama",
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        ),
        relevant_files=["config.json", "model.safetensors"],
        total_size_bytes=4 * GIB,
        warnings=[],
    )
    gpus = (
        [
            GpuInfo(
                name="Test GPU",
                vram_total_bytes=vram,
                vram_available_bytes=vram,
                driver_version="1.0",
                cuda_version="12.0",
            )
        ]
        if vram is not None
        else []
    )
    profile = HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=32 * GIB),
        storage=[],
        gpus=gpus,
        warnings=[],
    )

    class _Client:
        def model_info(self, repo_id: str):  # pragma: no cover - unused here
            raise NotImplementedError

        def download_small_file(self, repo_id: str, filename: str):  # pragma: no cover
            raise NotImplementedError

        def safetensors_summary(self, repo_id: str):
            return None

    return service.estimate_memory(
        analysis=analysis,
        hardware=profile,
        inference_cfg=InferenceConfiguration(
            context_length=context,
            target_device=TargetDevice.AUTO,
        ),
        client=_Client(),
        resolve_base_model=False,
    )


def test_the_estimator_attaches_the_fit_to_the_estimate() -> None:
    """The whole point: a real estimate carries its placement."""

    estimate = _estimate(vram=24 * GIB)

    assert estimate.hardware_fit is not None
    assert estimate.hardware_fit.mode is HardwareFitMode.GPU_RESIDENT
    assert estimate.hardware_fit.gpu_physical_bytes is not None


def test_a_cpu_only_machine_still_carries_its_placement() -> None:
    estimate = _estimate(vram=None)

    assert estimate.hardware_fit is not None
    assert estimate.hardware_fit.mode is HardwareFitMode.CPU_RAM
    assert estimate.hardware_fit.available_vram_bytes is None


def test_reporting_json_carries_the_placement() -> None:
    payload = estimate_to_json_dict(_estimate(vram=24 * GIB))

    fit = payload["hardware_fit"]
    assert fit is not None
    assert fit["mode"] == "gpu_resident"
    assert fit["placement_method"] in {"transformer_blocks", "estimated_bytes"}
    assert fit["gpu_transformer_blocks"] == 32
    assert fit["total_transformer_blocks"] == 32
    assert "gpu_layers" not in fit
    assert "total_layers" not in fit
    assert fit["gpu_required_bytes"] > fit["gpu_physical_bytes"]
    assert fit["ram_required_bytes"] == 0
    assert fit["offload_diagnostics"] is None
    assert isinstance(fit["reason"], str)


def test_reporting_json_carries_the_offload_decision_boundary() -> None:
    fit = _scenario("gpu_offload_transformer_block_split").analyze()
    estimate = _estimate(vram=24 * GIB).model_copy(update={"hardware_fit": fit})

    payload = estimate_to_json_dict(estimate)

    diagnostics = payload["hardware_fit"]["offload_diagnostics"]
    assert diagnostics is not None
    assert diagnostics["search_ceiling_transformer_blocks"] >= (
        fit.gpu_transformer_blocks
    )
    selected = diagnostics["selected"]
    rejected = diagnostics["first_rejected_higher"]
    assert selected["gpu_transformer_blocks"] == fit.gpu_transformer_blocks
    assert selected["gpu_required_bytes"] == fit.gpu_required_bytes
    assert selected["headroom_bytes"] == (
        selected["available_vram_bytes"] - selected["gpu_required_bytes"]
    )
    assert selected["excess_bytes"] == 0
    assert rejected["gpu_transformer_blocks"] == fit.gpu_transformer_blocks + 1
    assert rejected["gpu_required_bytes"] > rejected["available_vram_bytes"]
    assert rejected["headroom_bytes"] == 0
    assert rejected["excess_bytes"] == (
        rejected["gpu_required_bytes"] - rejected["available_vram_bytes"]
    )
    for candidate in (selected, rejected):
        assert candidate["gpu_required_bytes"] == (
            candidate["gpu_weight_bytes"]
            + candidate["gpu_kv_cache_bytes"]
            + candidate["device_reserve_bytes"]
            + candidate["gpu_overhead_bytes"]
            + candidate["gpu_safety_margin_bytes"]
        )
        assert candidate["ram_required_bytes"] == (
            candidate["ram_weight_bytes"]
            + candidate["ram_kv_cache_bytes"]
            + candidate["ram_overhead_bytes"]
            + candidate["ram_safety_margin_bytes"]
        )
        # The total is carried alongside the two shares so the split is
        # auditable from the payload rather than taken on trust.
        assert candidate["gpu_kv_cache_bytes"] + candidate["ram_kv_cache_bytes"] == (
            candidate["kv_cache_bytes"]
        )


def test_legacy_hardware_fit_layer_fields_still_load() -> None:
    fit = HardwareFitResult.model_validate(
        {
            "mode": "gpu_offload",
            "memory_topology": "discrete_memory",
            "weights_bytes": 100,
            "kv_cache_bytes": 10,
            "overhead_bytes": 5,
            "gpu_layers": 18,
            "total_layers": 28,
            "placement_method": "layers",
            "reason": "legacy",
        }
    )

    assert fit.gpu_transformer_blocks == 18
    assert fit.total_transformer_blocks == 28
    assert fit.placement_method is HardwareFitPlacementMethod.TRANSFORMER_BLOCKS


def test_reporting_json_says_null_when_there_is_no_placement() -> None:
    """An explicit null distinguishes "no fit" from "this report is older"."""

    estimate = _estimate(vram=24 * GIB).model_copy(
        update={"hardware_fit": None}
    )

    assert estimate_to_json_dict(estimate)["hardware_fit"] is None


def test_reporting_keeps_the_compatibility_summary_alongside_the_fit() -> None:
    payload = estimate_to_json_dict(_estimate(vram=24 * GIB))

    assert payload["assessment"]["status"]
    assert payload["assessment"]["effective_device"]
    assert payload["hardware_fit"]["mode"]


# ---------------------------------------------------------------------------
# Ranking must not notice
# ---------------------------------------------------------------------------
def test_ranking_is_identical_with_and_without_a_fit() -> None:
    """Carrying placement detail must not move a single ranking number.

    This milestone propagates data; consuming it in the ranker is a separate
    decision, and this test is what makes that decision explicit rather than
    accidental.
    """

    from tests.test_recommendation_ranking import _evaluated

    without = _evaluated()
    fit = _scenario("gpu_offload_transformer_block_split").analyze()
    with_fit = without.model_copy(
        update={
            "memory_estimate": without.memory_estimate.model_copy(
                update={"hardware_fit": fit}
            )
        }
    )
    requirements = build_requirements(
        answers(priority=RecommendationPriority.BALANCED), hardware()
    )

    baseline = recommend([without], requirements)
    enriched = recommend([with_fit], requirements)

    assert len(baseline) == len(enriched) == 1
    assert baseline[0].score == enriched[0].score
    assert baseline[0].evaluated.hardware_fit_score == (
        enriched[0].evaluated.hardware_fit_score
    )
    assert baseline[0].tier == enriched[0].tier


def test_the_fit_is_not_read_by_the_compatibility_summary() -> None:
    """A fit attached by hand must not alter the summary already computed."""

    estimate = _estimate(vram=24 * GIB)
    swapped: HardwareFitResult = _scenario("too_large_for_both_pools").analyze()

    mutated = estimate.model_copy(update={"hardware_fit": swapped})

    assert mutated.assessment == estimate.assessment
