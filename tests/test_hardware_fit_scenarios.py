"""A controlled battery over the hardware fit analyzer.

``tests/test_hardware_fit_analyzer.py`` covers the analyzer branch by branch.
This file is the other half: one catalogue of deterministic scenarios
(``tests/_hardware_fit_scenarios.py``), every one of them checked against the
same set of invariants, so a placement change shows up everywhere it applies
rather than only where somebody wrote an assertion.

Three kinds of assertion live here:

* **Per-scenario expectations** — the mode and placement method each fixture
  was designed to produce.
* **Conservation invariants** — weights, overhead and safety margin are split
  between the pools, never created or lost. These hold for *every* scenario in
  every mode, so they are the strongest statement the battery makes.
* **Controlled comparisons** — pairs that differ in exactly one input, which
  is asserted rather than assumed, so the mode change they demonstrate can
  only be attributed to that input.

The recorded snapshot pins every observable field so the numbers behind the
comparison with llmfit cannot drift silently. Regenerate it deliberately with
``uv run python scripts/hardware_fit_matrix.py --write-snapshot``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaull.domain.estimation import (
    HardwareFitMode,
    HardwareFitPlacementMethod,
    HardwareMemoryTopology,
)
from tests._hardware_fit_scenarios import (
    CONTROLLED_PAIRS,
    OBSERVED_FIELDS,
    REQUIRED_COVERAGE,
    SCENARIOS,
    Scenario,
    scenario,
)

SNAPSHOT = Path(__file__).resolve().parent / "snapshots" / "hardware_fit_scenarios.json"

_PLACED_MODES = (
    HardwareFitMode.GPU_RESIDENT,
    HardwareFitMode.GPU_OFFLOAD,
    HardwareFitMode.CPU_RAM,
)

_ALL = pytest.mark.parametrize(
    "case", SCENARIOS, ids=[item.name for item in SCENARIOS]
)


# ---------------------------------------------------------------------------
# Per-scenario expectations
# ---------------------------------------------------------------------------


@_ALL
def test_scenario_reaches_its_expected_mode(case: Scenario) -> None:
    result = case.analyze()

    assert result.mode is case.expected_mode, (
        f"{case.name}: {case.question}\n"
        f"  expected {case.expected_mode.value}, got {result.mode.value}\n"
        f"  reason: {result.reason}"
    )


@_ALL
def test_scenario_uses_its_expected_placement_method(case: Scenario) -> None:
    result = case.analyze()

    assert result.placement_method is case.expected_placement


# ---------------------------------------------------------------------------
# Conservation invariants — every scenario, every mode
# ---------------------------------------------------------------------------


@_ALL
def test_weights_are_split_between_pools_and_never_duplicated(case: Scenario) -> None:
    result = case.analyze()

    assert result.gpu_weight_bytes + result.ram_weight_bytes == result.weights_bytes


@_ALL
def test_overhead_is_split_between_pools_and_never_duplicated(case: Scenario) -> None:
    result = case.analyze()

    assert result.gpu_overhead_bytes + result.ram_overhead_bytes == result.overhead_bytes


@_ALL
def test_safety_margin_is_split_between_pools_and_never_duplicated(
    case: Scenario,
) -> None:
    result = case.analyze()

    assert (
        result.gpu_safety_margin_bytes + result.ram_safety_margin_bytes
        == result.safety_margin_bytes
    )


@_ALL
def test_a_claimed_placement_fits_the_pools_it_claims(case: Scenario) -> None:
    """A mode other than TOO_LARGE promises each pool holds the share it was given.

    ``gpu_required_bytes`` is only a placement on the two GPU modes. On
    CPU_RAM it is reported as the *hypothetical* resident cost — what the GPU
    would have needed — which is by definition larger than the VRAM available,
    so only the RAM side is a claim there.
    """
    result = case.analyze()
    if result.mode not in _PLACED_MODES:
        return

    if result.mode in (HardwareFitMode.GPU_RESIDENT, HardwareFitMode.GPU_OFFLOAD):
        assert result.gpu_required_bytes is not None
        assert result.available_vram_bytes is not None
        assert result.gpu_required_bytes <= result.available_vram_bytes

    assert result.ram_required_bytes is not None
    assert result.available_ram_bytes is not None
    assert result.ram_required_bytes <= result.available_ram_bytes


@_ALL
def test_gpu_layers_never_exceed_the_layers_the_model_has(case: Scenario) -> None:
    result = case.analyze()
    if result.gpu_layers is None or result.total_layers is None:
        return

    assert 0 <= result.gpu_layers <= result.total_layers


@_ALL
def test_the_weight_split_agrees_with_the_mode(case: Scenario) -> None:
    """Each mode makes a specific claim about where the weights ended up."""
    result = case.analyze()

    if result.mode is HardwareFitMode.GPU_RESIDENT:
        assert result.gpu_weight_bytes == result.weights_bytes
        assert result.ram_weight_bytes == 0
    elif result.mode is HardwareFitMode.GPU_OFFLOAD:
        assert result.gpu_weight_bytes > 0
        assert result.ram_weight_bytes > 0
    else:
        assert result.gpu_weight_bytes == 0
        assert result.ram_weight_bytes == result.weights_bytes


# ---------------------------------------------------------------------------
# Controlled comparisons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "variable"),
    CONTROLLED_PAIRS,
    ids=[f"{a}__vs__{b}" for a, b, _ in CONTROLLED_PAIRS],
)
def test_a_controlled_pair_differs_in_exactly_one_input(
    first: str, second: str, variable: str
) -> None:
    """Otherwise the mode change it demonstrates proves nothing."""
    left = scenario(first).inputs()
    right = scenario(second).inputs()

    differing = {key for key in left if left[key] != right[key]}

    assert differing == {variable}


def test_a_longer_context_moves_the_same_weights_off_the_gpu() -> None:
    short = scenario("kv_small_keeps_the_model_on_the_gpu").analyze()
    long = scenario("kv_large_pushes_the_model_off_the_gpu").analyze()

    assert short.mode is HardwareFitMode.GPU_RESIDENT
    assert long.mode is HardwareFitMode.GPU_OFFLOAD
    assert long.ram_weight_bytes > 0


def test_concurrency_moves_the_same_weights_off_the_gpu() -> None:
    one = scenario("single_user_kv_fits_on_the_gpu").analyze()
    four = scenario("four_concurrent_users_force_offload").analyze()

    assert one.mode is HardwareFitMode.GPU_RESIDENT
    assert four.mode is HardwareFitMode.GPU_OFFLOAD
    assert four.kv_cache_bytes == 4 * one.kv_cache_bytes
    assert four.gpu_weight_bytes < one.gpu_weight_bytes


def test_layer_metadata_decides_how_the_split_is_expressed() -> None:
    """Same placement question, two vocabularies for the answer."""
    with_layers = scenario("layer_placement_with_layer_metadata").analyze()
    without = scenario("byte_fallback_without_layer_metadata").analyze()

    assert with_layers.placement_method is HardwareFitPlacementMethod.LAYERS
    assert with_layers.gpu_layers is not None
    assert without.placement_method is HardwareFitPlacementMethod.ESTIMATED_BYTES
    assert without.gpu_layers is None
    assert without.warnings, "a byte-estimated split must say so"
    # Both are GPU_OFFLOAD; only the granularity of the split differs.
    assert with_layers.mode is without.mode is HardwareFitMode.GPU_OFFLOAD


# ---------------------------------------------------------------------------
# Boundaries and pool separation
# ---------------------------------------------------------------------------


def test_an_exact_vram_boundary_counts_as_a_fit() -> None:
    result = scenario("exact_vram_boundary_is_a_fit").analyze()

    assert result.gpu_required_bytes == result.available_vram_bytes
    assert result.mode is HardwareFitMode.GPU_RESIDENT


def test_an_exact_ram_boundary_counts_as_a_fit() -> None:
    result = scenario("exact_ram_boundary_is_a_fit").analyze()

    assert result.ram_required_bytes == result.available_ram_bytes
    assert result.mode is HardwareFitMode.CPU_RAM


def test_vram_and_ram_are_never_added_into_one_pool() -> None:
    """The sum covers the requirement exactly, and it still does not fit."""
    case = scenario("vram_plus_ram_is_not_a_single_pool")
    result = case.analyze()
    assert result.available_vram_bytes is not None
    assert result.available_ram_bytes is not None

    combined = result.available_vram_bytes + result.available_ram_bytes

    assert (
        result.weights_bytes + result.kv_cache_bytes + result.overhead_bytes == combined
    )
    assert result.mode is HardwareFitMode.TOO_LARGE
    assert result.gpu_weight_bytes == 0


def test_unified_memory_reports_no_separate_vram_pool() -> None:
    result = scenario("unified_memory_uses_one_shared_pool").analyze()

    assert result.memory_topology is HardwareMemoryTopology.UNIFIED_MEMORY
    assert result.available_vram_bytes is None
    assert result.mode is HardwareFitMode.CPU_RAM


# ---------------------------------------------------------------------------
# Two decided semantics, pinned so they cannot drift back
# ---------------------------------------------------------------------------


def test_gpu_layers_is_none_when_there_is_no_gpu_and_zero_when_there_is_one() -> None:
    """``0`` and ``None`` answer different questions and must stay distinct.

    ``0`` means a GPU exists and no layer was placed on it — a count that can
    be validated against ``--n-gpu-layers``. ``None`` means there is no GPU, so
    the question does not apply. Both no-GPU modes must agree on that.
    """
    no_gpu_fits = scenario("exact_ram_boundary_is_a_fit").analyze()
    no_gpu_too_large = scenario("no_gpu_too_large").analyze()
    with_gpu = scenario("cpu_ram_when_no_gpu_placement_is_viable").analyze()

    assert no_gpu_fits.available_vram_bytes is None
    assert no_gpu_fits.gpu_layers is None
    assert no_gpu_too_large.available_vram_bytes is None
    assert no_gpu_too_large.gpu_layers is None

    assert with_gpu.available_vram_bytes is not None
    assert with_gpu.gpu_layers == 0


def test_unified_memory_charges_the_device_reserve_to_the_shared_pool() -> None:
    """A reserve the caller passes must never be silently discarded.

    On discrete hardware the reserve is VRAM, so a CPU_RAM placement does not
    pay for it. On unified memory the accelerator shares the CPU's pool, so it
    does — and the value comes back on the result rather than vanishing.
    """
    without = scenario("unified_memory_uses_one_shared_pool")
    with_reserve = scenario("unified_memory_charges_device_reserve_to_the_pool")
    reserve = with_reserve.device_reserve_bytes
    assert without.device_reserve_bytes == 0
    assert reserve > 0

    plain = without.analyze()
    charged = with_reserve.analyze()

    assert charged.device_reserve_bytes == reserve
    assert plain.ram_required_bytes is not None
    assert charged.ram_required_bytes == plain.ram_required_bytes + reserve


def test_a_discrete_cpu_ram_placement_does_not_pay_the_device_reserve() -> None:
    """The other half of the rule: unused VRAM is not charged to system RAM."""
    case = scenario("cpu_ram_when_no_gpu_placement_is_viable")
    result = case.analyze()

    assert case.device_reserve_bytes > 0
    assert result.ram_required_bytes == (
        case.weights_bytes
        + case.kv_cache_bytes
        + case.overhead_bytes
        + case.safety_margin_bytes
    )


# ---------------------------------------------------------------------------
# The catalogue itself
# ---------------------------------------------------------------------------


def test_catalogue_covers_every_required_case() -> None:
    """Guards the battery against losing a case to an innocent fixture edit."""
    covered: set[str] = set()
    for case in SCENARIOS:
        covered |= case.covers

    assert REQUIRED_COVERAGE - covered == set()
    assert covered - REQUIRED_COVERAGE == set(), "undeclared coverage tag"


def test_scenario_names_are_unique() -> None:
    names = [case.name for case in SCENARIOS]

    assert len(names) == len(set(names))


def test_observations_match_the_recorded_snapshot() -> None:
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = {
        case.name: {"inputs": case.inputs(), "observed": case.observe()}
        for case in SCENARIOS
    }

    assert recorded["observed_fields"] == list(OBSERVED_FIELDS)
    assert recorded["scenarios"] == current, (
        "Hardware fit observations changed. If that was intended, review the "
        "diff and regenerate with:\n"
        "  uv run python scripts/hardware_fit_matrix.py --write-snapshot"
    )
