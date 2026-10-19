"""Hardware Fit blocks and llama.cpp ``--n-gpu-layers`` on a simulated Intel Arc.

These are two different quantities derived from the same estimate, and neither is
converted from the other:

* Hardware Fit answers "how many transformer blocks are viable on this hardware"
  (runtime-agnostic, byte budgets per pool);
* the llama.cpp launch policy answers "which integer does ``--n-gpu-layers`` take",
  with its own conservative heuristic (see ``runtime/llama_cpp_launch_policy.py``).

The tests pin the properties that must hold for a Vulkan device without claiming
the two numbers are equal. The real estimator and the real launch policy run; only
the Hub is faked. Nothing here is a measurement on a B60.
"""

from __future__ import annotations

from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import ComputeBackend, HardwareProfile, MemoryInfo
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import ModelConfig
from jaull.estimator.service import estimate_memory
from jaull.evaluation.comparison import ExecutedPlacementEvidence, _verified_gpu_blocks
from jaull.runtime.backend_selection import select_runtime_backend
from tests._intel_fixtures import B60_FULL, GIB, WINDOWS_OS, scan
from tests._workflow_fixtures import FakeHfClient, gguf_analysis
from tests.test_prediction_comparison import verified_capability

BLOCKS = 32
LLAMA_7B = ModelConfig(
    model_type="llama",
    max_position_embeddings=32768,
    hidden_size=4096,
    num_hidden_layers=BLOCKS,
    num_attention_heads=32,
    num_key_value_heads=8,
)


def _machine(vram_gib: float, *, budget: bool = True) -> HardwareProfile:
    """The scanned B60 profile with its card resized, so one model can fit or not."""
    full = B60_FULL if budget else "\n".join(
        line for line in B60_FULL.splitlines() if "budget" not in line
    )
    profile = scan(WINDOWS_OS, full=full)
    accelerator = profile.accelerators[0].model_copy(
        update={
            "dedicated_memory_bytes": int(vram_gib * GIB),
            "available_memory_bytes": int(vram_gib * GIB * 0.95) if budget else None,
        }
    )
    return profile.model_copy(
        update={
            "accelerators": [accelerator],
            "memory": MemoryInfo(total_bytes=64 * GIB, available_bytes=60 * GIB),
        }
    )


def _estimate(profile: HardwareProfile, weights_gib: int) -> MemoryEstimate:
    analysis = gguf_analysis(quantizations=("Q4_K_M",), base_bytes=weights_gib * GIB).model_copy(
        update={"config": LLAMA_7B}
    )
    config = InferenceConfiguration(
        context_length=4096, quantization="Q4_K_M", target_device=TargetDevice.AUTO
    )
    return estimate_memory(analysis, profile, config, FakeHfClient(), resolve_base_model=False)


def _ngl(estimate: MemoryEstimate) -> int | None:
    recommendation = estimate.runtime_recommendation
    if recommendation is None:
        return None
    value = next((f.value for f in recommendation.flags if f.name == "--n-gpu-layers"), None)
    return int(value) if value is not None else None


def test_the_simulated_machine_selects_vulkan_for_the_intel_card() -> None:
    selection = select_runtime_backend(_machine(24))

    assert selection.selected_backend is ComputeBackend.VULKAN


def test_a_model_that_fits_is_all_on_the_gpu_in_both_vocabularies() -> None:
    estimate = _estimate(_machine(24), weights_gib=5)

    fit = estimate.hardware_fit
    assert fit is not None
    assert fit.mode.value == "gpu_resident"
    assert fit.gpu_transformer_blocks == fit.total_transformer_blocks == BLOCKS
    # "All of it" in llama.cpp's own words, not a block count that happens to equal one.
    assert _ngl(estimate) == -1


def test_the_estimate_uses_the_budget_not_the_physical_capacity() -> None:
    profile = _machine(24)

    estimate = _estimate(profile, weights_gib=5)

    budget = profile.accelerators[0].available_memory_bytes
    assert budget is not None and budget < profile.accelerators[0].dedicated_memory_bytes  # type: ignore[operator]
    assert estimate.assessment.available_vram_bytes == budget


def test_capacity_without_a_budget_never_claims_the_model_fits_in_vram() -> None:
    estimate = _estimate(_machine(24, budget=False), weights_gib=5)

    assert estimate.assessment.available_vram_bytes is None
    assert estimate.assessment.effective_device is TargetDevice.CPU
    fit = estimate.hardware_fit
    assert fit is not None and fit.mode.value != "gpu_resident"
    assert _ngl(estimate) in (None, 0)


def test_partial_offload_keeps_each_number_inside_its_own_bounds() -> None:
    estimate = _estimate(_machine(8), weights_gib=20)

    fit = estimate.hardware_fit
    assert fit is not None and fit.mode.value == "gpu_offload"
    blocks = fit.gpu_transformer_blocks
    assert blocks is not None and 0 < blocks < BLOCKS
    units = _ngl(estimate)
    # A launch-unit count: never "all" (-1) when the model does not fit, never more
    # than the model has, never offloading when Hardware Fit finds no room.
    assert units is not None and 0 < units <= BLOCKS
    recommendation = estimate.runtime_recommendation
    assert recommendation is not None
    assert any("llama.cpp GPU units" in reason for reason in recommendation.reasons)


def test_no_room_for_any_block_means_no_offload() -> None:
    estimate = _estimate(_machine(0.5), weights_gib=40)

    fit = estimate.hardware_fit
    assert fit is not None
    blocks = fit.gpu_transformer_blocks or 0
    units = _ngl(estimate) or 0
    assert blocks == 0 and units == 0


def test_launch_units_are_not_turned_into_blocks_for_a_vulkan_run() -> None:
    """The unit-to-block mapping was read from one llama.cpp build on CUDA, and nothing
    else; for Vulkan a physical VRAM comparison stays refused."""
    profile = _machine(8)
    estimate = _estimate(profile, weights_gib=20)
    selection = select_runtime_backend(profile)
    evidence = ExecutedPlacementEvidence(
        hardware=profile,
        runtime_capability=verified_capability(),
        backend_selection=selection,
    )

    assert _verified_gpu_blocks(_ngl(estimate) or 0, estimate, evidence) is None
