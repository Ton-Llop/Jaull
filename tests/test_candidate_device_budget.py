from __future__ import annotations

import pytest

from jaull.discovery.candidate_filter import coarse_placement_hint
from jaull.domain.estimation import HardwareFitMode
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
)
from tests._workflow_fixtures import GIB, candidate, hardware
from tests.test_discovery_search import _requirements


def test_moe_active_size_does_not_replace_total_size_in_the_memory_envelope():
    from jaull.discovery.candidate_filter import parameter_count_hint

    profile = hardware(vram_gib=6, ram_gib=16)
    req = _requirements()
    dense = candidate("vendor/Model-30B-Instruct-GGUF", tags=["gguf"])
    sparse = candidate("vendor/Model-30B-A3B-Instruct-GGUF", tags=["gguf", "moe"])
    assert parameter_count_hint(sparse.repo_id) == 30
    assert coarse_placement_hint(sparse, req, profile) == coarse_placement_hint(dense, req, profile)


def test_moe_and_dense_candidates_use_the_same_offload_memory_rules():
    profile = hardware(vram_gib=6, ram_gib=16)
    req = _requirements()
    dense = candidate("vendor/Model-18B-Instruct-GGUF", tags=["gguf"])
    sparse = candidate("vendor/Model-18B-A2B-Instruct-GGUF", tags=["gguf", "moe"])
    dense_hint = coarse_placement_hint(dense, req, profile)
    sparse_hint = coarse_placement_hint(sparse, req, profile)
    assert sparse_hint == dense_hint
    assert sparse_hint.mode is HardwareFitMode.GPU_OFFLOAD


def _card(backend, gib, availability=BackendAvailability.AVAILABLE, *, software=False):
    return AcceleratorProfile(
        name=f"{backend.value}-{gib}",
        type=AcceleratorType.DEDICATED,
        dedicated_memory_bytes=gib * GIB,
        backends=[
            ComputeBackendInfo(
                backend=backend,
                availability=availability,
                software_renderer=software,
            )
        ],
    )


@pytest.mark.parametrize("backend", [ComputeBackend.CUDA, ComputeBackend.HIP])
def test_transformers_cannot_borrow_vram_from_a_larger_vulkan_only_card(backend):
    hw = hardware(ram_gib=4, vram_gib=None).model_copy(
        update={
            "accelerators": [_card(backend, 4), _card(ComputeBackend.VULKAN, 24)],
        }
    )
    req = _requirements()
    native = coarse_placement_hint(candidate("org/Model-7B-Instruct"), req, hw)
    gguf = coarse_placement_hint(candidate("org/Model-7B-Instruct-GGUF"), req, hw)
    assert native.mode is HardwareFitMode.TOO_LARGE
    assert gguf.mode is HardwareFitMode.GPU_RESIDENT
    assert native.estimated_weight_bytes == 14_000_000_000


def test_cuda_only_quantization_cannot_borrow_the_hip_cards_memory():
    hw = hardware(ram_gib=4, vram_gib=None).model_copy(
        update={
            "accelerators": [_card(ComputeBackend.CUDA, 4), _card(ComputeBackend.HIP, 24)],
        }
    )
    hint = coarse_placement_hint(candidate("org/Model-7B-AWQ"), _requirements(), hw)
    assert hint.mode is not HardwareFitMode.GPU_RESIDENT
    assert hint.gpu_weight_bytes is None or hint.gpu_weight_bytes <= 4 * GIB


@pytest.mark.parametrize(
    "availability", [BackendAvailability.UNKNOWN, BackendAvailability.UNAVAILABLE]
)
def test_explicit_backend_probe_results_override_the_legacy_nvml_view(availability):
    hw = hardware(ram_gib=32, vram_gib=24).model_copy(
        update={
            "accelerators": [_card(ComputeBackend.CUDA, 24, availability)],
        }
    )
    hint = coarse_placement_hint(candidate("org/Model-7B-Instruct"), _requirements(), hw)
    assert hint.mode is HardwareFitMode.CPU_RAM


def test_vulkan_only_machine_can_still_consider_transformers_on_cpu():
    hw = hardware(ram_gib=32, vram_gib=None).model_copy(
        update={
            "accelerators": [_card(ComputeBackend.VULKAN, 24)],
        }
    )
    assert (
        coarse_placement_hint(
            candidate("org/Model-7B-Instruct"),
            _requirements(),
            hw,
        ).mode
        is HardwareFitMode.CPU_RAM
    )


def test_two_small_compatible_cards_do_not_form_a_combined_vram_pool():
    hw = hardware(ram_gib=4, vram_gib=None).model_copy(
        update={
            "accelerators": [_card(ComputeBackend.CUDA, 8), _card(ComputeBackend.CUDA, 8)],
        }
    )
    assert (
        coarse_placement_hint(
            candidate("org/Model-7B-Instruct"),
            _requirements(),
            hw,
        ).mode
        is not HardwareFitMode.GPU_RESIDENT
    )


def test_legacy_nvml_only_hardware_keeps_its_cuda_capacity():
    assert (
        coarse_placement_hint(
            candidate("org/Model-7B-Instruct"),
            _requirements(),
            hardware(vram_gib=24),
        ).mode
        is HardwareFitMode.GPU_RESIDENT
    )


def test_software_backend_does_not_supply_a_gpu_memory_budget():
    hw = hardware(ram_gib=32, vram_gib=None).model_copy(
        update={
            "accelerators": [_card(ComputeBackend.VULKAN, 24, software=True)],
        }
    )
    assert (
        coarse_placement_hint(
            candidate("org/Model-7B-Instruct-GGUF"),
            _requirements(),
            hw,
        ).mode
        is HardwareFitMode.CPU_RAM
    )
