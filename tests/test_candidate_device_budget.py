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
