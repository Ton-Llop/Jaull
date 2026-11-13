from __future__ import annotations

import math

import pytest
from typer.testing import CliRunner

from jaull.cli import app as cli_module
from jaull.domain.estimation import CompatibilityStatus, HardwareFitMode
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import GgufVariant, ModelConfig, ModelFile
from jaull.estimator.hardware_fit import analyze_estimate
from jaull.estimator.service import estimate_memory
from jaull.presentation.plan_labels import hardware_summary
from jaull.reporting.estimation import estimate_to_json_dict
from tests.test_memory_estimator import (
    GIB,
    MIB,
    _FakeClient,
    _gguf_analysis,
    _hardware,
    _transformers_analysis,
)


def _laptop(*, shared: bool = False, vram: int | None = None) -> HardwareProfile:
    profile = _hardware(ram=8 * GIB, vram=vram)
    return profile.model_copy(update={
        "memory": MemoryInfo(total_bytes=7983022080, available_bytes=5330571264),
        "accelerators": [AcceleratorProfile(
            name="Microsoft Direct3D12 (AMD Radeon(TM) Graphics)",
            type=AcceleratorType.INTEGRATED, shared_memory=True,
            backends=[ComputeBackendInfo(
                backend=ComputeBackend.VULKAN, availability=BackendAvailability.AVAILABLE,
            )],
        )] if shared else [],
    })


def _estimate(hardware: HardwareProfile, target: TargetDevice = TargetDevice.AUTO, **knobs):
    size = round(3.6 * GIB)
    variant = GgufVariant(
        quantization="Q3_K_M", total_bytes=size,
        files=[ModelFile(path="model.Q3_K_M.gguf", size_bytes=size)],
    )
    analysis = _gguf_analysis([variant]).model_copy(update={"config": ModelConfig(
        num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4,
        hidden_size=3584,
    )})
    return estimate_memory(
        analysis, hardware,
        InferenceConfiguration(context_length=4096, target_device=target, **knobs),
        _FakeClient(), resolve_base_model=False, recommend_runtime=False,
    )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("target", [TargetDevice.CPU, TargetDevice.AUTO])
def test_cpu_uses_ram_reserve_even_with_a_vulkan_igpu(shared, target) -> None:
    estimate = _estimate(_laptop(shared=shared), target, device_reserve_bytes=4 * GIB)
    assert estimate.assessment.effective_device is TargetDevice.CPU
    assert estimate.assessment.status in {
        CompatibilityStatus.COMPATIBLE, CompatibilityStatus.TIGHT,
    }
    assert estimate.device_reserve.name == "RAM reserve"
    assert estimate.device_reserve.bytes == 512 * MIB
    assert estimate.safety_margin is None
    overhead = 512 * MIB + int(estimate.weights.component.bytes * 0.1)
    assert estimate.runtime_overhead.component.bytes == overhead
    physical = estimate.weights.component.bytes + 224 * MIB + overhead
    assert estimate.total_bytes == physical + 512 * MIB
    assert estimate.total_bytes + 512 * MIB > 5330571264  # the old 1 GiB floor rejected it
    if estimate.hardware_fit is not None:
        assert estimate.hardware_fit.ram_reserve_bytes == 512 * MIB
        assert estimate.hardware_fit.ram_required_bytes == estimate.total_bytes
        assert estimate.hardware_fit.ram_physical_bytes == physical


@pytest.mark.parametrize("reserve", [0, 256 * MIB, 2 * GIB])
def test_explicit_cpu_reserve_survives_json_and_is_not_floored(reserve) -> None:
    estimate = _estimate(_laptop(shared=True), ram_reserve_bytes=reserve)
    assert estimate.device_reserve.bytes == reserve
    payload = estimate_to_json_dict(estimate)
    restored = InferenceConfiguration.model_validate(payload["inference_configuration"])
    assert restored.ram_reserve_bytes == reserve
    assert payload["memory"]["device_reserve_bytes"] == reserve
    assert payload["memory"]["components"][3]["name"] == "RAM reserve"


def test_consuming_reserve_is_tight_not_insufficient() -> None:
    estimate = _estimate(_laptop(), ram_reserve_bytes=2 * GIB)
    assert estimate.total_bytes < 7983022080
    assert estimate.total_bytes > 5330571264
    assert estimate.assessment.status is CompatibilityStatus.TIGHT
    assert estimate.physical_bytes < 5330571264
    assert estimate.assessment.reserve_shortfall_bytes > 0


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("target", [TargetDevice.CPU, TargetDevice.AUTO])
def test_cpu_pressure_is_experimental_not_a_confirmed_fit(shared, target) -> None:
    profile = _laptop(shared=shared).model_copy(update={
        "memory": MemoryInfo(total_bytes=8 * GIB, available_bytes=4 * GIB),
    })
    estimate = _estimate(profile, target)
    assert estimate.assessment.status is CompatibilityStatus.MEMORY_PRESSURE
    assert estimate.physical_bytes > profile.memory.available_bytes
    assert estimate.assessment.physical_bytes == estimate.physical_bytes
    if estimate.hardware_fit is not None:
        assert estimate.hardware_fit.mode is HardwareFitMode.CPU_MEMORY_PRESSURE
        assert estimate.hardware_fit.ram_physical_bytes == estimate.physical_bytes


def test_cpu_consumption_larger_than_total_ram_stays_insufficient() -> None:
    profile = _laptop().model_copy(update={
        "memory": MemoryInfo(total_bytes=4 * GIB, available_bytes=3 * GIB),
    })
    assert _estimate(profile).assessment.status is CompatibilityStatus.INSUFFICIENT


def test_pressure_never_earns_a_recommended_or_primary_label() -> None:
    from jaull.recommendation.policies import has_confirmed_memory_compatibility
    from jaull.recommendation.tier import RecommendationTier, choose_tier

    assessment = _estimate(_laptop().model_copy(update={
        "memory": MemoryInfo(total_bytes=8 * GIB, available_bytes=4 * GIB),
    })).assessment
    assert not has_confirmed_memory_compatibility(assessment.status)
    assert choose_tier(
        assessment.status, assessment.confidence, 1.0,
    ) is RecommendationTier.BEST_EFFORT


@pytest.mark.parametrize("target", [TargetDevice.CPU, TargetDevice.AUTO])
def test_native_cpu_execution_uses_the_separate_ram_policy(target) -> None:
    estimate = estimate_memory(
        _transformers_analysis(ModelConfig(
            num_hidden_layers=28, num_attention_heads=28,
            num_key_value_heads=4, hidden_size=3584,
        )),
        _hardware(ram=32 * GIB, vram=None),
        InferenceConfiguration(
            context_length=4096, target_device=target,
            device_reserve_bytes=8 * GIB, ram_reserve_bytes=256 * MIB,
        ),
        _FakeClient(), resolve_base_model=False, recommend_runtime=False,
    )
    assert estimate.assessment.effective_device is TargetDevice.CPU
    assert estimate.device_reserve.bytes == 256 * MIB
    subtotal = (
        estimate.weights.component.bytes + estimate.kv_cache.component.bytes
        + estimate.runtime_overhead.component.bytes + 256 * MIB
    )
    assert estimate.safety_margin.bytes == math.ceil(subtotal * 0.1)
    assert estimate.total_bytes == subtotal + estimate.safety_margin.bytes


def test_auto_cpu_fallback_uses_its_own_reserve_and_margin() -> None:
    hardware = _laptop(vram=128 * MIB)
    estimate = _estimate(hardware, ram_reserve_bytes=256 * MIB, safety_margin_percent=10)
    assert estimate.hardware_fit.mode is HardwareFitMode.CPU_RAM
    assert estimate.device_reserve.bytes == 256 * MIB
    subtotal = (
        estimate.weights.component.bytes + 224 * MIB
        + estimate.runtime_overhead.component.bytes + 256 * MIB
    )
    assert estimate.safety_margin.bytes == math.ceil(subtotal * 0.1)
    assert estimate.hardware_fit.ram_required_bytes == estimate.total_bytes
    assert estimate.hardware_fit.ram_physical_bytes == subtotal - 256 * MIB
    repeated = analyze_estimate(estimate, hardware)
    assert repeated.ram_required_bytes == estimate.total_bytes
    assert repeated.ram_physical_bytes == estimate.hardware_fit.ram_physical_bytes


@pytest.mark.parametrize("target", [TargetDevice.AUTO, TargetDevice.GPU])
def test_gpu_policy_still_has_one_gib_floor_and_ignores_cpu_reserve(target) -> None:
    hardware = _hardware(ram=16 * GIB, vram=16 * GIB)
    estimate = _estimate(hardware, target, device_reserve_bytes=0, ram_reserve_bytes=8 * GIB)
    assert estimate.assessment.effective_device is TargetDevice.GPU
    assert estimate.device_reserve.bytes == GIB
    assert estimate.device_reserve.name == "Device reserve"
    assert estimate.total_bytes == estimate.weights.component.bytes + 224 * MIB + 256 * MIB + GIB


def test_summary_puts_total_and_available_ram_together() -> None:
    text = hardware_summary(_laptop(shared=True))
    assert "RAM 7.43 GiB total (4.96 GiB available)" in text


def test_cli_keeps_cpu_and_gpu_reserve_options_independent(monkeypatch) -> None:
    captured = []

    def run(reference, options):
        captured.append(options)
        return 0

    monkeypatch.setattr(cli_module, "run_estimate", run)
    result = CliRunner().invoke(cli_module.app, [
        "estimate", "user/model", "--ram-reserve-gib", "0.25", "--device-reserve-gib", "2",
    ])
    assert result.exit_code == 0, result.output
    assert captured[0].ram_reserve_gib == 0.25
    assert captured[0].device_reserve_gib == 2

    from jaull.cli.estimate import _build_configuration

    config = _build_configuration(None, captured[0])
    assert config.ram_reserve_bytes == 256 * MIB
    assert config.device_reserve_bytes == 2 * GIB
