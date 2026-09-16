from __future__ import annotations

import pytest

from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimateSource,
    EstimationConfidence,
    KvCacheEstimate,
    MemoryComponent,
    MemoryEstimate,
    RuntimeOverheadEstimate,
    WeightEstimate,
)
from jaull.domain.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import ModelRepositoryInfo
from jaull.runtime import transformers as transformers_runtime
from jaull.runtime.transformers_quantization import (
    build_quantization_config,
    quantization_mode,
)

GIB = 1024**3


def _hw() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=32 * GIB),
        storage=[],
        gpus=[
            GpuInfo(
                name="Test",
                vram_total_bytes=24 * GIB,
                vram_available_bytes=24 * GIB,
                driver_version="1.0",
                cuda_version="12.4",
            )
        ],
        warnings=[],
    )


def _estimate(
    *,
    status: CompatibilityStatus,
    effective_device: TargetDevice,
    precision: WeightPrecision | None = WeightPrecision.FLOAT16,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="user/tf-model"),
        repository_type=RepositoryType.TRANSFORMERS,
        inference_configuration=InferenceConfiguration(
            context_length=2048, target_device=TargetDevice.AUTO
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=2 * GIB,
                source=EstimateSource.METADATA,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            precision=precision,
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=256 * 1024 * 1024,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=32,
            kv_heads=32,
            head_dim=128,
            context_length=2048,
            batch_size=1,
            dtype_bytes=2,
            formula="test",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=512 * 1024 * 1024,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=512 * 1024 * 1024,
            weight_fraction=0.1,
            minimum_bytes=256 * 1024 * 1024,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=0,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=3 * GIB,
        assessment=CompatibilityAssessment(
            status=status,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.AUTO,
            effective_device=effective_device,
            available_vram_bytes=24 * GIB,
            available_ram_bytes=32 * GIB,
            ratio=0.1,
        ),
    )


def test_gpu_recommends_cuda_device_map() -> None:
    est = _estimate(
        status=CompatibilityStatus.COMFORTABLE, effective_device=TargetDevice.GPU
    )
    rec = transformers_runtime.build(est, _hw())
    device_map_flag = next(f for f in rec.flags if f.name == "device_map")
    assert device_map_flag.value == "cuda"
    assert rec.python_snippet is not None
    assert "cuda" in rec.python_snippet


def test_offloading_uses_auto_device_map_with_warning() -> None:
    est = _estimate(
        status=CompatibilityStatus.OFFLOADING_REQUIRED,
        effective_device=TargetDevice.GPU,
    )
    rec = transformers_runtime.build(est, _hw())
    device_map_flag = next(f for f in rec.flags if f.name == "device_map")
    assert device_map_flag.value == "auto"
    assert any("accelerate" in w.lower() for w in rec.warnings)


def test_cpu_recommends_cpu_device_map() -> None:
    est = _estimate(
        status=CompatibilityStatus.COMPATIBLE, effective_device=TargetDevice.CPU
    )
    rec = transformers_runtime.build(est, _hw())
    device_map_flag = next(f for f in rec.flags if f.name == "device_map")
    assert device_map_flag.value == "cpu"
    assert rec.python_snippet is not None
    assert "cpu" in rec.python_snippet


def test_device_map_policy_cpu_when_effective_device_is_cpu() -> None:
    from jaull.runtime.transformers_launch_policy import pick_device_map

    plan = pick_device_map(
        _estimate(
            status=CompatibilityStatus.COMPATIBLE, effective_device=TargetDevice.CPU
        )
    )
    assert plan.device_map == "cpu"


def test_device_map_policy_auto_when_offloading() -> None:
    from jaull.runtime.transformers_launch_policy import pick_device_map

    plan = pick_device_map(
        _estimate(
            status=CompatibilityStatus.OFFLOADING_REQUIRED,
            effective_device=TargetDevice.GPU,
        )
    )
    assert plan.device_map == "auto"
    assert plan.warnings


# ---------------------------------------------------------------------------
# The estimate and the command must describe the same run
#
# `torch_dtype=torch.int8` was emitted for both int8 and int4 estimates, and the
# workers passed it straight to `from_pretrained` with no quantization config.
# So a plan predicted at 0.5 bytes/parameter executed as something else, and a
# recorded experiment would have compared a prediction against a run of a
# different configuration. Nothing in the suite noticed, because nothing
# exercised a quantized precision end to end.
# ---------------------------------------------------------------------------

_PRECISION_FLAGS = ("torch_dtype", "quantization")


def _precision_flags(rec: object) -> list[object]:
    return [f for f in rec.flags if f.name in _PRECISION_FLAGS]


@pytest.mark.parametrize("precision", list(WeightPrecision))
def test_exactly_one_flag_carries_the_precision(precision: WeightPrecision) -> None:
    rec = transformers_runtime.build(
        _estimate(
            status=CompatibilityStatus.COMFORTABLE,
            effective_device=TargetDevice.GPU,
            precision=precision,
        ),
        _hw(),
    )

    assert len(_precision_flags(rec)) == 1, (
        f"{precision.value} produced {[f.name for f in _precision_flags(rec)]}"
    )


def test_no_two_precisions_share_a_flag_value() -> None:
    """The regression, stated directly: int4 and int8 both said `torch.int8`."""
    seen: dict[tuple[str, str], WeightPrecision] = {}
    for precision in WeightPrecision:
        rec = transformers_runtime.build(
            _estimate(
                status=CompatibilityStatus.COMFORTABLE,
                effective_device=TargetDevice.GPU,
                precision=precision,
            ),
            _hw(),
        )
        flag = _precision_flags(rec)[0]
        key = (flag.name, flag.value)
        assert key not in seen, (
            f"{precision.value} and {seen[key].value} both emit {flag.name}={flag.value}"
        )
        seen[key] = precision


@pytest.mark.parametrize(
    "precision", [WeightPrecision.INT4, WeightPrecision.INT8]
)
def test_a_quantized_estimate_asks_for_a_quantization_config(
    precision: WeightPrecision,
) -> None:
    """Never a torch dtype: `from_pretrained(torch_dtype=torch.int8)` does not quantize."""
    rec = transformers_runtime.build(
        _estimate(
            status=CompatibilityStatus.COMFORTABLE,
            effective_device=TargetDevice.GPU,
            precision=precision,
        ),
        _hw(),
    )

    flag = _precision_flags(rec)[0]
    assert flag.name == "quantization"
    assert flag.value == quantization_mode(precision)
    assert rec.python_snippet is not None
    assert "BitsAndBytesConfig" in rec.python_snippet
    assert "torch_dtype" not in rec.python_snippet
    # The library is not a Jaull dependency, and the user is told so.
    assert any("bitsandbytes" in warning for warning in rec.warnings)


@pytest.mark.parametrize(
    "precision",
    [WeightPrecision.FLOAT32, WeightPrecision.FLOAT16, WeightPrecision.BFLOAT16],
)
def test_a_float_estimate_still_loads_as_a_dtype(
    precision: WeightPrecision,
) -> None:
    rec = transformers_runtime.build(
        _estimate(
            status=CompatibilityStatus.COMFORTABLE,
            effective_device=TargetDevice.GPU,
            precision=precision,
        ),
        _hw(),
    )

    flag = _precision_flags(rec)[0]
    assert flag.name == "torch_dtype"
    assert flag.value == f"torch.{precision.value}"
    assert rec.python_snippet is not None
    assert "BitsAndBytesConfig" not in rec.python_snippet


def test_an_unknown_quantization_mode_is_refused_not_ignored() -> None:
    with pytest.raises(RuntimeError, match="Unknown quantization mode"):
        build_quantization_config("3bit")


def test_no_quantization_requested_is_not_an_error() -> None:
    assert build_quantization_config(None) is None
    assert build_quantization_config("") is None
