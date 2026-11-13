"""Server sizing: the largest shared KV pool that fits with the serving margin."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jaull.application.serving.planner import ServerSizingError, size_server
from jaull.domain.estimation import (
    HardwareFitMode,
    HardwareFitResult,
    HardwareMemoryTopology,
)
from jaull.domain.inference import InferenceConfiguration

GiB = 1024**3
MiB = 1024**2
VRAM = 24 * GiB
WEIGHTS = 16 * GiB
KV_PER_TOKEN = 20 * MiB // 1024  # 20 MiB per 1K tokens, as Qwen3.6 on the B60
STATE_PER_SEQUENCE = 63 * MiB


def _fit(mode: HardwareFitMode, gpu_bytes: int, *, vram: int = VRAM) -> HardwareFitResult:
    return HardwareFitResult(
        mode=mode,
        memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
        weights_bytes=WEIGHTS,
        kv_cache_bytes=gpu_bytes - WEIGHTS,
        overhead_bytes=0,
        available_vram_bytes=vram,
        gpu_required_bytes=gpu_bytes,
        reason="test",
    )


class FakeEstimator:
    """GPU bytes = weights + KV per token x tokens + recurrent state per sequence."""

    def __init__(self, *, vram: int = VRAM, mode: HardwareFitMode | None = None) -> None:
        self.vram = vram
        self.mode = mode
        self.calls: list[InferenceConfiguration] = []

    def __call__(self, config: InferenceConfiguration) -> Any:
        self.calls.append(config)
        users = config.concurrent_users
        gpu = WEIGHTS + KV_PER_TOKEN * config.context_length * users + STATE_PER_SEQUENCE * users
        mode = self.mode or (
            HardwareFitMode.GPU_RESIDENT if gpu <= self.vram else HardwareFitMode.GPU_OFFLOAD
        )
        return SimpleNamespace(hardware_fit=_fit(mode, gpu, vram=self.vram))


BASE = InferenceConfiguration(context_length=4096)


def test_one_user_gets_the_requested_context_without_a_shared_pool() -> None:
    sizing = size_server(users=1, context_per_user=32768, base=BASE, estimate_for=FakeEstimator())

    assert sizing.parallel == 1
    assert sizing.total_context == 32768
    assert sizing.unified_kv is False
    assert sizing.estimate.hardware_fit.mode is HardwareFitMode.GPU_RESIDENT


def test_everyone_gets_the_full_context_when_it_fits() -> None:
    sizing = size_server(users=4, context_per_user=16384, base=BASE, estimate_for=FakeEstimator())

    assert sizing.parallel == 4
    assert sizing.total_context == 4 * 16384
    assert sizing.unified_kv is True


def test_pool_shrinks_to_the_largest_size_under_the_margin() -> None:
    estimator = FakeEstimator()
    sizing = size_server(users=8, context_per_user=131072, base=BASE, estimate_for=estimator)

    # 8 x 128K would need 16 GiB + 20 GiB of KV; the pool is cut down but keeps 128K for one user.
    assert 131072 <= sizing.total_context < 8 * 131072
    assert sizing.total_context % (8 * 4096) == 0
    assert sizing.unified_kv is True
    assert sizing.context_per_user == 131072
    physical = sizing.estimate.hardware_fit.gpu_physical_bytes
    assert physical <= 0.875 * VRAM
    # One more step would cross the margin.
    bigger = estimator(BASE.model_copy(update={
        "context_length": sizing.total_context // 8 + 4096, "concurrent_users": 8,
    }))
    assert bigger.hardware_fit.gpu_physical_bytes > 0.875 * VRAM
    # Binary search, not a linear scan.
    assert len(estimator.calls) < 12


def test_context_limit_is_the_largest_that_fits_and_matches_sizing() -> None:
    from jaull.application.serving.planner import context_limit

    limit = context_limit(users=8, base=BASE, estimate_for=FakeEstimator())

    assert limit.largest_per_user is not None and limit.largest_per_user % 4096 == 0
    # It really fits, and one step more does not.
    size_server(users=8, context_per_user=limit.largest_per_user, base=BASE,
                estimate_for=FakeEstimator(), shared_context=True)
    with pytest.raises(ServerSizingError):
        size_server(users=8, context_per_user=limit.largest_per_user + 4096, base=BASE,
                    estimate_for=FakeEstimator(), shared_context=True)


def test_context_limit_never_exceeds_the_native_context() -> None:
    from jaull.application.serving.planner import context_limit

    limit = context_limit(users=1, base=BASE, estimate_for=FakeEstimator(), native_limit=32768)

    assert limit.largest_per_user == 32768 and limit.native == 32768
    nothing = context_limit(users=1, base=BASE,
                            estimate_for=FakeEstimator(mode=HardwareFitMode.TOO_LARGE))
    assert nothing.largest_per_user is None


def test_more_than_the_native_context_is_refused_with_the_limit() -> None:
    with pytest.raises(ServerSizingError, match="trained for 8192 tokens") as error:
        size_server(users=1, context_per_user=2_000_000, base=BASE,
                    estimate_for=FakeEstimator(), native_limit=8192)

    assert error.value.largest_context_per_user == 8192


@pytest.mark.parametrize("native", [2048, 6000, 10000])
@pytest.mark.parametrize("users", [1, 3])
def test_native_limit_off_the_default_grid_is_respected(native: int, users: int) -> None:
    from jaull.application.serving.planner import context_limit

    estimator = FakeEstimator()
    limit = context_limit(users=users, base=BASE, estimate_for=estimator, native_limit=native)
    assert limit.largest_per_user == native
    sizing = size_server(users=users, context_per_user=native, base=BASE,
                         estimate_for=estimator, native_limit=native)
    assert sizing.context_per_user == native
    assert sizing.total_context == users * native
    assert all(call.context_length <= native for call in estimator.calls)


def test_shared_context_keeps_one_pool_of_the_asked_context() -> None:
    estimator = FakeEstimator()
    sizing = size_server(users=8, context_per_user=131072, base=BASE,
                         estimate_for=estimator, shared_context=True)

    assert sizing.parallel == 8 and sizing.unified_kv
    assert sizing.total_context == 131072
    assert sizing.context_per_user == 131072
    # Sized for the whole pool, split across 8 sequences.
    assert estimator.calls[-1].context_length * 8 >= 131072
    assert estimator.calls[-1].concurrent_users == 8


def test_shared_context_still_refuses_what_does_not_fit() -> None:
    with pytest.raises(ServerSizingError):
        size_server(users=4, context_per_user=2_000_000, base=BASE,
                    estimate_for=FakeEstimator(), shared_context=True)


def test_estimates_are_sized_per_sequence_share_of_the_pool() -> None:
    estimator = FakeEstimator()
    sizing = size_server(users=2, context_per_user=8192, base=BASE, estimate_for=estimator)

    assert sizing.estimate.hardware_fit.gpu_required_bytes == (
        WEIGHTS + KV_PER_TOKEN * 16384 + STATE_PER_SEQUENCE * 2
    )
    assert all(call.concurrent_users == 2 for call in estimator.calls)


def test_too_large_request_names_the_largest_context_that_fits() -> None:
    small = FakeEstimator(vram=20 * GiB)

    with pytest.raises(ServerSizingError) as error:
        size_server(users=4, context_per_user=262144, base=BASE, estimate_for=small)

    assert error.value.largest_context_per_user is not None
    assert error.value.largest_context_per_user < 262144
    assert error.value.largest_context_per_user % 4096 == 0
    assert "fewer users or a shorter context" in str(error.value)


def test_model_that_never_fits_has_no_suggestion() -> None:
    with pytest.raises(ServerSizingError) as error:
        size_server(
            users=1, context_per_user=4096, base=BASE,
            estimate_for=FakeEstimator(mode=HardwareFitMode.TOO_LARGE),
        )

    assert error.value.largest_context_per_user is None
    assert "smaller quantization" in str(error.value)


def test_gpu_offload_is_not_served() -> None:
    with pytest.raises(ServerSizingError):
        size_server(
            users=1, context_per_user=4096, base=BASE,
            estimate_for=FakeEstimator(mode=HardwareFitMode.GPU_OFFLOAD),
        )


def test_cpu_server_accepts_resident_ram_only() -> None:
    def cpu(mode: HardwareFitMode) -> Any:
        def estimate(config: InferenceConfiguration) -> Any:
            return SimpleNamespace(hardware_fit=HardwareFitResult(
                mode=mode, memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
                weights_bytes=GiB, kv_cache_bytes=0, overhead_bytes=0,
                ram_required_bytes=2 * GiB, reason="test",
            ))
        return estimate

    sizing = size_server(
        users=1, context_per_user=4096, base=BASE, estimate_for=cpu(HardwareFitMode.CPU_RAM),
    )
    assert sizing.total_context == 4096
    with pytest.raises(ServerSizingError):
        size_server(
            users=1, context_per_user=4096, base=BASE,
            estimate_for=cpu(HardwareFitMode.CPU_MEMORY_PRESSURE),
        )


def test_context_is_rounded_up_to_the_planning_step() -> None:
    sizing = size_server(users=1, context_per_user=5000, base=BASE, estimate_for=FakeEstimator())

    assert sizing.context_per_user == 8192


@pytest.mark.parametrize(("users", "context"), [(0, 4096), (1, 0)])
def test_rejects_non_positive_inputs(users: int, context: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        size_server(users=users, context_per_user=context, base=BASE, estimate_for=FakeEstimator())


def test_explicit_cpu_target_is_judged_by_the_assessment() -> None:
    # A CPU-target estimate carries no hardware-fit placement (reproduced with
    # Qwen2.5-1.5B on the laptop), only the compatibility assessment.
    from jaull.domain.estimation import CompatibilityStatus
    from jaull.domain.inference import TargetDevice

    def estimate_with(status: CompatibilityStatus) -> Any:
        def estimate(config: InferenceConfiguration) -> Any:
            return SimpleNamespace(
                hardware_fit=None, inference_configuration=config,
                assessment=SimpleNamespace(status=status),
            )
        return estimate

    cpu = BASE.model_copy(update={"target_device": TargetDevice.CPU})
    sizing = size_server(users=2, context_per_user=8192, base=cpu,
                         estimate_for=estimate_with(CompatibilityStatus.COMFORTABLE))
    assert sizing.total_context == 2 * 8192
    assert sizing.gpu_budget_bytes is None
    with pytest.raises(ServerSizingError):
        size_server(users=1, context_per_user=8192, base=cpu,
                    estimate_for=estimate_with(CompatibilityStatus.TIGHT))
    with pytest.raises(ServerSizingError):
        size_server(users=1, context_per_user=8192, base=BASE,
                    estimate_for=estimate_with(CompatibilityStatus.COMFORTABLE))
