"""The context sweep: how much memory a model needs at each context, and where it stops fitting."""

from __future__ import annotations

from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence, MemoryEstimate
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis, ModelConfig
from jaull.estimator import policies
from jaull.estimator.context_fit import native_context, sweep_context
from tests._workflow_fixtures import GIB, gguf_analysis, memory_estimate

KIB = 1024
BUDGET = 24 * GIB


def _status(total: int) -> CompatibilityStatus:
    ratio = total / BUDGET
    if ratio <= policies.COMFORTABLE_MAX:
        return CompatibilityStatus.COMFORTABLE
    if ratio <= policies.COMPATIBLE_MAX:
        return CompatibilityStatus.COMPATIBLE
    if ratio <= policies.TIGHT_MAX:
        return CompatibilityStatus.TIGHT
    return CompatibilityStatus.INSUFFICIENT


def _linear_estimator(
    fixed_bytes: int, bytes_per_token: int, calls: list[int] | None = None
) -> object:
    """Total memory that grows linearly with the context, like a KV cache does."""

    def estimate(analysis: ModelAnalysis, config: InferenceConfiguration) -> MemoryEstimate:
        if calls is not None:
            calls.append(config.context_length)
        total = fixed_bytes + bytes_per_token * config.context_length
        return memory_estimate(
            analysis, config, total_bytes=total, status=_status(total), available_vram=BUDGET
        )

    return estimate


def _config(context: int = 4096) -> InferenceConfiguration:
    return InferenceConfiguration(context_length=context, quantization="Q4_K_M")


def test_one_point_per_rung_in_ascending_order() -> None:
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(10 * GIB, 20 * KIB)  # type: ignore[arg-type]
    )
    contexts = [point.context_length for point in sweep.points]
    assert contexts == sorted(policies.CONTEXT_LADDER)


def test_estimator_is_called_with_the_rung_and_nothing_else_changed() -> None:
    seen: list[InferenceConfiguration] = []
    base = _linear_estimator(10 * GIB, 20 * KIB)

    def spy(analysis: ModelAnalysis, config: InferenceConfiguration) -> MemoryEstimate:
        seen.append(config)
        return base(analysis, config)  # type: ignore[operator, no-any-return]

    sweep_context(gguf_analysis(), _config(4096), spy, ladder=(4096, 8192))
    assert [c.context_length for c in seen] == [4096, 8192]
    assert all(c.quantization == "Q4_K_M" for c in seen)
    assert all(c.batch_size == 1 for c in seen)


def test_largest_fitting_is_the_highest_rung_that_fits() -> None:
    # 10 GiB fixed + 20 KiB per token: 131072 tokens are 2.5 GiB (12.5 GiB, comfortable),
    # 262144 tokens are 5 GiB (15 GiB, comfortable). Make the cache heavier to hit the wall.
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(10 * GIB, 128 * KIB)  # type: ignore[arg-type]
    )
    # 128 KiB/token: 65536 -> 8 GiB (18 GiB), 131072 -> 16 GiB (26 GiB, insufficient).
    assert sweep.largest_fitting == 65536
    by_context = {p.context_length: p for p in sweep.points}
    assert by_context[131072].status is CompatibilityStatus.INSUFFICIENT


def test_everything_fits_reports_the_top_of_the_ladder() -> None:
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(4 * GIB, 1 * KIB)  # type: ignore[arg-type]
    )
    assert sweep.largest_fitting == max(policies.CONTEXT_LADDER)


def test_a_ladder_that_never_runs_out_says_the_real_limit_may_be_higher() -> None:
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(4 * GIB, 1 * KIB)  # type: ignore[arg-type]
    )
    assert any("may fit more" in note for note in sweep.notes)


def test_a_wall_inside_the_ladder_does_not_claim_there_is_more() -> None:
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(10 * GIB, 128 * KIB)  # type: ignore[arg-type]
    )
    assert not any("may fit more" in note for note in sweep.notes)


def test_nothing_fits_reports_none_and_says_so() -> None:
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(30 * GIB, 1 * KIB)  # type: ignore[arg-type]
    )
    assert sweep.largest_fitting is None
    assert any("smallest" in note.lower() for note in sweep.notes)


def test_a_rung_above_a_failure_never_counts_even_if_the_estimator_says_it_fits() -> None:
    def odd(analysis: ModelAnalysis, config: InferenceConfiguration) -> MemoryEstimate:
        fits = config.context_length != 16384
        status = CompatibilityStatus.COMFORTABLE if fits else CompatibilityStatus.INSUFFICIENT
        return memory_estimate(analysis, config, total_bytes=8 * GIB, status=status)

    sweep = sweep_context(gguf_analysis(), _config(), odd, ladder=(4096, 8192, 16384, 32768))
    assert sweep.largest_fitting == 8192


def test_offloading_is_not_a_fit() -> None:
    def offload_from_32k(
        analysis: ModelAnalysis, config: InferenceConfiguration
    ) -> MemoryEstimate:
        status = (
            CompatibilityStatus.OFFLOADING_REQUIRED
            if config.context_length >= 32768
            else CompatibilityStatus.COMPATIBLE
        )
        return memory_estimate(analysis, config, total_bytes=8 * GIB, status=status)

    sweep = sweep_context(
        gguf_analysis(), _config(), offload_from_32k, ladder=(8192, 16384, 32768, 65536)
    )
    assert sweep.largest_fitting == 16384


def test_unknown_status_is_not_a_fit() -> None:
    def unknown(analysis: ModelAnalysis, config: InferenceConfiguration) -> MemoryEstimate:
        return memory_estimate(
            analysis, config, total_bytes=8 * GIB, status=CompatibilityStatus.UNKNOWN
        )

    sweep = sweep_context(gguf_analysis(), _config(), unknown, ladder=(4096, 8192))
    assert sweep.largest_fitting is None
    assert all(p.status is CompatibilityStatus.UNKNOWN for p in sweep.points)


def test_native_context_caps_the_ladder_and_says_so() -> None:
    calls: list[int] = []
    sweep = sweep_context(
        gguf_analysis(),
        _config(),
        _linear_estimator(4 * GIB, 1 * KIB, calls),  # type: ignore[arg-type]
        native_limit=32768,
    )
    assert max(p.context_length for p in sweep.points) == 32768
    assert max(calls) == 32768
    assert sweep.largest_fitting == 32768
    assert sweep.native_limit == 32768
    assert any("32768" in note for note in sweep.notes)


def test_a_native_limit_between_rungs_is_itself_a_point() -> None:
    sweep = sweep_context(
        gguf_analysis(),
        _config(),
        _linear_estimator(4 * GIB, 1 * KIB),  # type: ignore[arg-type]
        ladder=(4096, 8192, 16384),
        native_limit=12000,
    )
    assert [p.context_length for p in sweep.points] == [4096, 8192, 12000]
    assert sweep.largest_fitting == 12000


def test_points_carry_the_estimate_they_came_from() -> None:
    sweep = sweep_context(
        gguf_analysis(),
        _config(),
        _linear_estimator(10 * GIB, 20 * KIB),  # type: ignore[arg-type]
        ladder=(4096,),
    )
    point = sweep.points[0]
    assert point.total_bytes == 10 * GIB + 20 * KIB * 4096
    assert point.kv_cache_bytes == GIB // 2  # what the fixture estimate reports
    assert point.confidence is EstimationConfidence.HIGH


def test_the_sweep_confidence_is_the_weakest_of_its_points() -> None:
    def mixed(analysis: ModelAnalysis, config: InferenceConfiguration) -> MemoryEstimate:
        confidence = (
            EstimationConfidence.LOW
            if config.context_length >= 8192
            else EstimationConfidence.HIGH
        )
        return memory_estimate(
            analysis,
            config,
            total_bytes=8 * GIB,
            status=CompatibilityStatus.COMFORTABLE,
            confidence=confidence,
        )

    sweep = sweep_context(gguf_analysis(), _config(), mixed, ladder=(4096, 8192))
    assert sweep.weakest_confidence is EstimationConfidence.LOW


def test_a_sweep_with_no_points_is_unknown_not_high() -> None:
    sweep = sweep_context(
        gguf_analysis(), _config(), _linear_estimator(4 * GIB, KIB), ladder=()  # type: ignore[arg-type]
    )
    assert sweep.weakest_confidence is EstimationConfidence.UNKNOWN


def test_the_native_context_comes_from_the_declared_max_position_embeddings() -> None:
    analysis = gguf_analysis().model_copy(
        update={"config": ModelConfig(max_position_embeddings=32768)}
    )
    assert native_context(analysis) == 32768


def test_there_is_no_native_context_without_a_config_or_without_the_field() -> None:
    assert native_context(gguf_analysis()) is None
    analysis = gguf_analysis().model_copy(update={"config": ModelConfig()})
    assert native_context(analysis) is None


def test_the_default_ladder_is_ascending_and_lives_in_policies() -> None:
    assert list(policies.CONTEXT_LADDER) == sorted(set(policies.CONTEXT_LADDER))
    assert policies.CONTEXT_LADDER[0] <= 4096
