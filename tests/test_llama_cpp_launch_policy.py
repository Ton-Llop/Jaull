from __future__ import annotations

from jaull.domain.estimation import EstimationConfidence
from jaull.runtime.llama_cpp_launch_policy import pick_gpu_layers
from tests._execution_fixtures import qwen_ctx4096_estimate


def test_qwen_ctx4096_launch_policy_still_picks_23_layers() -> None:
    plan = pick_gpu_layers(qwen_ctx4096_estimate(with_runtime_recommendation=False))
    assert plan.n_gpu_layers == 23
    assert plan.confidence is EstimationConfidence.MEDIUM


def test_launch_policy_number_is_not_the_hardware_fit_block_count() -> None:
    """23 (policy) must not equal 18 or 19 (HFA) — proves no blocks->layers wiring."""
    plan = pick_gpu_layers(qwen_ctx4096_estimate(with_runtime_recommendation=False))
    assert plan.n_gpu_layers not in {18, 19}
