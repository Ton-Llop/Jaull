from __future__ import annotations

from jaull.domain.estimation import (
    EstimationConfidence,
    MemoryEstimate,
    TransformerBlockWeightDecomposition,
    TransformerBlockWeightDecompositionMethod,
)
from jaull.runtime.llama_cpp_launch_policy import pick_gpu_layers
from tests._execution_fixtures import (
    QWEN_BLOCKS,
    QWEN_VRAM,
    QWEN_WEIGHTS,
    qwen_ctx4096_estimate,
)

# The real decomposition the estimator produced for the B001-R4 baseline, from
# validation/bundles/b001-r4-launch-policy/records/experiment.json.
QWEN_NON_BLOCK = 670_300_265
QWEN_BLOCK_WEIGHTS = 4_012_773_975
QWEN_BYTES_PER_BLOCK = 143_313_357
# Qwen2.5-7B does not tie its embeddings, so the two halves are equal in
# parameters and the byte projection splits the aggregate down the middle.
QWEN_EMBEDDINGS = QWEN_NON_BLOCK // 2
QWEN_OUTPUT_HEAD = QWEN_NON_BLOCK - QWEN_EMBEDDINGS
# The card had more free VRAM the day the baseline ran than the shared fixture
# assumes, which is exactly why the two disagree by one block.
BASELINE_VRAM = 5_197_422_592


def _with_decomposition(
    *,
    method: TransformerBlockWeightDecompositionMethod = (
        TransformerBlockWeightDecompositionMethod.CONFIG_PARAMETER_DECOMPOSITION
    ),
    vram_bytes: int | None = None,
    include_component_estimates: bool = False,
) -> MemoryEstimate:
    estimate = qwen_ctx4096_estimate(with_runtime_recommendation=False)
    weights = estimate.weights.model_copy(
        update={
            "transformer_block_decomposition": TransformerBlockWeightDecomposition(
                total_weight_bytes=QWEN_WEIGHTS,
                estimated_transformer_block_weight_bytes=QWEN_BLOCK_WEIGHTS,
                estimated_non_block_weight_bytes=QWEN_NON_BLOCK,
                estimated_embedding_weight_bytes=(
                    QWEN_EMBEDDINGS if include_component_estimates else None
                ),
                estimated_output_head_weight_bytes=(
                    QWEN_OUTPUT_HEAD if include_component_estimates else None
                ),
                estimated_bytes_per_transformer_block=QWEN_BYTES_PER_BLOCK,
                total_transformer_blocks=QWEN_BLOCKS,
                method=method,
            )
        }
    )
    updates: dict[str, object] = {"weights": weights}
    if vram_bytes is not None:
        updates["assessment"] = estimate.assessment.model_copy(
            update={"available_vram_bytes": vram_bytes}
        )
    return estimate.model_copy(update=updates)


def test_without_a_decomposition_uses_the_uniform_weight_fallback() -> None:
    """No usable weight split is known, so every byte is charged to a block."""
    plan = pick_gpu_layers(qwen_ctx4096_estimate(with_runtime_recommendation=False))
    assert plan.n_gpu_layers == 23
    assert plan.confidence is EstimationConfidence.MEDIUM


def test_launch_policy_number_is_not_the_hardware_fit_block_count() -> None:
    """The policy must not equal 18 or 19 (HFA) — proves no blocks->layers wiring."""
    plan = pick_gpu_layers(qwen_ctx4096_estimate(with_runtime_recommendation=False))
    assert plan.n_gpu_layers not in {18, 19}


def test_an_unsplit_aggregate_is_charged_whole() -> None:
    """Without the embedding/output-head split there is nothing finer to use.

    Records written before the split existed, and architectures the config
    parser cannot decompose, land here. Charging all of it is the conservative
    reading and keeps those estimates where they were.
    """
    plan = pick_gpu_layers(_with_decomposition())

    assert plan.n_gpu_layers == 23
    assert any("conservative bound" in reason for reason in plan.reasons)


def test_component_estimates_do_not_relax_the_conservative_non_block_budget() -> None:
    """Parameter-projected components are not proof of llama.cpp placement."""
    aggregate = pick_gpu_layers(_with_decomposition())
    components = pick_gpu_layers(
        _with_decomposition(include_component_estimates=True)
    )

    assert components.n_gpu_layers == aggregate.n_gpu_layers
    assert any("conservative bound" in reason for reason in components.reasons)


def test_measured_baseline_does_not_turn_component_estimates_into_a_mapping() -> None:
    """B001-R4 is evidence, not a config-bytes-to-runtime-units mapping."""
    plan = pick_gpu_layers(
        _with_decomposition(
            vram_bytes=BASELINE_VRAM,
            include_component_estimates=True,
        )
    )

    assert 0 < plan.n_gpu_layers < QWEN_BLOCKS


def test_an_unsplit_decomposition_never_loosens_the_budget() -> None:
    """The aggregate path stays no less conservative than knowing nothing."""
    blind = pick_gpu_layers(qwen_ctx4096_estimate(with_runtime_recommendation=False))
    informed = pick_gpu_layers(_with_decomposition())

    assert informed.n_gpu_layers <= blind.n_gpu_layers


def test_the_kv_cache_is_charged_in_proportion_to_the_offload() -> None:
    """Only the offloaded blocks keep their KV in VRAM.

    Subtracting the whole cache up front prices a partial offload as if it were
    a full one. With a cache this large the difference is a whole block.
    """
    estimate = _with_decomposition()
    fat_kv = estimate.kv_cache.model_copy(
        update={
            "component": estimate.kv_cache.component.model_copy(
                update={"bytes": 2 * 1024**3}
            )
        }
    )
    estimate = estimate.model_copy(update={"kv_cache": fat_kv})

    plan = pick_gpu_layers(estimate)

    budget = QWEN_VRAM - estimate.inference_configuration.device_reserve_bytes - 256 * 1024**2
    charged_whole_cache = (budget - 2 * 1024**3) // QWEN_BYTES_PER_BLOCK
    assert plan.n_gpu_layers > charged_whole_cache
    assert 0 < plan.n_gpu_layers < QWEN_BLOCKS


def test_offload_required_does_not_infer_minus_one_from_block_counts() -> None:
    """Runtime offload units are not transformer blocks."""
    plan = pick_gpu_layers(_with_decomposition(vram_bytes=BASELINE_VRAM))

    assert plan.n_gpu_layers != -1
    assert 0 < plan.n_gpu_layers < QWEN_BLOCKS


def test_offload_required_never_offers_minus_one_from_config_decomposition() -> None:
    plan = pick_gpu_layers(_with_decomposition(vram_bytes=12 * 1024**3))

    assert plan.n_gpu_layers != -1
    assert plan.n_gpu_layers <= QWEN_BLOCKS


def test_a_guessed_split_does_not_change_the_offload_only_rule() -> None:
    """``uniform_weight_fallback`` means the split is unknown, not absent."""
    plan = pick_gpu_layers(
        _with_decomposition(
            method=TransformerBlockWeightDecompositionMethod.UNIFORM_WEIGHT_FALLBACK,
            vram_bytes=12 * 1024**3,
        )
    )

    assert plan.n_gpu_layers != -1


def test_no_vram_means_no_offload() -> None:
    plan = pick_gpu_layers(_with_decomposition(vram_bytes=0))

    assert plan.n_gpu_layers == 0
