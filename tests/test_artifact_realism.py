from __future__ import annotations

from jaull.discovery.models import EvaluatedCandidate, ModelCandidate
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.recommendation.scoring import artifact_realism


def _config(
    quantization: str | None = None, precision: WeightPrecision | None = None
) -> InferenceConfiguration:
    return InferenceConfiguration(
        context_length=4096,
        target_device=TargetDevice.AUTO,
        quantization=quantization,
        precision=precision,
    )


def test_real_gguf_variant_scores_full_realism() -> None:
    ev = EvaluatedCandidate(
        candidate=ModelCandidate(repo_id="user/model-GGUF"),
        selected_configuration=_config(quantization="Q4_K_M"),
    )
    assert artifact_realism(ev) == 1.0


def test_theoretical_int4_without_tag_drops_realism() -> None:
    ev = EvaluatedCandidate(
        candidate=ModelCandidate(repo_id="user/model", tags=["text-generation"]),
        selected_configuration=_config(precision=WeightPrecision.INT4),
    )
    assert artifact_realism(ev) == 0.4


def test_bnb_4bit_tag_confirms_real_quant() -> None:
    ev = EvaluatedCandidate(
        candidate=ModelCandidate(repo_id="user/model", tags=["bnb-4bit"]),
        selected_configuration=_config(precision=WeightPrecision.INT4),
    )
    assert artifact_realism(ev) == 1.0


def test_native_fp16_transformers_scores_native_dtype() -> None:
    ev = EvaluatedCandidate(
        candidate=ModelCandidate(repo_id="user/model"),
        selected_configuration=_config(precision=WeightPrecision.FLOAT16),
    )
    assert artifact_realism(ev) == 0.75
