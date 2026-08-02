from __future__ import annotations

from local_ai_check.domain.enums import Format, RepositoryType
from local_ai_check.domain.inference import WeightPrecision
from local_ai_check.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from local_ai_check.estimator import weights
from local_ai_check.estimator.policies import bytes_per_parameter


def _analysis(files: list[ModelFile], primary: RepositoryType) -> ModelAnalysis:
    classification = RepositoryClassification(
        primary_type=primary,
        detected_types={primary},
        formats=(
            {Format.SAFETENSORS} if primary is RepositoryType.TRANSFORMERS else set()
        ),
        gguf_variants=[],
    )
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="user/model"),
        files=files,
        classification=classification,
        config=None,
        relevant_files=[],
        total_size_bytes=None,
        warnings=[],
    )


def test_bytes_per_parameter_table_matches_spec() -> None:
    assert bytes_per_parameter(WeightPrecision.FLOAT32) == 4.0
    assert bytes_per_parameter(WeightPrecision.FLOAT16) == 2.0
    assert bytes_per_parameter(WeightPrecision.BFLOAT16) == 2.0
    assert bytes_per_parameter(WeightPrecision.INT8) == 1.0
    assert bytes_per_parameter(WeightPrecision.INT4) == 0.5


def test_weights_from_gguf_returns_exact_size() -> None:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.Q4_K_M.gguf", size_bytes=4_921_000_000)],
        total_bytes=4_921_000_000,
    )
    est = weights.estimate_weights_from_gguf(variant)
    assert est.component.bytes == 4_921_000_000
    assert est.component.source.value == "exact"
    assert est.component.confidence.value == "high"
    assert est.gguf_variant == "Q4_K_M"


def test_weights_from_safetensors_uses_requested_precision() -> None:
    summary = SafetensorsSummary(
        total_parameters=1_000_000_000,
        parameters_by_dtype={"BF16": 1_000_000_000},
    )
    est = weights.estimate_weights_from_safetensors(
        summary=summary,
        requested_precision=WeightPrecision.INT8,
        config=None,
    )
    assert est.component.bytes == 1_000_000_000  # 1B * 1 byte
    assert est.precision is WeightPrecision.INT8
    assert est.num_parameters == 1_000_000_000


def test_weights_from_safetensors_falls_back_to_dominant_dtype() -> None:
    summary = SafetensorsSummary(
        total_parameters=1_000_000_000,
        parameters_by_dtype={"F16": 1_000_000_000},
    )
    est = weights.estimate_weights_from_safetensors(
        summary=summary,
        requested_precision=None,
        config=None,
    )
    assert est.precision is WeightPrecision.FLOAT16
    assert est.component.bytes == 2_000_000_000


def test_weights_from_files_sums_safetensors_and_bin() -> None:
    analysis = _analysis(
        files=[
            ModelFile(path="config.json", size_bytes=1024),
            ModelFile(path="model-00001-of-00002.safetensors", size_bytes=3_000_000_000),
            ModelFile(path="model-00002-of-00002.safetensors", size_bytes=3_000_000_000),
            ModelFile(path="tokenizer.json", size_bytes=2048),
        ],
        primary=RepositoryType.TRANSFORMERS,
    )
    est = weights.estimate_weights_from_files(
        analysis=analysis,
        requested_precision=None,
        config=None,
    )
    assert est.component.bytes == 6_000_000_000
    assert est.component.source.value == "derived"
    assert est.precision is WeightPrecision.FLOAT16


def test_weights_from_files_scales_when_precision_differs() -> None:
    analysis = _analysis(
        files=[
            ModelFile(path="model.safetensors", size_bytes=8_000_000_000),
        ],
        primary=RepositoryType.TRANSFORMERS,
    )
    config = ModelConfig(torch_dtype="bfloat16")
    est = weights.estimate_weights_from_files(
        analysis=analysis,
        requested_precision=WeightPrecision.INT4,
        config=config,
    )
    # 8B bytes at bf16 (2 bytes/param) -> 4B params -> at int4 (0.5) = 2B bytes.
    assert est.component.bytes == 2_000_000_000
    assert est.precision is WeightPrecision.INT4


def test_weights_from_files_unknown_when_no_weights_present() -> None:
    analysis = _analysis(
        files=[ModelFile(path="README.md", size_bytes=100)],
        primary=RepositoryType.TRANSFORMERS,
    )
    est = weights.estimate_weights_from_files(
        analysis=analysis, requested_precision=None, config=None
    )
    assert est.component.bytes is None
    assert est.component.source.value == "unknown"
