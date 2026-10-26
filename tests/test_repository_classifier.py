from __future__ import annotations

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.model import ModelFile
from jaull.huggingface.classifiers import classify_repository


def _files(*paths: str) -> list[ModelFile]:
    return [ModelFile(path=p, size_bytes=1024) for p in paths]


def test_classifies_transformers_repository() -> None:
    files = _files(
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model.safetensors.index.json",
    )
    result = classify_repository(files)
    assert result.primary_type is RepositoryType.TRANSFORMERS
    assert RepositoryType.TRANSFORMERS in result.detected_types
    assert Format.SAFETENSORS in result.formats


def test_classifies_gguf_repository_with_variants() -> None:
    files = [
        ModelFile(path="README.md", size_bytes=1024),
        ModelFile(path="model-Q4_K_M.gguf", size_bytes=4_000_000_000),
        ModelFile(path="model-Q5_K_M.gguf", size_bytes=5_000_000_000),
        ModelFile(path="model-Q8_0.gguf", size_bytes=8_000_000_000),
    ]
    result = classify_repository(files)
    assert result.primary_type is RepositoryType.GGUF
    quantizations = {v.quantization for v in result.gguf_variants}
    assert quantizations == {"Q4_K_M", "Q5_K_M", "Q8_0"}
    assert Format.GGUF in result.formats


def test_gguf_multipart_variants_are_grouped() -> None:
    files = [
        ModelFile(path="model.Q4_K_M-00001-of-00002.gguf", size_bytes=1),
        ModelFile(path="model.Q4_K_M-00002-of-00002.gguf", size_bytes=1),
    ]
    result = classify_repository(files)
    assert len(result.gguf_variants) == 1
    assert result.gguf_variants[0].quantization == "Q4_K_M"
    assert len(result.gguf_variants[0].files) == 2


def test_a_quantization_published_whole_and_as_parts_is_counted_once() -> None:
    """Regression found on a B60 box: Qwen's GGUF repos publish each quantization both as one
    file and as its split parts, and the weights were counted twice (Q8_0 7B: 15 GiB, not 7.5)."""
    files = [
        ModelFile(path="m-q8_0.gguf", size_bytes=7_540),
        ModelFile(path="m-q8_0-00001-of-00003.gguf", size_bytes=3_710),
        ModelFile(path="m-q8_0-00002-of-00003.gguf", size_bytes=3_670),
        ModelFile(path="m-q8_0-00003-of-00003.gguf", size_bytes=160),
        ModelFile(path="m-q5_0-00001-of-00002.gguf", size_bytes=3_730),  # parts only
        ModelFile(path="m-q5_0-00002-of-00002.gguf", size_bytes=1_220),
        ModelFile(path="m-q2_k.gguf", size_bytes=2_810),  # one file only
    ]

    from jaull.domain.gguf_variants import preferred_gguf_variants

    variants = {
        v.quantization: v
        for v in preferred_gguf_variants(classify_repository(files).gguf_variants)
    }

    assert variants["Q8_0"].total_bytes == 7_540
    assert [f.path for f in variants["Q8_0"].files] == ["m-q8_0.gguf"]
    assert variants["Q5_0"].total_bytes == 4_950  # parts that are all there is still add up
    assert len(variants["Q5_0"].files) == 2
    assert variants["Q2_K"].total_bytes == 2_810


def test_classifies_diffusers_repository() -> None:
    files = _files(
        "model_index.json",
        "scheduler/scheduler_config.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    )
    result = classify_repository(files)
    assert result.primary_type is RepositoryType.DIFFUSERS
    assert RepositoryType.DIFFUSERS in result.detected_types


def test_classifies_onnx_repository() -> None:
    files = _files("model.onnx", "config.json")
    result = classify_repository(files)
    assert RepositoryType.ONNX in result.detected_types
    assert result.primary_type is RepositoryType.ONNX
    assert Format.ONNX in result.formats


def test_classifies_adapter_repository() -> None:
    files = _files("adapter_config.json", "adapter_model.safetensors")
    result = classify_repository(files)
    assert result.primary_type is RepositoryType.ADAPTER
    assert RepositoryType.ADAPTER in result.detected_types


def test_unknown_when_nothing_matches() -> None:
    files = _files("README.md", "LICENSE")
    result = classify_repository(files)
    assert result.primary_type is RepositoryType.UNKNOWN
    assert RepositoryType.UNKNOWN in result.detected_types


def test_transformers_plus_gguf_is_detected_as_both() -> None:
    files = _files(
        "config.json",
        "model.safetensors",
        "model-Q4_K_M.gguf",
    )
    result = classify_repository(files)
    assert RepositoryType.TRANSFORMERS in result.detected_types
    assert RepositoryType.GGUF in result.detected_types
    # Transformers wins over gguf in the priority order.
    assert result.primary_type is RepositoryType.TRANSFORMERS
