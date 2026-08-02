from __future__ import annotations

from enum import StrEnum


class RepositoryType(StrEnum):
    TRANSFORMERS = "transformers"
    GGUF = "gguf"
    DIFFUSERS = "diffusers"
    ONNX = "onnx"
    ADAPTER = "adapter"
    UNKNOWN = "unknown"


class Format(StrEnum):
    SAFETENSORS = "safetensors"
    PYTORCH_BIN = "pytorch_bin"
    GGUF = "gguf"
    ONNX = "onnx"
    TFLITE = "tflite"
    OPENVINO = "openvino"


class DiagnosticStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
