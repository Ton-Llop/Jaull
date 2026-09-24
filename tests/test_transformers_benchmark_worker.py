"""The benchmark worker's load path, which nothing exercised.

`transformers_benchmark_worker.py` sat at 0% coverage while gaining the
quantization branch: `build_quantization_config`, the `place_at_load` decision
and `_load_device`. The runner-side test only checks that `--quantization`
reaches the command line; what the *loader* receives was unverified, and that is
the half where a quantized plan silently becomes an unquantized measurement.

Everything here fakes torch and transformers. The worker runs in the user's
PyTorch environment, so the tests must not need one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import types
from typing import Any

import pytest

from jaull.runtime import transformers_benchmark_worker as worker
from jaull.runtime import transformers_quantization as quant


class _FakeTensor:
    shape = (1, 4)

    def to(self, device: Any) -> _FakeTensor:
        del device
        return self

    def argmax(self, dim: int = -1) -> _FakeTensor:
        del dim
        return self

    def __getitem__(self, item: Any) -> _FakeTensor:
        del item
        return self


class _FakeOutputs:
    def __init__(self) -> None:
        self.past_key_values = object()
        self.logits = _FakeTensor()


class _FakeParameter:
    device = "cpu"


class _FakeModel:
    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def __call__(self, **kwargs: Any) -> _FakeOutputs:
        del kwargs
        return _FakeOutputs()

    def to(self, target: str) -> None:
        self.moved_to.append(target)

    def parameters(self) -> Any:
        return iter([_FakeParameter()])


class _FakeAutoModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.model = _FakeModel()

    def from_pretrained(self, model_ref: str, **kwargs: Any) -> _FakeModel:
        del model_ref
        self.kwargs = kwargs
        return self.model


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        del text, return_tensors
        return {"input_ids": _FakeTensor()}

    def from_pretrained(self, model_ref: str, **kwargs: Any) -> _FakeTokenizer:
        del model_ref, kwargs
        return self


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install(
    monkeypatch: pytest.MonkeyPatch, *, bitsandbytes: bool = True
) -> _FakeAutoModel:
    auto_model = _FakeAutoModel()
    torch = types.SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        float16="torch.float16-sentinel",
        float32="torch.float32-sentinel",
    )
    transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer(),
        AutoModelForCausalLM=auto_model,
        BitsAndBytesConfig=_FakeBitsAndBytesConfig,
    )
    real_import = worker.importlib.import_module

    def fake_import(name: str) -> Any:
        if name == "torch":
            return torch
        if name == "transformers":
            return transformers
        if name == "bitsandbytes":
            if not bitsandbytes:
                raise ImportError("No module named 'bitsandbytes'")
            return types.SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)
    monkeypatch.setattr(quant.importlib, "import_module", fake_import)
    # The prompt builder loops until the tokenizer reports enough tokens, and the
    # fake always reports four.
    monkeypatch.setattr(worker, "_prompt_for_token_budget", lambda tok, n: "benchmark")
    return auto_model


def _args(**overrides: Any) -> argparse.Namespace:
    data: dict[str, Any] = {
        "model_ref": "org/model",
        "revision": None,
        "device_map": "cuda",
        "torch_dtype": None,
        "quantization": None,
        "prefill_sizes": "4",
        "generation_sizes": "2",
        "repetitions": 1,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


# ---------------------------------------------------------------------------
# What the loader receives
# ---------------------------------------------------------------------------


def test_a_quantized_benchmark_is_placed_at_load_not_moved_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bitsandbytes rejects `.to()` on a 4-bit model.

    Benchmarking a model moved after loading would also time a different
    placement from the one the plan describes.
    """
    auto_model = _install(monkeypatch)

    worker._benchmark(_args(quantization="4bit"))

    assert isinstance(auto_model.kwargs["quantization_config"], _FakeBitsAndBytesConfig)
    assert auto_model.kwargs["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "torch.float16-sentinel",
    }
    assert auto_model.kwargs["device_map"] == "cuda"
    assert auto_model.model.moved_to == []


def test_an_unquantized_benchmark_keeps_the_existing_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto_model = _install(monkeypatch)

    worker._benchmark(_args(torch_dtype="torch.float16"))

    assert "quantization_config" not in auto_model.kwargs
    assert "device_map" not in auto_model.kwargs
    assert auto_model.kwargs["torch_dtype"] == "torch.float16-sentinel"
    assert auto_model.model.moved_to == ["cuda"]


def test_device_map_auto_is_passed_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto_model = _install(monkeypatch)

    worker._benchmark(_args(device_map="auto"))

    assert auto_model.kwargs["device_map"] == "auto"
    assert auto_model.model.moved_to == []


@pytest.mark.parametrize(
    ("device_map", "expected"),
    [("cuda", "cuda"), ("hip", "cuda"), ("cpu", "cpu"), ("something", "cpu")],
)
def test_load_device_maps_the_backend_alias(device_map: str, expected: str) -> None:
    """HIP reaches PyTorch through the CUDA device namespace."""
    assert worker._load_device(device_map) == expected


# ---------------------------------------------------------------------------
# Refusing rather than measuring the wrong thing
# ---------------------------------------------------------------------------


def test_a_missing_bitsandbytes_aborts_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto_model = _install(monkeypatch, bitsandbytes=False)

    with pytest.raises(RuntimeError, match="bitsandbytes"):
        worker._benchmark(_args(quantization="4bit"))

    assert auto_model.kwargs == {}, "the model must not be loaded"


def test_the_failure_is_emitted_as_a_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The runner parses stdout as JSON; a traceback alone is not a result."""
    _install(monkeypatch, bitsandbytes=False)

    exit_code = worker.main(
        [
            "--model-ref",
            "org/model",
            "--device-map",
            "cuda",
            "--quantization",
            "4bit",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["success"] is False
    assert "bitsandbytes" in payload["error"]


# ---------------------------------------------------------------------------
# The payload the runner reads back
# ---------------------------------------------------------------------------


def test_a_successful_benchmark_reports_the_expected_methodology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`local_evidence` matches records on this exact string."""
    _install(monkeypatch)

    payload = worker._benchmark(_args())

    assert payload["methodology"] == "transformers_isolated_inference_v2"
    kinds = {m["kind"] for m in payload["measurements"]}
    assert kinds == {"prefill", "generation"}
    assert payload["model_load_seconds"] is not None
    assert payload["time_to_first_token_seconds"] is not None
