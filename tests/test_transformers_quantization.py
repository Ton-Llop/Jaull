"""The quantized load path itself, not just the flag that describes it.

Three things have to hold, and none of them is visible from the runtime flags:

* the workers must stay importable in the user's PyTorch environment, which is
  not Jaull's virtualenv and need not have Pydantic;
* a requested mode must reach ``from_pretrained`` as a real quantization config,
  with the device applied at load time rather than by a later ``.to()`` — which
  bitsandbytes rejects on a 4-bit model;
* a missing ``bitsandbytes`` must fail loudly, never degrade to an unquantized
  load that would be measured as though it were the estimated configuration.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from jaull.runtime import transformers_quantization as quant

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# Worker isolation
# ---------------------------------------------------------------------------


def test_the_helper_imports_without_pydantic() -> None:
    """The workers run under an interpreter Jaull does not control.

    Importing ``jaull.domain`` here once made the worker die with
    ``ModuleNotFoundError: pydantic`` before it could emit its structured error.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['pydantic'] = None;"
            " from jaull.runtime.transformers_quantization import quantization_mode;"
            " print(quantization_mode('int4'))",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4bit"


# ---------------------------------------------------------------------------
# The wire vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("int4", "4bit"), ("int8", "8bit"), ("float16", None), ("float32", None)],
)
def test_quantization_mode_maps_only_the_quantized_precisions(
    precision: str, expected: str | None
) -> None:
    assert quant.quantization_mode(precision) == expected


def test_quantization_mode_accepts_the_enum_without_importing_it() -> None:
    """Callers on the recommendation side pass ``WeightPrecision`` itself."""
    from jaull.domain.inference import WeightPrecision

    assert quant.quantization_mode(WeightPrecision.INT4) == "4bit"
    assert quant.quantization_mode(WeightPrecision.FLOAT16) is None


# ---------------------------------------------------------------------------
# Building the config
# ---------------------------------------------------------------------------


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, bitsandbytes: bool) -> None:
    torch = types.SimpleNamespace(float16="torch.float16-sentinel")
    transformers = types.SimpleNamespace(BitsAndBytesConfig=_FakeBitsAndBytesConfig)
    real_import = quant.importlib.import_module

    def fake_import(name: str) -> Any:
        if name == "bitsandbytes":
            if not bitsandbytes:
                raise ImportError("No module named 'bitsandbytes'")
            return types.SimpleNamespace()
        if name == "torch":
            return torch
        if name == "transformers":
            return transformers
        return real_import(name)

    monkeypatch.setattr(quant.importlib, "import_module", fake_import)


def test_four_bit_asks_for_load_in_4bit_with_a_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, bitsandbytes=True)

    config = quant.build_quantization_config("4bit")

    assert isinstance(config, _FakeBitsAndBytesConfig)
    assert config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "torch.float16-sentinel",
    }


def test_eight_bit_asks_for_load_in_8bit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch, bitsandbytes=True)

    config = quant.build_quantization_config("8bit")

    assert isinstance(config, _FakeBitsAndBytesConfig)
    assert config.kwargs == {"load_in_8bit": True}


def test_a_missing_bitsandbytes_fails_instead_of_loading_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, bitsandbytes=False)

    with pytest.raises(RuntimeError) as excinfo:
        quant.build_quantization_config("4bit")

    assert "bitsandbytes" in str(excinfo.value)
    # The reason matters: silence here would produce a measurement that does
    # not describe the predicted configuration.
    assert "not describe the configuration that was predicted" in str(excinfo.value)


def test_an_unknown_mode_is_refused_before_any_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(name: str) -> Any:
        raise AssertionError(f"should not have imported {name}")

    monkeypatch.setattr(quant.importlib, "import_module", explode)

    with pytest.raises(RuntimeError, match="Unknown quantization mode"):
        quant.build_quantization_config("3bit")


def test_no_mode_requested_is_not_an_error() -> None:
    assert quant.build_quantization_config(None) is None
    assert quant.build_quantization_config("") is None


# ---------------------------------------------------------------------------
# What the loader actually receives
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, target: str) -> None:
        self.moved_to.append(target)

    def parameters(self) -> Any:
        return iter(())

    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        return [[1, 2, 3]]


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

    def decode(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "generated"

    def from_pretrained(self, model_ref: str, **kwargs: Any) -> _FakeTokenizer:
        del model_ref, kwargs
        return self


class _FakeTensor:
    shape = (1, 3)

    def to(self, device: Any) -> _FakeTensor:
        del device
        return self


def _run_worker_generate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_map: str,
    quantization: str | None,
) -> _FakeAutoModel:
    """Drive the real worker with a fake torch/transformers and report the load."""
    import argparse
    import contextlib

    from jaull.runtime import transformers_worker as worker

    auto_model = _FakeAutoModel()
    tokenizer = _FakeTokenizer()
    torch = types.SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        float16="torch.float16-sentinel",
    )
    transformers = types.SimpleNamespace(
        AutoTokenizer=tokenizer,
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
            return types.SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)
    monkeypatch.setattr(quant.importlib, "import_module", fake_import)
    monkeypatch.setattr(worker, "_prompt_for_model", lambda tok, prompt: prompt)
    monkeypatch.setattr(worker, "_decode_output", lambda *a, **k: "generated", raising=False)

    args = argparse.Namespace(
        model_ref="org/model",
        prompt="hi",
        revision=None,
        device_map=device_map,
        torch_dtype=None,
        quantization=quantization,
        max_new_tokens=4,
    )
    with contextlib.suppress(Exception):
        worker._generate(args)
    return auto_model


def test_a_quantized_load_is_placed_at_load_time_not_moved_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bitsandbytes rejects ``.to()`` on a 4-bit model, and the snippet passes
    ``device_map`` to ``from_pretrained``. Both halves must agree."""
    auto_model = _run_worker_generate(
        monkeypatch, device_map="cuda", quantization="4bit"
    )

    assert "quantization_config" in auto_model.kwargs
    assert auto_model.kwargs["device_map"] == "cuda"
    assert auto_model.model.moved_to == []


def test_an_unquantized_cuda_load_keeps_the_existing_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The float path is unchanged: load, then move."""
    auto_model = _run_worker_generate(monkeypatch, device_map="cuda", quantization=None)

    assert "quantization_config" not in auto_model.kwargs
    assert "device_map" not in auto_model.kwargs
    assert auto_model.model.moved_to == ["cuda"]
