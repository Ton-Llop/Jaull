"""The one place that knows how a quantized estimate becomes a Transformers load.

``WeightPrecision.INT4`` and ``INT8`` are not ``torch_dtype`` values. Passing
``torch_dtype=torch.int8`` to ``from_pretrained`` does not quantize anything: it
asks for a tensor dtype the loader cannot use for weights, so the model either
fails to load or loads at a size that has nothing to do with the estimate. The
supported mechanism in Transformers is ``quantization_config``.

This module holds both halves of that translation so the recommendation builder,
the runner, and the two workers cannot drift apart:

* :func:`quantization_mode` turns an estimated precision into the wire value
  that travels on a :class:`RuntimeFlag` and on the worker command line;
* :func:`build_quantization_config` turns that wire value back into a real
  ``BitsAndBytesConfig`` inside a worker.

**This module must stay importable on a bare Python.** The workers run as
``python -m jaull.runtime.transformers_worker`` inside the user's *PyTorch*
environment, which is not Jaull's virtualenv and is not required to have
Pydantic installed. So nothing here imports ``jaull.domain``: the precision is
taken as its plain ``str`` value, which is what ``WeightPrecision`` serialises to
anyway. Importing the domain here made the worker die with
``ModuleNotFoundError: pydantic`` before it could emit its structured error.

``bitsandbytes`` is deliberately **not** a Jaull dependency: it is CUDA-centric
and awkward on Windows, and a quantized Transformers plan is already reported as
a theoretical artifact. So a worker asked for one without the library installed
fails with an explicit message instead of quietly loading unquantized weights
and reporting the result as though it were the estimated configuration.

Even with a real ``BitsAndBytesConfig`` the loaded footprint is not exactly
``parameters x 0.5 bytes``: bitsandbytes keeps some modules (embeddings, the
output head, layer norms) at a higher precision. The estimate stays an
approximation, which is why a quantized plan is reported as theoretical rather
than confirmed. See https://huggingface.co/docs/transformers/quantization/bitsandbytes.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Wire values for the ``quantization`` runtime flag and ``--quantization``,
#: keyed by ``WeightPrecision`` *values* so this module needs no domain import.
QUANTIZATION_MODES: dict[str, str] = {
    "int4": "4bit",
    "int8": "8bit",
}

#: The dtype bitsandbytes computes in while the weights stay quantized.
COMPUTE_DTYPE = "torch.float16"

MISSING_LIBRARY_MESSAGE = (
    "This configuration needs the 'bitsandbytes' package, which Jaull does not "
    "install. Install it in the PyTorch environment, or pick a plan whose "
    "precision loads without on-the-fly quantization. Jaull will not run an "
    "unquantized model in place of a quantized estimate: the measurement would "
    "not describe the configuration that was predicted."
)


def quantization_mode(precision: Any) -> str | None:
    """The wire value for ``precision``, or ``None`` when it loads as a dtype.

    Accepts a ``WeightPrecision`` or its plain string value; only the value is
    read, so callers in the workers never need the enum.
    """
    if precision is None:
        return None
    value = getattr(precision, "value", precision)
    if not isinstance(value, str):
        return None
    return QUANTIZATION_MODES.get(value)


def requires_bitsandbytes(mode_or_precision: Any) -> bool:
    """Whether a runtime mode requires the bitsandbytes loader path."""
    value = getattr(mode_or_precision, "value", mode_or_precision)
    return value in set(QUANTIZATION_MODES) or value in set(QUANTIZATION_MODES.values())


def build_quantization_config(mode: str | None) -> Any | None:
    """Return a ``BitsAndBytesConfig`` for ``mode``, or ``None`` when unset.

    Raises ``RuntimeError`` when the mode is unknown, or when bitsandbytes is
    not importable — never silently degrades to an unquantized load.
    """
    if not mode:
        return None
    if mode not in set(QUANTIZATION_MODES.values()):
        raise RuntimeError(
            f"Unknown quantization mode {mode!r}; "
            f"expected one of {sorted(set(QUANTIZATION_MODES.values()))}."
        )

    try:
        importlib.import_module("bitsandbytes")
    except ImportError as exc:
        raise RuntimeError(MISSING_LIBRARY_MESSAGE) from exc

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    config_cls = transformers.BitsAndBytesConfig

    if mode == "4bit":
        return config_cls(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return config_cls(load_in_8bit=True)


__all__ = [
    "COMPUTE_DTYPE",
    "MISSING_LIBRARY_MESSAGE",
    "QUANTIZATION_MODES",
    "build_quantization_config",
    "quantization_mode",
    "requires_bitsandbytes",
]
