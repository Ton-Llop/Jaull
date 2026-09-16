"""Isolated worker for one Transformers/PyTorch generation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from typing import Any, cast

from jaull.runtime.transformers_quantization import build_quantization_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--torch-dtype")
    parser.add_argument("--quantization")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    try:
        text = _generate(args)
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        _emit({"success": False, "error": str(exc)})
        return 1

    _emit({"success": True, "text": text})
    return 0


def _generate(args: argparse.Namespace) -> str:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    auto_tokenizer = cast(Any, transformers).AutoTokenizer
    auto_model = cast(Any, transformers).AutoModelForCausalLM

    tokenizer_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if args.revision:
        tokenizer_kwargs["revision"] = args.revision
        model_kwargs["revision"] = args.revision

    dtype = _torch_dtype(torch, args.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    # Raises when bitsandbytes is absent instead of loading unquantized weights
    # and reporting them as the estimated configuration.
    quantization_config = build_quantization_config(args.quantization)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    device_map = str(args.device_map or "cpu")
    # A quantized model is placed while it loads: bitsandbytes rejects a later
    # ``.to()``, and the snippet passes ``device_map`` to ``from_pretrained``
    # too. Loading first and moving afterwards would be a different contract
    # from the one the recommendation shows.
    place_at_load = device_map == "auto" or quantization_config is not None
    if place_at_load:
        model_kwargs["device_map"] = (
            "auto" if device_map == "auto" else _load_device(device_map)
        )

    tokenizer = auto_tokenizer.from_pretrained(args.model_ref, **tokenizer_kwargs)
    model = auto_model.from_pretrained(args.model_ref, **model_kwargs)
    if not place_at_load:
        model.to(_load_device(device_map))

    input_text = _prompt_for_model(tokenizer, args.prompt)
    inputs = tokenizer(input_text, return_tensors="pt")
    input_device = _input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_length:]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def _prompt_for_model(tokenizer: object, prompt: str) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template and callable(apply_chat_template):
        return str(
            apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompt


def _load_device(device_map: str) -> str:
    """The concrete device behind a non-``auto`` device map."""
    return "cuda" if device_map == "cuda" else "cpu"


def _torch_dtype(torch: object, value: str | None) -> object | None:
    if not value or value == "auto":
        return None
    name = value.removeprefix("torch.")
    dtype: object = getattr(torch, name)
    return dtype


def _input_device(model: Any) -> object:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
