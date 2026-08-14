"""Isolated worker for one Transformers/PyTorch generation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from typing import Any, cast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--torch-dtype")
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

    device_map = str(args.device_map or "cpu")
    use_accelerate_device_map = device_map == "auto"
    if use_accelerate_device_map:
        model_kwargs["device_map"] = "auto"

    tokenizer = auto_tokenizer.from_pretrained(args.model_ref, **tokenizer_kwargs)
    model = auto_model.from_pretrained(args.model_ref, **model_kwargs)
    if not use_accelerate_device_map:
        target = "cuda" if device_map == "cuda" else "cpu"
        model.to(target)

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
