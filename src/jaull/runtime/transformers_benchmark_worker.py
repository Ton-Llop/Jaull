"""Isolated worker for one Transformers/PyTorch benchmark run."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
import traceback
from typing import Any, cast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--torch-dtype")
    parser.add_argument("--prefill-sizes", default="128,512")
    parser.add_argument("--generation-sizes", default="128")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        payload = _benchmark(args)
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        _emit({"success": False, "error": str(exc)})
        return 1

    _emit({"success": True, **payload})
    return 0


def _benchmark(args: argparse.Namespace) -> dict[str, object]:
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

    load_start = time.perf_counter()
    tokenizer = auto_tokenizer.from_pretrained(args.model_ref, **tokenizer_kwargs)
    model = auto_model.from_pretrained(args.model_ref, **model_kwargs)
    if not use_accelerate_device_map:
        target = "cuda" if device_map in {"cuda", "hip"} else "cpu"
        model.to(target)
    model_load_seconds = time.perf_counter() - load_start

    input_device = _input_device(model)
    warmup_start = time.perf_counter()
    _generate_once(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        prompt="Benchmark warmup.",
        input_device=input_device,
        max_new_tokens=1,
    )
    warmup_seconds = time.perf_counter() - warmup_start

    repetitions = max(1, int(args.repetitions))
    measurements: list[dict[str, object]] = []
    for target_tokens in _sizes(args.prefill_sizes):
        prompt = _prompt_for_token_budget(tokenizer, target_tokens)
        values: list[float] = []
        actual_tokens = 0
        for _ in range(repetitions):
            duration, input_tokens, _generated_tokens = _generate_once(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                input_device=input_device,
                max_new_tokens=1,
            )
            actual_tokens = input_tokens
            values.append(input_tokens / duration if duration > 0 else 0.0)
        measurements.append(_measurement("prefill", actual_tokens, values))

    last_generation_latency: float | None = None
    for target_tokens in _sizes(args.generation_sizes):
        values = []
        actual_generated = 0
        latencies: list[float] = []
        for _ in range(repetitions):
            duration, _input_tokens, generated_tokens = _generate_once(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt="Benchmark generation.",
                input_device=input_device,
                max_new_tokens=target_tokens,
            )
            actual_generated = generated_tokens
            latencies.append(duration)
            values.append(generated_tokens / duration if duration > 0 else 0.0)
        last_generation_latency = statistics.mean(latencies) if latencies else None
        measurements.append(_measurement("generation", actual_generated, values))

    return {
        "methodology": "transformers_generate_steady_state_v1",
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "time_to_first_token_seconds": None,
        "generation_latency_seconds": last_generation_latency,
        "measurements": measurements,
    }


def _generate_once(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    input_device: object,
    max_new_tokens: int,
) -> tuple[float, int, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    duration = time.perf_counter() - start
    output_tokens = int(output_ids[0].shape[-1])
    return duration, input_tokens, max(0, output_tokens - input_tokens)


def _prompt_for_token_budget(tokenizer: Any, target_tokens: int) -> str:
    text = "benchmark"
    while _token_count(tokenizer, text) < target_tokens:
        text = f"{text} benchmark"
    return text


def _token_count(tokenizer: Any, text: str) -> int:
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[-1])


def _measurement(kind: str, tokens: int, values: list[float]) -> dict[str, object]:
    mean = statistics.mean(values) if values else 0.0
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    label = ("pp" if kind == "prefill" else "tg") + str(tokens)
    return {
        "kind": kind,
        "tokens": tokens,
        "mean_tokens_per_second": mean,
        "stddev_tokens_per_second": stddev,
        "source_label": label,
    }


def _sizes(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        stripped = item.strip()
        if stripped:
            values.append(max(1, int(stripped)))
    return values or [1]


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
