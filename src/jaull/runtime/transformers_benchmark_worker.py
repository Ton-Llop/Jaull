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
    _decode_once(
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
        durations: list[float] = []
        actual_tokens = 0
        for _ in range(repetitions):
            duration, input_tokens = _prefill_once(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                input_device=input_device,
            )
            actual_tokens = input_tokens
            durations.append(duration)
            values.append(input_tokens / duration if duration > 0 else 0.0)
        measurements.append(_measurement("prefill", actual_tokens, values, durations))

    last_generation_latency: float | None = None
    last_generation_latency_stddev: float | None = None
    ttft_values: list[float] = []
    for target_tokens in _sizes(args.generation_sizes):
        values = []
        actual_generated = 0
        latencies: list[float] = []
        for _ in range(repetitions):
            ttft = _time_to_first_token(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt="Benchmark generation.",
                input_device=input_device,
            )
            duration, generated_tokens = _decode_once(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt="Benchmark generation.",
                input_device=input_device,
                max_new_tokens=target_tokens,
            )
            ttft_values.append(ttft)
            actual_generated = generated_tokens
            latencies.append(duration)
            values.append(generated_tokens / duration if duration > 0 else 0.0)
        last_generation_latency = statistics.mean(latencies) if latencies else None
        last_generation_latency_stddev = _stddev(latencies)
        measurements.append(
            _measurement("generation", actual_generated, values, latencies)
        )

    return {
        "methodology": "transformers_isolated_inference_v2",
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "time_to_first_token_seconds": (
            statistics.mean(ttft_values) if ttft_values else None
        ),
        "time_to_first_token_stddev_seconds": _stddev(ttft_values),
        "generation_latency_seconds": last_generation_latency,
        "generation_latency_stddev_seconds": last_generation_latency_stddev,
        "measurements": measurements,
    }


def _prefill_once(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    input_device: object,
) -> tuple[float, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        model(**inputs, use_cache=True)
    duration = time.perf_counter() - start
    return duration, input_tokens


def _decode_once(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    input_device: object,
    max_new_tokens: int,
) -> tuple[float, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated_tokens = 0
        start = time.perf_counter()
        for _ in range(max(1, int(max_new_tokens))):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            generated_tokens += 1
    duration = time.perf_counter() - start
    return duration, generated_tokens


def _time_to_first_token(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    input_device: object,
) -> float:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        outputs.logits[:, -1:].argmax(dim=-1)
    return time.perf_counter() - start


def _prompt_for_token_budget(tokenizer: Any, target_tokens: int) -> str:
    text = "benchmark"
    while _token_count(tokenizer, text) < target_tokens:
        text = f"{text} benchmark"
    return text


def _token_count(tokenizer: Any, text: str) -> int:
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[-1])


def _measurement(
    kind: str,
    tokens: int,
    values: list[float],
    durations: list[float],
) -> dict[str, object]:
    mean = statistics.mean(values) if values else 0.0
    stddev = _stddev(values) or 0.0
    duration_mean = statistics.mean(durations) if durations else None
    label = ("pp" if kind == "prefill" else "tg") + str(tokens)
    return {
        "kind": kind,
        "tokens": tokens,
        "mean_tokens_per_second": mean,
        "stddev_tokens_per_second": stddev,
        "mean_duration_seconds": duration_mean,
        "stddev_duration_seconds": _stddev(durations),
        "source_label": label,
    }


def _stddev(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return statistics.stdev(values)


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
