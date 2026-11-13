"""Concurrency load test against an OpenAI-compatible llama-server.

For each concurrency level N it sends N chat requests at the same time, streams the
answers and records, per request: time to first token, generated tokens and decode
speed; per round: wall time and aggregate tokens per second. Standard library only.

    python3 scripts/concurrency-bench/load.py --url http://127.0.0.1:8081 \
        --levels 1 2 4 --repeats 3 --out results-np4.csv

Every request starts with a unique nonce, so the server cannot reuse a cached prompt
prefix from another request and every prefill is real. Token counts come from the
server's ``usage`` (or ``timings``) when it reports them; otherwise from the number of
streamed content chunks, which the summary marks as approximate.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import statistics
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field

PARAGRAPH = (
    "Local inference keeps data on the machine that owns it. A laboratory that analyses "
    "biological samples produces reports, spreadsheets and spectra every day, and the "
    "people who read them need short answers about dates, sample counts and values. "
)
QUESTION = (
    "Summarise the text above in five numbered points, then explain in one paragraph "
    "which risks a laboratory should consider before using a local language model."
)


@dataclass
class RequestResult:
    level: int
    repeat: int
    index: int
    ok: bool
    ttft_s: float | None = None
    total_s: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    tokens_exact: bool = False
    error: str = ""
    decode_tok_s: float | None = field(default=None)


def build_prompt(prompt_paragraphs: int) -> str:
    return f"[request {uuid.uuid4()}]\n\n" + PARAGRAPH * prompt_paragraphs + "\n\n" + QUESTION


def one_request(url: str, model: str, prompt: str, max_tokens: int, api_key: str | None,
                timeout: float, result: RequestResult, start_barrier: threading.Barrier) -> None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)
    chunks = 0
    start_barrier.wait()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        if result.ttft_s is None:
                            result.ttft_s = time.perf_counter() - started
                        chunks += 1
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    result.completion_tokens = usage["completion_tokens"]
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.tokens_exact = True
                timings = event.get("timings")
                if timings and not result.tokens_exact and timings.get("predicted_n") is not None:
                    result.completion_tokens = timings["predicted_n"]
                    result.prompt_tokens = timings.get("prompt_n")
                    result.tokens_exact = True
        result.total_s = time.perf_counter() - started
        if result.completion_tokens is None:
            result.completion_tokens = chunks
        decode_s = result.total_s - (result.ttft_s or 0.0)
        if result.completion_tokens and decode_s > 0:
            result.decode_tok_s = result.completion_tokens / decode_s
        result.ok = result.ttft_s is not None
        if not result.ok:
            result.error = "no content streamed"
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # A failed request is a result to record, not a reason to stop the round.
        result.total_s = time.perf_counter() - started
        result.error = f"{type(exc).__name__}: {exc}"


def run_round(
    args: argparse.Namespace, level: int, repeat: int,
) -> tuple[list[RequestResult], float]:
    barrier = threading.Barrier(level)
    results = [RequestResult(level=level, repeat=repeat, index=i, ok=False) for i in range(level)]
    threads = [
        threading.Thread(target=one_request, args=(
            args.url, args.model, build_prompt(args.prompt_paragraphs), args.max_tokens,
            args.api_key, args.timeout, results[i], barrier,
        ))
        for i in range(level)
    ]
    wall_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, time.perf_counter() - wall_start


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="local")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-paragraphs", type=int, default=12,
                        help="Repetitions of the base paragraph (~55 tokens each).")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", required=True, help="CSV with one row per request.")
    args = parser.parse_args()

    rows: list[RequestResult] = []
    print(f"{'N':>3} {'rep':>3} {'ok':>5} {'TTFT med s':>10} {'tok/s/user med':>14} "
          f"{'agg tok/s':>9} {'wall s':>7}")
    summary: dict[int, list[tuple[list[RequestResult], float]]] = {}
    for level in args.levels:
        for repeat in range(1, args.repeats + 1):
            results, wall = run_round(args, level, repeat)
            rows.extend(results)
            summary.setdefault(level, []).append((results, wall))
            ok = [r for r in results if r.ok]
            tokens = sum(r.completion_tokens or 0 for r in ok)
            print(f"{level:>3} {repeat:>3} {len(ok):>2}/{level:<2} "
                  f"{fmt(median([r.ttft_s for r in ok if r.ttft_s is not None])):>10} "
                  f"{fmt(median([r.decode_tok_s for r in ok if r.decode_tok_s])):>14} "
                  f"{fmt(tokens / wall if wall else None):>9} {fmt(wall, 1):>7}")

    print("\nPer level (medians over all requests and rounds):")
    print(f"{'N':>3} {'TTFT s':>7} {'tok/s/user':>10} {'agg tok/s':>9} {'failed':>6} "
          f"{'exact tokens':>12}")
    for level, rounds in summary.items():
        all_ok = [r for results, _ in rounds for r in results if r.ok]
        failed = sum(1 for results, _ in rounds for r in results if not r.ok)
        aggregates = [sum(r.completion_tokens or 0 for r in results if r.ok) / wall
                      for results, wall in rounds if wall]
        exact = "-" if not all_ok else (
            "yes" if all(r.tokens_exact for r in all_ok) else "approx"
        )
        print(f"{level:>3} {fmt(median([r.ttft_s for r in all_ok if r.ttft_s is not None])):>7} "
              f"{fmt(median([r.decode_tok_s for r in all_ok if r.decode_tok_s])):>10} "
              f"{fmt(median(aggregates)):>9} {failed:>6} {exact:>12}")

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["level", "repeat", "index", "ok", "ttft_s", "total_s", "completion_tokens",
                         "prompt_tokens", "decode_tok_s", "tokens_exact", "error"])
        for r in rows:
            writer.writerow([r.level, r.repeat, r.index, r.ok, r.ttft_s, r.total_s,
                             r.completion_tokens, r.prompt_tokens, r.decode_tok_s,
                             r.tokens_exact, r.error])
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
