"""One-machine pilot: one loaded server, 1/2/4 simultaneous native completions.

Run from the repository root with uv run python <this file>. Results are raw
pilot evidence, not Jaull ExperimentRecords or qualification verdicts.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import psutil
import pynvml

from jaull.hardware.detector import detect_hardware
from jaull.runtime.llama_cpp_memory_report import parse_llama_cpp_allocation

OUT = Path(__file__).resolve().parent
SERVER = Path.home() / "tools/llama.cpp/build-cuda/bin/llama-server"
MODEL = Path.home() / "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
EXPECTED_SHA = "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423"
BASE = "http://127.0.0.1:18086"
HTTP = build_opener(ProxyHandler({}))


def save(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def get(path: str, value: object | None = None) -> object:
    data = json.dumps(value).encode() if value is not None else None
    request = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(request, timeout=10) as response:
        return json.load(response)


def complete(label: str, slot: int, prompt: list[int], barrier: threading.Barrier) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": 128,
        "stream": True,
        "return_tokens": True,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "cache_prompt": False,
        "id_slot": slot,
    }
    barrier.wait(timeout=30)
    started = time.perf_counter()
    result = {"slot_requested": slot, "started": started, "success": False}
    first = last = None
    tokens_streamed = 0
    final = None
    events = []
    try:
        request = Request(
            BASE + "/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with HTTP.open(request, timeout=180) as response:
            result["http_status"] = response.status
            for line in response:
                received = time.perf_counter()
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                events.append({"received_after_ms": (received - started) * 1000, "data": event})
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("stop"):
                    final = event
                    break
                if event.get("tokens"):
                    first = received if first is None else first
                    last = received
                    tokens_streamed += len(event["tokens"])
        if final is None:
            raise RuntimeError("Stream ended without a terminal completion event")
        count = final["tokens_predicted"]
        result.update(
            success=count == 128
            and final.get("tokens_evaluated") == 512
            and final.get("id_slot") == slot
            and first is not None
            and not final.get("truncated", False),
            ttft_ms=(first - started) * 1000 if first is not None else None,
            tokens_predicted=count,
            tokens_evaluated=final.get("tokens_evaluated"),
            tokens_streamed=tokens_streamed,
            slot_observed=final.get("id_slot"),
            server_timings=final.get("timings"),
            truncated=final.get("truncated"),
            client_decode_tps=(count - 1) / (last - first)
            if tokens_streamed == count and first is not None and last > first
            else None,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["ended"] = time.perf_counter()
    result["latency_ms"] = (result["ended"] - started) * 1000
    save(f"{label}-slot{slot}.json", {"request": payload, "result": result, "events": events})
    return result


def run_group(label: str, users: int, prompt: list[int]) -> dict:
    barrier = threading.Barrier(users)
    with ThreadPoolExecutor(max_workers=users) as pool:
        futures = [pool.submit(complete, label, slot, prompt, barrier) for slot in range(users)]
        requests = [future.result() for future in futures]
    started = min(r["started"] for r in requests)
    ended = max(r["ended"] for r in requests)
    success = all(r["success"] for r in requests)
    group = {
        "label": label,
        "users": users,
        "requests": requests,
        "started": started,
        "ended": ended,
        "success": success,
        "launch_skew_ms": (max(r["started"] for r in requests) - started) * 1000,
        "wall_seconds": ended - started,
        "aggregate_output_tps": sum(r["tokens_predicted"] for r in requests) / (ended - started)
        if success
        else None,
    }
    save(f"{label}.json", group)
    print(
        label,
        "success=" + str(success),
        "aggregate_tps=" + str(group["aggregate_output_tps"]),
        flush=True,
    )
    return group


def main() -> None:
    if (OUT / "metadata.json").exists():
        raise RuntimeError("Output already contains a run; do not overwrite evidence")
    if not SERVER.is_file() or sha(MODEL) != EXPECTED_SHA:
        raise RuntimeError("Server missing or reference artifact digest differs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 18086))
    command = [
        str(SERVER),
        "--model",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        "18086",
        "--ctx-size",
        "16384",
        "--parallel",
        "4",
        "--no-kv-unified",
        "--n-gpu-layers",
        "24",
        "--threads",
        "4",
        "--threads-batch",
        "4",
        "--fit",
        "off",
        "--cache-ram",
        "0",
        "--no-context-shift",
        "--slots",
    ]
    metadata = {
        "methodology": "llama_server_concurrency_pilot_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "collector_sha256": sha(Path(__file__)),
        "runtime_commit": subprocess.check_output(
            ["git", "-C", str(SERVER.parents[2]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_source_status": subprocess.check_output(
            ["git", "-C", str(SERVER.parents[2]), "status", "--short"], text=True
        ),
        "runtime_sha256": {
            p.name: sha(p)
            for p in sorted(SERVER.parent.iterdir())
            if p.is_file() and (p.name == "llama-server" or ".so" in p.name)
        },
        "artifact": {
            "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": MODEL.name,
            "sha256": EXPECTED_SHA,
            "revision": None,
            "revision_note": "Local historical file; repository revision not inferred",
        },
        "hardware": detect_hardware().model_dump(mode="json"),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "server_command": command,
        "workload": {
            "context_per_slot": 4096,
            "input_tokens": 512,
            "output_tokens": 128,
            "concurrent_users": [1, 2, 4],
            "repetitions": 3,
            "mode": "service_pilot",
            "slo": None,
            "order": [[1, 2, 4], [4, 1, 2], [2, 4, 1]],
        },
    }
    save("metadata.json", metadata)
    stop = threading.Event()
    samples = []
    groups = []
    with (OUT / "server.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        watcher = None
        try:
            pynvml.nvmlInit()
            gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

            def monitor() -> None:
                while not stop.is_set():
                    sample = {"at": time.perf_counter()}
                    try:
                        sample["server_rss_bytes"] = psutil.Process(process.pid).memory_info().rss
                        sample["device_wide_used_bytes"] = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
                        attributed = [
                            p.usedGpuMemory
                            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
                            if p.pid == process.pid
                            and p.usedGpuMemory is not None
                            and p.usedGpuMemory != (1 << 64) - 1
                        ]
                        sample["nvml_process_bytes"] = max(attributed) if attributed else None
                    except (psutil.Error, pynvml.NVMLError) as exc:
                        sample["memory_error"] = str(exc)
                    try:
                        slots = get("/slots")
                        sample["slots"] = [
                            {
                                "id": s["id"],
                                "is_processing": s["is_processing"],
                                "n_ctx": s["n_ctx"],
                            }
                            for s in slots
                        ]
                    except (OSError, ValueError, KeyError) as exc:
                        sample["slots_error"] = str(exc)
                    samples.append(sample)
                    stop.wait(0.1)

            watcher = threading.Thread(target=monitor)
            watcher.start()
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {process.returncode}")
                try:
                    if get("/health").get("status") == "ok":
                        break
                except URLError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server startup exceeded 300 seconds")
                time.sleep(0.5)
            save("props.json", get("/props"))
            slots = get("/slots")
            save("slots-before.json", slots)
            assert len(slots) == 4 and all(s["n_ctx"] == 4096 for s in slots), slots
            text = "Explain how to review Python code for correctness and performance. " * 100
            tokens = get("/tokenize", {"content": text, "add_special": True})["tokens"][:512]
            assert len(tokens) == 512
            save("prompt.json", {"source_text": text, "tokens": tokens})
            warmup = run_group("warmup", 4, tokens)
            if not warmup["success"]:
                raise RuntimeError("Warm-up failed; measured groups were not started")
            for repetition, order in enumerate(metadata["workload"]["order"], 1):
                for users in order:
                    group = run_group(f"r{repetition}-u{users}", users, tokens)
                    groups.append(group)
                    if not group["success"]:
                        raise RuntimeError(
                            "A group failed; aborting instead of overlapping unfinished tasks"
                        )
                    time.sleep(0.25)
        except (OSError, ValueError, RuntimeError, AssertionError, pynvml.NVMLError) as exc:
            save("failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            stop.set()
            if watcher is not None:
                watcher.join(timeout=15)
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            save("memory-and-slots.json", samples)
            save("groups.json", groups)
            allocation = parse_llama_cpp_allocation(
                (OUT / "server.log").read_text(errors="replace"),
                runtime_build=metadata["runtime_commit"],
            )
            save(
                "runtime-allocation.json",
                allocation.model_dump(mode="json") if allocation else None,
            )
            with suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
    for group in groups:
        during = [s for s in samples if group["started"] <= s["at"] <= group["ended"]]
        group["mean_ttft_ms"] = mean(r["ttft_ms"] for r in group["requests"])
        group["mean_request_latency_ms"] = mean(r["latency_ms"] for r in group["requests"])
        group["mean_generation_tps"] = mean(
            r["server_timings"]["predicted_per_second"] for r in group["requests"]
        )
        group["peak_active_slots"] = max(
            (sum(s["is_processing"] for s in sample.get("slots", [])) for sample in during),
            default=0,
        )
        for key in ("server_rss_bytes", "device_wide_used_bytes", "nvml_process_bytes"):
            group["peak_" + key] = max(
                (s[key] for s in during if s.get(key) is not None), default=None
            )
    summary = {}
    for users in (1, 2, 4):
        selected = [g for g in groups if g["users"] == users]
        assert len(selected) == 3 and all(g["success"] for g in selected)
        summary[str(users)] = {
            metric: {
                "mean": mean(g[metric] for g in selected),
                "sample_stdev": stdev(g[metric] for g in selected),
            }
            for metric in (
                "mean_ttft_ms",
                "mean_request_latency_ms",
                "mean_generation_tps",
                "aggregate_output_tps",
            )
        }
        summary[str(users)]["groups"] = selected
    save("summary.json", summary)


if __name__ == "__main__":
    main()
