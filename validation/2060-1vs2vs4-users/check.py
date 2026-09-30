"""Offline collector guard: no GPU/server or evidence writes required."""

import hashlib
import io
import json
import runpy
import threading
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

complete = runpy.run_path(str(Path(__file__).with_name("run.py")))["complete"]

with TemporaryDirectory() as temporary:
    for input_tokens, terminal, expected in [
        (512, True, True),
        (511, True, False),
        (512, False, False),
    ]:
        events = [{"stop": False, "tokens": [n], "content": ""} for n in range(128)]
        if terminal:
            events.append(
                {
                    "stop": True,
                    "tokens_predicted": 128,
                    "tokens_evaluated": input_tokens,
                    "id_slot": 0,
                }
            )
        response = io.BytesIO(
            b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        )
        response.status = 200
        with patch.dict(
            complete.__globals__,
            OUT=Path(temporary),
            HTTP=SimpleNamespace(open=lambda *a, response=response, **k: response),
        ):
            result = complete("offline", 0, list(range(512)), threading.Barrier(1))
        assert result["success"] is expected, result
        if terminal:
            assert result["tokens_streamed"] == 128
            assert result["ttft_ms"] is not None
    print("Collector checks passed: token-based TTFT, exact workload, incomplete-stream rejection.")

root = Path(__file__).resolve().parent
if (root / "summary.json").exists():
    metadata = json.loads((root / "metadata.json").read_text())
    assert (
        hashlib.sha256((root / "run.py").read_bytes()).hexdigest() == metadata["collector_sha256"]
    )
    summary = json.loads((root / "summary.json").read_text())
    count = 0
    for users, level in summary.items():
        assert len(level["groups"]) == 3
        for group in level["groups"]:
            original = json.loads((root / (group["label"] + ".json")).read_text())
            assert group["requests"] == original["requests"]
            assert group["success"] and group["peak_active_slots"] == int(users)
            assert len(group["requests"]) == int(users)
            for result in group["requests"]:
                raw = json.loads(
                    (root / f"{group['label']}-slot{result['slot_requested']}.json").read_text()
                )
                assert raw["result"] == result
                assert result["tokens_evaluated"] == 512 and result["tokens_predicted"] == 128
                assert result["tokens_streamed"] == 128
                assert result["server_timings"]["prompt_n"] == 512
                assert result["server_timings"]["cache_n"] == 0
                count += 1
        for metric in (
            "mean_ttft_ms",
            "mean_request_latency_ms",
            "mean_generation_tps",
            "aggregate_output_tps",
        ):
            values = [g[metric] for g in level["groups"]]
            assert level[metric] == {"mean": mean(values), "sample_stdev": stdev(values)}
    assert count == 21
    print(
        "Evidence checks passed: 21 exact completions, overlap, summary and collector hash."
    )
