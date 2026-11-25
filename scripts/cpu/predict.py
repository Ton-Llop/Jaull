"""Write Jaull's CPU predictions for every case in cases.csv, before any measurement.

Run from the repository root, in WSL (it only reads Hub metadata, it does not download weights):

    uv run python scripts/cpu/predict.py

It stores each full JSON report in validation/cpu-20261125/predicciones/ and a summary CSV
next to it. For another campaign: --cases scripts/cpu/cases-v2.csv --out <folder>.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cases", default="scripts/cpu/cases.csv")
parser.add_argument("--out", default="validation/cpu-20261125/predicciones")
args = parser.parse_args()
CASES = Path(args.cases)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

GIT_HEAD = ["git", "rev-parse", "HEAD"]
GIT_SRC_STATUS = ["git", "status", "--porcelain", "--", "src"]
commit = subprocess.run(GIT_HEAD, capture_output=True, text=True, check=True).stdout.strip()
status = subprocess.run(GIT_SRC_STATUS, capture_output=True, text=True, check=True).stdout
dirty = bool(status.strip())
(OUT / "commit.txt").write_text(f"jaull_commit={commit}\nsrc_dirty={dirty}\n")

rows = []
for case in csv.DictReader(CASES.open(encoding="utf-8")):
    cmd = [
        "uv", "run", "jaull", "estimate", case["repo"], "-q", case["quant"], "-c", case["ctx"],
        "--device", "cpu", "--json", "--no-runtime-recommendation",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{case['id']} failed:\n{proc.stderr[-600:]}")
    report = json.loads(proc.stdout)
    (OUT / f"{case['id']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    mem = report["memory"]
    rows.append({
        "id": case["id"], "repo": case["repo"], "quant": case["quant"], "ctx": case["ctx"],
        "weights_mib": round(mem["weights_bytes"] / 2**20, 1),
        "kv_mib": round(mem["kv_cache_bytes"] / 2**20, 1),
        "overhead_mib": round(mem["runtime_overhead_bytes"] / 2**20, 1),
        "physical_mib": round(mem["physical_bytes"] / 2**20, 1),
        "reserve_mib": round(mem["device_reserve_bytes"] / 2**20, 1),
        "total_mib": round(mem["total_bytes"] / 2**20, 1),
        "confidence": report["assessment"]["confidence"],
    })
    last = rows[-1]
    print(f"{case['id']}  physical {last['physical_mib']:>8} MiB  total {last['total_mib']:>8} MiB")

with (OUT / "predicciones.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"commit {commit} (src dirty: {dirty}); {len(rows)} cases written to {OUT}")
