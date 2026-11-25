"""Build docs/TFG/preregistre-cpu.md from the predictions that predict.py already wrote.

The Catalan prose lives in preregistre.template.md; this script only fills in the numbers.

    uv run python scripts/cpu/make_preregistre.py
"""

from __future__ import annotations

import csv
from pathlib import Path

PRED_DIR = Path("validation/cpu-20261125/predicciones")
CASES = Path("scripts/cpu/cases.csv")
TEMPLATE = Path("scripts/cpu/preregistre.template.md")
OUT = Path("docs/TFG/preregistre-cpu.md")

HEADER = ["Cas", "Model", "Quant.", "Context", "Pesos", "KV", "Overhead", "Físic", "Total", "Modes"]
SEPARATOR = "|---|---|---|---:|---:|---:|---:|---:|---:|---|"


def fmt(value: str) -> str:
    """Thousands with a dot, as in the rest of the thesis."""
    return f"{float(value):,.0f}".replace(",", ".")


commit = (PRED_DIR / "commit.txt").read_text().splitlines()[0].split("=")[1]
cases = {c["id"]: c for c in csv.DictReader(CASES.open(encoding="utf-8"))}
rows = list(csv.DictReader((PRED_DIR / "predicciones.csv").open(encoding="utf-8")))
by_id = {r["id"]: r for r in rows}

table = ["| " + " | ".join(HEADER) + " |", SEPARATOR]
for r in rows:
    c = cases[r["id"]]
    model = c["repo"].split("/")[1].replace("-GGUF", "")
    quant = "MXFP4_MOE" if c["quant"] == "unknown" else c["quant"]
    cells = [
        r["id"], model, quant, fmt(r["ctx"]), fmt(r["weights_mib"]), fmt(r["kv_mib"]),
        fmt(r["overhead_mib"]), fmt(r["physical_mib"]), fmt(r["total_mib"]),
        c["modes"].replace(";", " i "),
    ]
    table.append("| " + " | ".join(cells) + " |")

mmap_ids = [c["id"] for c in cases.values() if "mmap" in c["modes"].split(";")]
values = {
    "@@COMMIT@@": commit[:7],
    "@@TABLE@@": "\n".join(table),
    "@@MMAP_IDS@@": f"{mmap_ids[0]} a {mmap_ids[-1]}",
    "@@ORNITH_KV4@@": fmt(by_id["C23"]["kv_mib"]),
    "@@ORNITH_KV32@@": fmt(by_id["C24"]["kv_mib"]),
}
text = TEMPLATE.read_text(encoding="utf-8")
for token, value in values.items():
    text = text.replace(token, value)
OUT.write_text(text, encoding="utf-8", newline="\n")
print(f"{OUT} written")
