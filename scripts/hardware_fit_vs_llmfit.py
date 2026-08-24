#!/usr/bin/env python
"""Put Jaull's hardware fit analyzer and llmfit on the same placement question.

    uv run python scripts/hardware_fit_vs_llmfit.py
    uv run python scripts/hardware_fit_vs_llmfit.py --format markdown
    uv run python scripts/hardware_fit_vs_llmfit.py --format json > comparison.json

llmfit is *not* treated as ground truth. A divergence is a result to explain,
not a Jaull defect, and nothing here is wired into production code.

## What is actually being compared

The two tools do not answer the same question by default, and pretending they
do would make every row meaningless:

* **llmfit** starts from a model and hardware and picks a quantization to fit
  ("which build of this model should you run, and where?"). Its memory number
  therefore moves when the hardware moves.
* **Jaull's analyzer** starts from an artifact whose memory is already
  estimated and decides where its bytes go ("this exact GGUF — does it fit,
  and how is it split?"). It never re-quantizes.

So the harness pins llmfit to one quantization via ``llmfit plan --quant``,
reads back llmfit's *own* weight and KV-cache numbers, and feeds those to
Jaull's ``analyze_components``. With identical inputs, a mode difference can
only come from placement policy — which is the thing worth comparing.

Each case is analyzed twice on the Jaull side:

* ``bare`` — zero overhead, reserve and safety margin, i.e. llmfit's own
  accounting. This isolates the placement rule.
* ``policy`` — Jaull's production overhead/reserve/margin (imported from
  ``jaull.estimator``, never re-typed here). This is what Jaull would really
  answer, and the gap between the two columns *is* Jaull's conservatism.

``llmfit info`` is also recorded, unpinned, to show which quantization llmfit
would have chosen on its own for that hardware.

Requires the ``llmfit`` binary on PATH (developed against 1.1.10). Without it
the Jaull side is still printed, along with the exact commands to reproduce
the llmfit side elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # The Machine fixture is test data, and tests are not an installed package.
    sys.path.insert(0, str(REPO_ROOT))

from tests._hardware_fit_scenarios import GIB, Machine  # noqa: E402

from jaull.domain.estimation import HardwareFitMode, HardwareFitResult  # noqa: E402
from jaull.estimator.hardware_fit import analyze_components  # noqa: E402
from jaull.estimator.overhead import estimate_overhead  # noqa: E402
from jaull.estimator.policies import (  # noqa: E402
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)

# llmfit reports memory as ``*_gb`` but the values are binary gigabytes: for
# Qwen2.5-7B at 4096 tokens it prints kv_cache_gb 0.21875, and
# 2 * 28 layers * 4 kv heads * 128 head dim * 4096 * 2 bytes is exactly
# 0.21875 * 2**30. Converting with 1000**3 would silently shift every row.
LLMFIT_UNIT = GIB


@dataclass(frozen=True)
class Case:
    """One placement question posed to both tools."""

    name: str
    model: str
    quantization: str
    context_length: int
    machine: Machine
    # llmfit's embedded database, field ``num_hidden_layers``. Jaull needs it
    # to express an offload split in layers rather than in estimated bytes.
    total_layers: int
    asks: str

    def llmfit_flags(self) -> list[str]:
        vram = self.machine.vram_available_bytes or 0
        return [
            "--memory",
            f"{vram // (1024 ** 2)}M",
            "--ram",
            f"{self.machine.ram_total_bytes // (1024 ** 2)}M",
            "--max-context",
            str(self.context_length),
        ]


RTX_4060 = Machine(
    name="8 GiB VRAM / 32 GiB RAM",
    ram_available_bytes=28 * GIB,
    ram_total_bytes=32 * GIB,
    vram_available_bytes=8 * GIB,
)
RTX_2060_LAPTOP = Machine(
    name="6 GiB VRAM / 16 GiB RAM",
    ram_available_bytes=14 * GIB,
    ram_total_bytes=16 * GIB,
    vram_available_bytes=6 * GIB,
)
ENTRY_GPU = Machine(
    name="4 GiB VRAM / 32 GiB RAM",
    ram_available_bytes=28 * GIB,
    ram_total_bytes=32 * GIB,
    vram_available_bytes=4 * GIB,
)
WORKSTATION = Machine(
    name="24 GiB VRAM / 64 GiB RAM",
    ram_available_bytes=56 * GIB,
    ram_total_bytes=64 * GIB,
    vram_available_bytes=24 * GIB,
)
NO_GPU = Machine(
    name="no GPU / 32 GiB RAM",
    ram_available_bytes=28 * GIB,
    ram_total_bytes=32 * GIB,
)

# Deliberately mirrors the shape of tests/_hardware_fit_scenarios.py: one case
# per mode, plus the context and hardware sweeps that make a mode move.
CASES: tuple[Case, ...] = (
    Case(
        name="tinyllama_1b_on_8gb",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization="Q4_K_M",
        context_length=2048,
        machine=RTX_4060,
        total_layers=22,
        asks="A small model with room to spare — both tools should say GPU.",
    ),
    Case(
        name="qwen3b_on_8gb",
        model="Qwen/Qwen2.5-3B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=RTX_4060,
        total_layers=36,
        asks="The size Jaull's Top 5 currently favours on this machine.",
    ),
    Case(
        name="qwen7b_on_8gb",
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=RTX_4060,
        total_layers=28,
        asks="The reference case: does 7B Q4 stay resident on 8 GiB?",
    ),
    Case(
        name="qwen7b_on_8gb_long_context",
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="Q4_K_M",
        context_length=32768,
        machine=RTX_4060,
        total_layers=28,
        asks="Same artifact, 8x the context: does the KV cache move the fit?",
    ),
    Case(
        name="qwen7b_on_6gb",
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=RTX_2060_LAPTOP,
        total_layers=28,
        asks="Same artifact, smaller GPU: where does each tool draw the line?",
    ),
    Case(
        name="qwen7b_on_4gb",
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=ENTRY_GPU,
        total_layers=28,
        asks="Below the artifact's size: does llmfit answer about this build?",
    ),
    Case(
        name="qwen7b_without_gpu",
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=NO_GPU,
        total_layers=28,
        asks="No GPU at all — can llmfit even express that?",
    ),
    Case(
        name="qwen14b_on_8gb",
        model="Qwen/Qwen2.5-14B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=RTX_4060,
        total_layers=48,
        asks="Clearly beyond VRAM, comfortably inside RAM: split or CPU?",
    ),
    Case(
        name="qwen32b_on_6gb",
        model="Qwen/Qwen2.5-32B-Instruct",
        quantization="Q4_K_M",
        context_length=4096,
        machine=RTX_2060_LAPTOP,
        total_layers=64,
        asks="Beyond both pools — Jaull's TOO_LARGE. Has llmfit a mode for it?",
    ),
    Case(
        name="llama70b_on_24gb",
        model="meta-llama/Llama-3.1-70B",
        quantization="Q4_K_M",
        context_length=4096,
        machine=WORKSTATION,
        total_layers=80,
        asks="Large model, large machine: offload rather than refusal.",
    ),
)

# llmfit spells its run modes differently in each subcommand, so both spellings
# are mapped onto Jaull's vocabulary. Neither list has a mode for "does not fit
# anywhere", which is why TOO_LARGE has no counterpart on the llmfit side.
LLMFIT_INFO_MODE_EQUIVALENT: dict[str, HardwareFitMode] = {
    "GPU": HardwareFitMode.GPU_RESIDENT,
    "CPU+GPU": HardwareFitMode.GPU_OFFLOAD,
    "CPU": HardwareFitMode.CPU_RAM,
}
LLMFIT_PLAN_MODE_EQUIVALENT: dict[str, HardwareFitMode] = {
    "Gpu": HardwareFitMode.GPU_RESIDENT,
    "CpuOffload": HardwareFitMode.GPU_OFFLOAD,
    "CpuOnly": HardwareFitMode.CPU_RAM,
}


# ---------------------------------------------------------------------------
# llmfit
# ---------------------------------------------------------------------------


class LlmfitError(RuntimeError):
    pass


def _llmfit(binary: str, args: list[str], timeout: int) -> dict[str, object]:
    command = [binary, *args, "--json", "--no-dashboard"]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LlmfitError(f"{' '.join(command)}: {exc}") from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = (completed.stdout or completed.stderr).strip()[:200]
        raise LlmfitError(f"{' '.join(command)}: {detail or exc}") from exc
    if not isinstance(payload, dict):
        raise LlmfitError(f"{' '.join(command)}: unexpected JSON shape")
    return payload


def _llmfit_view(binary: str, case: Case, timeout: int) -> dict[str, object]:
    """llmfit's answer for one case: pinned plan plus its unpinned preference."""
    plan = _llmfit(
        binary,
        [
            *case.llmfit_flags(),
            "plan",
            case.model,
            "--context",
            str(case.context_length),
            "--quant",
            case.quantization,
        ],
        timeout,
    )
    info = _llmfit(binary, [*case.llmfit_flags(), "info", case.model], timeout)

    alternatives = plan.get("kv_alternatives")
    pinned = {}
    if isinstance(alternatives, list):
        pinned = next(
            (
                entry
                for entry in alternatives
                if isinstance(entry, dict) and entry.get("kv_quant") == "fp16"
            ),
            {},
        )
    models = info.get("models")
    model = models[0] if isinstance(models, list) and models else {}
    system = info.get("system") if isinstance(info.get("system"), dict) else {}
    current = plan.get("current") if isinstance(plan.get("current"), dict) else {}

    return {
        "run_mode": model.get("run_mode"),
        "fit_level": model.get("fit_level"),
        "best_quantization": model.get("best_quant"),
        "required_gib_for_chosen_quant": model.get("memory_required_gb"),
        "required_gib_for_pinned_quant": pinned.get("memory_required_gb"),
        "kv_cache_gib_for_pinned_quant": pinned.get("kv_cache_gb"),
        "plan_run_mode_for_pinned_quant": current.get("run_mode"),
        "plan_fit_level_for_pinned_quant": current.get("fit_level"),
        "effective_vram_gib": system.get("gpu_vram_gb"),
        "effective_ram_total_gib": system.get("total_ram_gb"),
        "effective_ram_available_gib": system.get("available_ram_gb"),
        "notes": model.get("notes"),
    }


# ---------------------------------------------------------------------------
# Jaull
# ---------------------------------------------------------------------------


def _jaull_view(case: Case, weights_bytes: int, kv_cache_bytes: int) -> dict[str, object]:
    """Jaull's placement for llmfit's own weight and KV numbers, twice."""
    bare = analyze_components(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=0,
        hardware=case.machine.profile(),
        total_layers=case.total_layers,
    )

    overhead_bytes = estimate_overhead(weights_bytes).component.bytes or 0
    reserve_bytes = DEVICE_RESERVE_DEFAULT_BYTES if case.machine.vram_available_bytes else 0
    subtotal = weights_bytes + kv_cache_bytes + overhead_bytes + reserve_bytes
    margin_bytes = math.ceil(subtotal * (SAFETY_MARGIN_DEFAULT_PERCENT / 100.0))
    policy = analyze_components(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        hardware=case.machine.profile(),
        device_reserve_bytes=reserve_bytes,
        safety_margin_bytes=margin_bytes,
        total_layers=case.total_layers,
    )

    return {
        "bare": _fit_summary(bare),
        "policy": _fit_summary(policy),
        "policy_overhead_gib": overhead_bytes / GIB,
        "policy_reserve_gib": reserve_bytes / GIB,
        "policy_margin_gib": margin_bytes / GIB,
    }


def _fit_summary(result: HardwareFitResult) -> dict[str, object]:
    return {
        "mode": result.mode.value,
        "placement_method": result.placement_method.value,
        "gpu_required_gib": _gib(result.gpu_required_bytes),
        "ram_required_gib": _gib(result.ram_required_bytes),
        "gpu_weight_gib": _gib(result.gpu_weight_bytes),
        "ram_weight_gib": _gib(result.ram_weight_bytes),
        "gpu_layers": result.gpu_layers,
        "total_layers": result.total_layers,
    }


def _gib(value: int | None) -> float | None:
    return None if value is None else round(value / GIB, 4)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _differences(case: Case, jaull: dict[str, object], llmfit: dict[str, object]) -> list[str]:
    """What to explain about this row. Neither column is the reference.

    The placement rule is compared against ``llmfit plan --quant``, because
    that is the only llmfit answer that is about the *same* artifact Jaull was
    asked about. ``llmfit info`` is reported alongside it, but it answers a
    different question — which build to run — so a mode difference there is
    expected rather than a divergence.
    """
    out: list[str] = []
    bare = jaull["bare"]
    policy = jaull["policy"]
    assert isinstance(bare, dict) and isinstance(policy, dict)

    pinned_mode = llmfit.get("plan_run_mode_for_pinned_quant")
    pinned_equivalent = LLMFIT_PLAN_MODE_EQUIVALENT.get(str(pinned_mode))
    if pinned_equivalent is None:
        out.append(f"llmfit plan run mode {pinned_mode!r} has no Jaull equivalent")
    elif pinned_equivalent.value != bare["mode"]:
        out.append(
            f"placement rule differs: on identical weights and KV cache, llmfit "
            f"places this {pinned_mode}, Jaull places it {bare['mode']}"
        )

    if bare["mode"] != policy["mode"]:
        out.append(
            f"Jaull's overhead, device reserve and safety margin alone move the "
            f"mode from {bare['mode']} to {policy['mode']} "
            f"(+{jaull['policy_overhead_gib']:.2f} overhead, "
            f"+{jaull['policy_reserve_gib']:.2f} reserve, "
            f"+{jaull['policy_margin_gib']:.2f} margin, in GiB)"
        )

    chosen = llmfit.get("best_quantization")
    if chosen and chosen != case.quantization:
        info_mode = llmfit.get("run_mode")
        out.append(
            f"different question: llmfit's own recommendation for this hardware "
            f"is {chosen} ({info_mode}), not the {case.quantization} artifact "
            "Jaull was asked to place — Jaull never substitutes a quantization"
        )
    elif chosen:
        info_equivalent = LLMFIT_INFO_MODE_EQUIVALENT.get(str(llmfit.get("run_mode")))
        if info_equivalent is not pinned_equivalent:
            out.append(
                f"llmfit disagrees with itself on the same {case.quantization} "
                f"artifact: `info` says {llmfit.get('run_mode')}, "
                f"`plan` says {pinned_mode}"
            )

    requested_vram = (case.machine.vram_available_bytes or 0) / GIB
    effective_vram = llmfit.get("effective_vram_gib")
    if isinstance(effective_vram, int | float) and abs(effective_vram - requested_vram) > 0.05:
        out.append(
            f"llmfit saw {effective_vram} GiB of VRAM, not the {requested_vram:g} "
            "GiB requested"
        )

    if case.machine.vram_available_bytes is None:
        out.append(
            "the machine has no GPU. llmfit's overrides cap VRAM but never remove "
            f"the device, so it still reports has_gpu and answers {pinned_mode}; "
            "Jaull models absence of a GPU as a distinct topology"
        )

    if policy["mode"] == HardwareFitMode.TOO_LARGE.value:
        out.append(
            "different encodings of infeasibility: Jaull makes it a placement "
            "state (TOO_LARGE), llmfit splits it across run_mode (which path it "
            f"describes) and fit_level (how well that path fits — here "
            f"{llmfit.get('plan_fit_level_for_pinned_quant')}), so llmfit's "
            "run_mode is not a standalone fit verdict"
        )

    return out or ["no divergence to explain"]


def _compare(binary: str | None, timeout: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASES:
        row: dict[str, object] = {
            "case": case.name,
            "model": case.model,
            "quantization": case.quantization,
            "context_length": case.context_length,
            "machine": case.machine.name,
            "asks": case.asks,
        }
        if binary is None:
            row["error"] = "llmfit not available"
            row["reproduce"] = _reproduce(case)
            rows.append(row)
            continue

        try:
            llmfit = _llmfit_view(binary, case, timeout)
        except LlmfitError as exc:
            row["error"] = str(exc)
            row["reproduce"] = _reproduce(case)
            rows.append(row)
            continue

        required = llmfit["required_gib_for_pinned_quant"]
        kv = llmfit["kv_cache_gib_for_pinned_quant"]
        if not isinstance(required, int | float) or not isinstance(kv, int | float):
            row["error"] = "llmfit did not report a pinned-quant memory breakdown"
            row["llmfit"] = llmfit
            rows.append(row)
            continue

        weights_bytes = round((required - kv) * LLMFIT_UNIT)
        kv_cache_bytes = round(kv * LLMFIT_UNIT)
        jaull = _jaull_view(case, weights_bytes, kv_cache_bytes)

        row["shared_inputs"] = {
            "weights_gib": round(weights_bytes / GIB, 4),
            "kv_cache_gib": round(kv_cache_bytes / GIB, 4),
            "source": "llmfit plan --quant (so both tools size the same artifact)",
        }
        row["jaull"] = jaull
        row["llmfit"] = llmfit
        row["differences"] = _differences(case, jaull, llmfit)
        rows.append(row)
    return rows


def _reproduce(case: Case) -> list[str]:
    flags = " ".join(case.llmfit_flags())
    return [
        f"llmfit {flags} plan '{case.model}' --context {case.context_length} "
        f"--quant {case.quantization} --json --no-dashboard",
        f"llmfit {flags} info '{case.model}' --json --no-dashboard",
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# "pinned" columns are the apples-to-apples comparison: both tools sizing the
# same artifact. "auto" columns are llmfit answering its own question — which
# build to run on this hardware — and are here to show that the questions differ.
_HEADER = (
    "case",
    "machine",
    "Jaull (policy)",
    "Jaull (bare)",
    "llmfit pinned",
    "pinned GiB",
    "llmfit auto",
    "auto quant",
    "auto GiB",
)


def _cells(row: dict[str, object]) -> list[str]:
    jaull = row.get("jaull")
    llmfit = row.get("llmfit")
    if not isinstance(jaull, dict) or not isinstance(llmfit, dict):
        return [
            str(row["case"]),
            str(row["machine"]),
            "unavailable",
            "unavailable",
            str(row.get("error", "unavailable")),
            "-",
            "-",
            "-",
            "-",
        ]
    bare = jaull["bare"]
    policy = jaull["policy"]
    assert isinstance(bare, dict) and isinstance(policy, dict)
    return [
        str(row["case"]),
        str(row["machine"]),
        str(policy["mode"]),
        str(bare["mode"]),
        str(llmfit.get("plan_run_mode_for_pinned_quant") or "-"),
        _number(llmfit.get("required_gib_for_pinned_quant")),
        str(llmfit.get("run_mode") or "-"),
        str(llmfit.get("best_quantization") or "-"),
        _number(llmfit.get("required_gib_for_chosen_quant")),
    ]


def _number(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, int | float) else "-"


def _table(rows: list[dict[str, object]], binary: str | None) -> str:
    cells = [list(_HEADER), *(_cells(row) for row in rows)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(_HEADER))]
    lines = [
        "Jaull hardware fit vs llmfit — same artifact, same machine",
        f"llmfit: {binary or 'NOT FOUND on PATH'}",
        "",
        "  ".join(cells[0][i].ljust(widths[i]) for i in range(len(_HEADER))).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(_HEADER))),
    ]
    lines.extend(
        "  ".join(row[i].ljust(widths[i]) for i in range(len(_HEADER))).rstrip()
        for row in cells[1:]
    )

    lines.extend(["", "Differences observed (llmfit is not the reference):", ""])
    for row in rows:
        lines.append(f"  {row['case']} — {row['asks']}")
        for note in _row_notes(row):
            lines.append(f"      - {note}")
    return "\n".join(lines)


def _row_notes(row: dict[str, object]) -> list[str]:
    differences = row.get("differences")
    if isinstance(differences, list):
        return [str(item) for item in differences]
    error = row.get("error")
    reproduce = row.get("reproduce")
    notes = [f"not compared: {error}"] if error else ["not compared"]
    if isinstance(reproduce, list):
        notes.extend(f"reproduce: {command}" for command in reproduce)
    return notes


def _markdown(rows: list[dict[str, object]], binary: str | None) -> str:
    lines = [
        "# Jaull hardware fit vs llmfit",
        "",
        f"Generated by `scripts/hardware_fit_vs_llmfit.py`. llmfit: "
        f"`{binary or 'NOT FOUND on PATH'}`.",
        "",
        "llmfit is not the reference implementation. A divergence is a result "
        "to explain.",
        "",
        "| " + " | ".join(_HEADER) + " |",
        "| " + " | ".join("---" for _ in _HEADER) + " |",
    ]
    lines.extend("| " + " | ".join(_cells(row)) + " |" for row in rows)
    lines.extend(["", "## Differences observed", ""])
    for row in rows:
        lines.append(f"**{row['case']}** — {row['asks']}")
        lines.append("")
        lines.extend(f"- {note}" for note in _row_notes(row))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    binary = args.llmfit or shutil.which("llmfit")
    rows = _compare(binary, args.timeout)

    if args.format == "json":
        print(json.dumps({"llmfit": binary, "cases": rows}, indent=2))
    elif args.format == "markdown":
        print(_markdown(rows, binary))
    else:
        print(_table(rows, binary))

    if binary is None:
        print(
            "\nllmfit was not found on PATH, so only the Jaull side is defined. "
            "Install it (https://github.com/alexsjones/llmfit) and re-run, or "
            "use the per-case `reproduce` commands above on a machine that has "
            "it.",
            file=sys.stderr,
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Jaull's placement with llmfit.")
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument("--llmfit", help="Path to the llmfit binary (default: PATH).")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-invocation llmfit timeout in seconds (default: 120).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
