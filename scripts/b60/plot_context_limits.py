"""Plot the B60 context-limit results: free VRAM after loading, versus context.

Reads the runs in validation/b60-contextos-20261124/ and draws, for each model, the free VRAM that
llama.cpp reported next to Jaull's estimate, with the 1 GiB rule llama.cpp applies. The gap between
the two limits is the context Jaull rejects although llama.cpp accepts it.

It does not need to touch the project's dependencies:

    uv run --no-project --with matplotlib python scripts/b60/plot_context_limits.py
"""

from __future__ import annotations

import argparse
import re
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FREE_MIB = 23_613  # free VRAM of the B60 before every load
RULE_MIB = 1_024  # llama.cpp wants at least this much left free
OVERHEAD_MIB = 256  # Jaull's flat compute allowance

# label, weights (file size, MiB), KV MiB per token, fixed recurrent state (MiB).
# The same constants as docs/TFG/preregistre-b60.md and preregistre-b60-2.md.
MODELS = {
    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf": ("Qwen3-Coder 30B-A3B Q4_K_M", 17_695, 0.09375, 0),
    "Qwen2.5-32B-Instruct-Q4_K_M.gguf": ("Qwen2.5 32B Q4_K_M", 18_934, 0.25, 0),
    "Qwen3.8-27B-UD-Q4_K_M.gguf": ("Qwen3.8 27B UD-Q4_K_M", 15_702, 0.0625, 153),
}
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

LANG = "en"  # "ca" writes the Catalan versions the thesis uses, with a -ca suffix
TEXT = {
    "en": {
        "short_title": "{label}\nJaull stops {gap}k tokens ({pct} %) short",
        "ctx_k": "Context (thousand tokens)",
        "free_vram": "Free VRAM after loading (GiB)",
        "loaded": "llama.cpp, loaded",
        "not_loaded": "llama.cpp projection, not loaded",
        "estimate": "Jaull estimate",
        "rule": "1 GiB rule (llama.cpp)",
        "fits_rejected": "Fits per llama.cpp, rejected by Jaull",
        "suptitle": "Intel Arc Pro B60: context that fits, llama.cpp versus Jaull",
        "wants": "llama.cpp wants at least 1 GiB left free",
        "enough": "Enough memory left",
        "not_enough": "Not enough memory left",
        "projection_only": "projection only,\nnot loaded",
        "measured": "Measured by llama.cpp",
        "jaull_est": "Jaull's estimate",
        "jaull_stops": "Jaull stops at {x}k",
        "llama_stops": "llama.cpp stops at {x}k",
        "note": "In this strip it fits\n(the blue line is above 1 GiB),\n"
        "but Jaull's line is below it",
        "ctx_ask": "Context you ask for (thousand tokens)",
        "left_after": "Memory left after loading the model (GiB)",
        "on_b60": "{label} on an Intel Arc Pro B60",
        "of": "{a}k of {b}k",
        "pct_short": "{n} % short",
        "approves": "Context Jaull approves, as % of what llama.cpp accepts",
        "limits_title": "How far Jaull's estimate stops before the real limit",
    },
    "ca": {
        "short_title": "{label}\nJaull s'atura {gap}k tokens ({pct} %) abans",
        "ctx_k": "Context (milers de tokens)",
        "free_vram": "VRAM lliure després de carregar (GiB)",
        "loaded": "llama.cpp, carregat",
        "not_loaded": "projecció de llama.cpp, sense carregar",
        "estimate": "Estimació de Jaull",
        "rule": "Regla d'1 GiB (llama.cpp)",
        "fits_rejected": "Hi cap segons llama.cpp, Jaull ho rebutja",
        "suptitle": "Intel Arc Pro B60: context que hi cap, llama.cpp contra Jaull",
        "wants": "llama.cpp vol almenys 1 GiB lliure",
        "enough": "Queda prou memòria",
        "not_enough": "No en queda prou",
        "projection_only": "només projecció,\nsense carregar",
        "measured": "Mesurat per llama.cpp",
        "jaull_est": "Estimació de Jaull",
        "jaull_stops": "Jaull s'atura a {x}k",
        "llama_stops": "llama.cpp s'atura a {x}k",
        "note": "En aquesta franja hi cap\n(la línia blava és per sobre d'1 GiB),\n"
        "però la de Jaull és per sota",
        "ctx_ask": "Context demanat (milers de tokens)",
        "left_after": "Memòria que queda després de carregar el model (GiB)",
        "on_b60": "{label} a una Intel Arc Pro B60",
        "of": "{a}k de {b}k",
        "pct_short": "{n} % per sota",
        "approves": "Context que aprova Jaull, en % del que accepta llama.cpp",
        "limits_title": "Quant abans s'atura l'estimació de Jaull respecte del límit real",
    },
}


def tx(key: str, **kw: object) -> str:
    """Text for the current language; Catalan writes the decimal comma."""
    if LANG == "ca":
        kw = {k: _decimal_comma(v) for k, v in kw.items()}
    return TEXT[LANG][key].format(**kw)


def _decimal_comma(value: object) -> object:
    """Turn a numeric string such as "49.5" into "49,5"; leave any other value alone."""
    if isinstance(value, str) and value.replace(".", "", 1).isdigit():
        return value.replace(".", ",")
    return value


def suffix() -> str:
    return "-ca" if LANG == "ca" else ""

FIT_RE = re.compile(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB")
HEAD_RE = re.compile(r"modelo=(\S+) ctx=(\d+)")
LOADED_RE = re.compile(r"cargo=(\d)")


def read_runs(root: Path) -> dict[str, dict[int, tuple[int, bool]]]:
    """model file -> {context: (free MiB after loading, did it load)}."""
    runs: dict[str, dict[int, tuple[int, bool]]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        summary, log = folder / "summary.txt", folder / "server.log"
        if not (summary.is_file() and log.is_file()):
            continue
        text = summary.read_text(errors="replace")
        head, fit = HEAD_RE.search(text), FIT_RE.search(log.read_text(errors="replace"))
        if not (head and fit):
            continue  # run 0 of P1 has no memory breakdown
        model, ctx = head.group(1), int(head.group(2))
        loaded_match = LOADED_RE.search(text)
        loaded = bool(loaded_match and loaded_match.group(1) == "1")
        projected, free = int(fit.group(1)), int(fit.group(2))
        previous = runs.setdefault(model, {}).get(ctx)
        runs[model][ctx] = (free - projected, loaded or bool(previous and previous[1]))
    return runs


def jaull_free(ctx: float, weights: int, kv: float, fixed: int) -> float:
    return FREE_MIB - (weights + fixed + kv * ctx + OVERHEAD_MIB)


def jaull_limit(weights: int, kv: float, fixed: int) -> float:
    return (FREE_MIB - RULE_MIB - weights - fixed - OVERHEAD_MIB) / kv


def llama_limit(points: list[tuple[int, int]]) -> float:
    """Context where llama.cpp's free VRAM crosses the 1 GiB rule, by linear interpolation."""
    for (x0, y0), (x1, y1) in pairwise(points):
        if y0 >= RULE_MIB > y1:
            return x0 + (y0 - RULE_MIB) * (x1 - x0) / (y0 - y1)
    raise ValueError("no pair of runs brackets the 1 GiB rule")


def draw_free_vram(runs, out: Path) -> dict[str, tuple[float, float]]:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    limits: dict[str, tuple[float, float]] = {}
    for ax, (key, (label, weights, kv, fixed)) in zip(axes, MODELS.items(), strict=True):
        points = sorted((ctx, v[0]) for ctx, v in runs[key].items())
        loaded = {ctx: v[1] for ctx, v in runs[key].items()}
        lim_llama, lim_jaull = llama_limit(points), jaull_limit(weights, kv, fixed)
        limits[label] = (lim_jaull, lim_llama)
        xs = np.array([c for c, _ in points]) / 1000
        ys = np.array([v for _, v in points]) / 1024
        grid = np.linspace(0, max(xs) * 1.04, 200)
        estimate = [jaull_free(g * 1000, weights, kv, fixed) / 1024 for g in grid]
        ax.plot(grid, estimate, "--", color=ORANGE, lw=2)
        ax.plot(xs, ys, "-", color=BLUE, lw=1.5, zorder=2)
        for (ctx, _), x, y in zip(points, xs, ys, strict=True):
            face = BLUE if loaded[ctx] else "white"
            ax.plot(x, y, "o", ms=8, zorder=3, color=face, markeredgecolor=BLUE, markeredgewidth=2)
        ax.axhline(RULE_MIB / 1024, color=GREY, ls=":", lw=1.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(lim_jaull / 1000, lim_llama / 1000, color=GREEN, alpha=0.28, lw=0)
        ax.axvline(lim_llama / 1000, color=BLUE, lw=1, alpha=0.6)
        ax.axvline(lim_jaull / 1000, color=ORANGE, lw=1, alpha=0.7)
        gap = lim_llama - lim_jaull
        pct = 100 * gap / lim_llama
        title = tx("short_title", label=label, gap=f"{gap / 1000:.1f}", pct=f"{pct:.0f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(tx("ctx_k"))
        top = max(jaull_free(0, weights, kv, fixed) / 1024, max(ys))
        ax.set_ylim(min(ys) - 0.8, top + 0.8)
        ax.set_xlim(0, max(xs) * 1.04)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(tx("free_vram"))
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", ms=8, lw=1.5, label=tx("loaded")),
        plt.Line2D(
            [], [], color=BLUE, marker="o", ms=8, lw=0, markerfacecolor="white",
            markeredgewidth=2, label=tx("not_loaded"),
        ),
        plt.Line2D([], [], color=ORANGE, ls="--", lw=2, label=tx("estimate")),
        plt.Line2D([], [], color=GREY, ls=":", lw=1.5, label=tx("rule")),
        Patch(color=GREEN, alpha=0.28, label=tx("fits_rejected")),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(tx("suptitle"), fontsize=13)
    for ext in ("svg",):
        fig.savefig(out / f"b60-free-vram-vs-context{suffix()}.{ext}", dpi=200)
    plt.close(fig)
    return limits


# Per model: file slug, axis limits (thousand tokens, GiB) and where the labels go. Label
# positions are fractions of the axes: (x, y) of the text, and where the arrow lands.
SIMPLE = {
    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf": {
        "slug": "coder", "xlim": (0, 68), "ylim": (-1, 7),
        "blue": (0.147, 0.744, None), "orange": (0.132, 0.425, 14), "note": (0.853, 0.55),
    },
    "Qwen2.5-32B-Instruct-Q4_K_M.gguf": {
        "slug": "32b", "xlim": (0, 28), "ylim": (-1, 6),
        "blue": (0.44, 0.60, 12.3), "orange": (0.02, 0.38, 5), "note": (0.82, 0.50),
    },
    "Qwen3.8-27B-UD-Q4_K_M.gguf": {
        "slug": "27b", "xlim": (60, 150), "ylim": (-1.5, 5),
        "blue": (0.50, 0.70, 98.3), "orange": (0.08, 0.22, 75), "note": (0.80, 0.56),
    },
}


def draw_simple(runs, out: Path, key: str) -> None:
    """One model, no legend, labels written on the plot. The easiest figure to read."""
    label, weights, kv, fixed = MODELS[key]
    cfg = SIMPLE[key]
    (xmin, xmax), (ymin, ymax) = cfg["xlim"], cfg["ylim"]
    points = sorted((ctx, v[0]) for ctx, v in runs[key].items())
    loaded = {ctx: v[1] for ctx, v in runs[key].items()}
    lim_llama, lim_jaull = llama_limit(points), jaull_limit(weights, kv, fixed)
    shown = [(c, c / 1000, v / 1024) for c, v in points if xmin <= c / 1000 <= xmax]
    rule = RULE_MIB / 1024
    span = xmax - xmin

    def jaull_at(x_k: float) -> float:
        return jaull_free(x_k * 1000, weights, kv, fixed) / 1024

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axhspan(rule, ymax, color=GREEN, alpha=0.10, lw=0)
    ax.axhspan(ymin, rule, color=ORANGE, alpha=0.10, lw=0)
    ax.axhline(rule, color=GREY, ls=":", lw=1.8)
    left = xmin + 0.015 * span
    ax.text(left, rule + 0.12, tx("wants"), color=GREY, fontsize=10)
    ax.text(left, ymax - 0.55, tx("enough"), color=GREEN, fontsize=11, fontweight="bold")
    ax.text(left, ymin + 0.25, tx("not_enough"), color=ORANGE, fontsize=11,
            fontweight="bold")

    grid = np.linspace(xmin, xmax, 100)
    ax.plot(grid, [jaull_at(g) for g in grid], "--", color=ORANGE, lw=2.5)
    ax.plot([x for _, x, _ in shown], [y for _, _, y in shown], "-", color=BLUE, lw=2)
    for ctx, x, y in shown:
        face = BLUE if loaded[ctx] else "white"
        ax.plot(x, y, "o", ms=8, color=face, markeredgecolor=BLUE, markeredgewidth=2, zorder=3)
        if not loaded[ctx]:
            ax.annotate(tx("projection_only"), xy=(x, y), xytext=(14, 10),
                        textcoords="offset points", ha="left", va="bottom", fontsize=9, color=GREY)

    fx, fy, arrow_x = cfg["blue"]
    blue_arrow = None if arrow_x is None else {"arrowstyle": "->", "color": BLUE, "lw": 1.5}
    target = next((y for _, x, y in shown if arrow_x is not None and abs(x - arrow_x) < 0.5), 0)
    ax.annotate(tx("measured"), xy=(arrow_x or 0, target), xytext=(fx, fy),
                xycoords="data", textcoords="axes fraction", color=BLUE, fontsize=11,
                fontweight="bold", arrowprops=blue_arrow)
    fx, fy, arrow_x = cfg["orange"]
    ax.annotate(tx("jaull_est"), xy=(arrow_x, jaull_at(arrow_x)), xytext=(fx, fy),
                textcoords="axes fraction", color=ORANGE, fontsize=11, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5})

    ax.axvspan(lim_jaull / 1000, lim_llama / 1000, color=GREEN, alpha=0.35, lw=0)
    top = ymax - 1.4
    ax.annotate(tx("jaull_stops", x=f"{lim_jaull / 1000:.1f}"), xy=(lim_jaull / 1000, top),
                xytext=(lim_jaull / 1000 - 0.01 * span, top), ha="right", va="center",
                color=ORANGE, fontsize=10.5, fontweight="bold")
    ax.annotate(tx("llama_stops", x=f"{lim_llama / 1000:.1f}"), xy=(lim_llama / 1000, top),
                xytext=(lim_llama / 1000 + 0.01 * span, top), ha="left", va="center",
                color=BLUE, fontsize=10.5, fontweight="bold")
    inside = [(x, y) for _, x, y in shown if lim_jaull / 1000 <= x <= lim_llama / 1000]
    note = tx("note")
    nx, ny = cfg["note"]
    ax.annotate(note, xy=(inside[0][0], inside[0][1] + 0.1), xytext=(nx, ny), ha="center",
                va="center", fontsize=10, xycoords="data", textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "lw": 1.2})

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(tx("ctx_ask"))
    ax.set_ylabel(tx("left_after"))
    ax.set_title(tx("on_b60", label=label), fontsize=13)
    ax.grid(alpha=0.2)
    for ext in ("svg",):
        fig.savefig(out / f"b60-simple-{cfg['slug']}{suffix()}.{ext}", dpi=200)
    plt.close(fig)


def draw_limits(limits: dict[str, tuple[float, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    labels = list(limits)
    shares = [100 * jaull / llama for jaull, llama in limits.values()]
    rows = range(len(labels))
    ax.barh(rows, shares, color=ORANGE, height=0.55)
    ax.barh(rows, [100 - s for s in shares], left=shares, color=GREEN, alpha=0.3, height=0.55)
    for i, (jaull, llama) in enumerate(limits.values()):
        ax.text(
            shares[i] - 1.5, i, tx("of", a=f"{jaull / 1000:.1f}", b=f"{llama / 1000:.1f}"),
            va="center", ha="right", color="white", fontsize=10, fontweight="bold",
        )
        ax.text(101, i, tx("pct_short", n=f"{100 - shares[i]:.0f}"), va="center", fontsize=10)
    ax.set_yticks(rows, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(tx("approves"))
    ax.set_title(tx("limits_title"), fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    for ext in ("svg",):
        fig.savefig(out / f"b60-context-limits{suffix()}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("validation/b60-contextos-20261124"))
    parser.add_argument("--out", type=Path, default=Path("docs/TFG/figures"))
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    args = parser.parse_args()
    global LANG
    LANG = args.lang
    args.out.mkdir(parents=True, exist_ok=True)
    runs = read_runs(args.runs)
    missing = [key for key in MODELS if key not in runs]
    if missing:
        raise SystemExit(f"no runs found for: {', '.join(missing)}")
    limits = draw_free_vram(runs, args.out)
    draw_limits(limits, args.out)
    for key in SIMPLE:
        draw_simple(runs, args.out, key)
    for label, (jaull, llama) in limits.items():
        short = 100 * (llama - jaull) / llama
        print(f"{label}: Jaull {jaull:,.0f} tokens, llama.cpp {llama:,.0f} tokens,"
              f" {short:.1f} % short")
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
