"""Plot two B60 speed results: speed against filled context, and speed against concurrent users.

The numbers are copied from the measured tables, they are not recomputed:

* full context: validation/b60-contextos-20261124/fill-P1-20261124-141935/summary.txt
  (Qwen3-Coder 30B-A3B, Q4_K_M, `-c 49152`, 128 generated tokens per request, one run).
* concurrency: docs/TFG/concurrencia-b60.md (Qwen3.6-35B-A3B UD-IQ4_XS, eight slots with a
  shared KV cache, median of three repetitions).

    uv run --no-project --with matplotlib python scripts/b60/plot_speed.py --lang ca
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#666666"

# prompt tokens in the context, prompt processing (tok/s), generation (tok/s); the first point
# is the almost empty context of case P1 (generation only).
FILL = [
    (0, None, 96.0),
    (3883, 890.5, 81.6),
    (15493, 331.1, 58.4),
    (30973, 179.7, 41.4),
    (43543, 130.8, 33.4),
]
# users, tok/s per user, aggregate tok/s
USERS = [(1, 62.5, 53.5), (4, 33.2, 104.1), (8, 19.4, 116.9), (16, 19.4, 118.5)]

TEXT = {
    "en": {
        "fill_title": "Qwen3-Coder 30B-A3B on an Intel Arc Pro B60: speed as the context fills",
        "gen": "Generation",
        "prompt": "Prompt processing",
        "x_fill": "Tokens already in the context (thousands)",
        "y_speed": "Speed (tokens per second)",
        "of_empty": "{n} % of the empty-context speed",
        "users_title": "Qwen3.6-35B-A3B on an Intel Arc Pro B60: several users at once",
        "per_user": "Per user",
        "aggregate": "All users together",
        "x_users": "Requests at the same time",
        "queued": "8 waiting\nin the queue",
    },
    "ca": {
        "fill_title": "Qwen3-Coder 30B-A3B a una Intel Arc Pro B60: "
        "velocitat a mesura que s'omple el context",
        "gen": "Generació",
        "prompt": "Processament del prompt",
        "x_fill": "Tokens ja al context (milers)",
        "y_speed": "Velocitat (tokens per segon)",
        "of_empty": "{n} % de la velocitat amb el context buit",
        "users_title": "Qwen3.6-35B-A3B a una Intel Arc Pro B60: diversos usuaris alhora",
        "per_user": "Per usuari",
        "aggregate": "Tots els usuaris junts",
        "x_users": "Peticions alhora",
        "queued": "8 esperen\na la cua",
    },
}
LANG = "en"
ARROW = {"arrowstyle": "->", "color": GREY, "lw": 1}


def tx(key: str, **kw: object) -> str:
    return TEXT[LANG][key].format(**kw)


def num(x: float) -> str:
    text = f"{x:.1f}"
    return text.replace(".", ",") if LANG == "ca" else text


def suffix() -> str:
    return "-ca" if LANG == "ca" else ""


def point_label(ax, text: str, xy: tuple[float, float], offset: tuple[int, int], color: str,
                ha: str = "center") -> None:
    """A value written next to a data point."""
    ax.annotate(text, xy, textcoords="offset points", xytext=offset, ha=ha, color=color,
                fontsize=10)


def draw_fill(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    xs = [t / 1000 for t, _, _ in FILL]
    gen = [g for _, _, g in FILL]
    ax.plot(xs, gen, "-o", color=BLUE, lw=2.2, ms=8, label=tx("gen"))
    for x, g in zip(xs, gen, strict=True):
        point_label(ax, num(g), (x, g), (0, 9), BLUE)
    pp = [(t / 1000, p) for t, p, _ in FILL if p is not None]
    ax.plot([x for x, _ in pp], [p for _, p in pp], "-s", color=ORANGE, lw=2.2, ms=7,
            label=tx("prompt"))
    for x, p in pp:
        point_label(ax, num(p), (x, p), (12, 6), ORANGE, ha="left")
    share = 100 * gen[-1] / gen[0]
    ax.annotate(tx("of_empty", n=f"{share:.0f}"), (xs[-1], gen[-1] + 8), xytext=(33, 480),
                ha="center", color=GREY, fontsize=10, arrowprops=ARROW)
    ax.set_xlabel(tx("x_fill"))
    ax.set_ylabel(tx("y_speed"))
    ax.set_title(tx("fill_title"), fontsize=11.5)
    ax.set_ylim(0, 1000)
    ax.set_xlim(-2, 48)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for ext in ("svg",):
        fig.savefig(out / f"b60-fullctx-speed{suffix()}.{ext}", dpi=200)
    plt.close(fig)


def draw_users(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    us = [u for u, _, _ in USERS]
    per = [p for _, p, _ in USERS]
    agg = [a for _, _, a in USERS]
    ax.plot(us, per, "-o", color=BLUE, lw=2.2, ms=8, label=tx("per_user"))
    ax.plot(us, agg, "-s", color=ORANGE, lw=2.2, ms=7, label=tx("aggregate"))
    for u, p in zip(us, per, strict=True):
        dy = 9 if u == 1 else -17  # with one user the per-user value is above the aggregate one
        point_label(ax, num(p), (u, p), (0, dy), BLUE)
    for u, a in zip(us, agg, strict=True):
        dy = -17 if u == 1 else 9
        point_label(ax, num(a), (u, a), (0, dy), ORANGE)
    ax.annotate(tx("queued"), (16, 19.4), textcoords="offset points", xytext=(-8, 30),
                ha="right", color=GREY, fontsize=9.5, arrowprops=ARROW)
    ax.set_xticks(us)
    ax.set_xlabel(tx("x_users"))
    ax.set_ylabel(tx("y_speed"))
    ax.set_title(tx("users_title"), fontsize=11.5)
    ax.set_ylim(0, 140)
    ax.set_xlim(0, 17.5)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center right")
    for ext in ("svg",):
        fig.savefig(out / f"b60-concurrency{suffix()}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/TFG/figures"))
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    args = parser.parse_args()
    global LANG
    LANG = args.lang
    args.out.mkdir(parents=True, exist_ok=True)
    draw_fill(args.out)
    draw_users(args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
