from __future__ import annotations

import logging
import sys

import typer

from jaull.cli.doctor import run_doctor
from jaull.cli.estimate import EstimateOptions, run_estimate
from jaull.cli.inspect import run_inspect
from jaull.cli.scan import run_scan
from jaull.domain.inference import TargetDevice, WeightPrecision
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)

_DEFAULT_RESERVE_GIB = DEVICE_RESERVE_DEFAULT_BYTES / (1024**3)

app = typer.Typer(
    add_completion=False,
    # No `no_args_is_help`: the callback decides between launching the UI and
    # printing help, depending on whether there is a terminal to draw on.
    invoke_without_command=True,
    help="Analyze local hardware and inspect Hugging Face model repositories.",
)


def _is_interactive_terminal() -> bool:
    """True only when a full-screen UI can actually be drawn.

    Bare `jaull` opens the guided UI for interactive users, but a
    piped or redirected invocation (`jaull > out.txt`, a CI step, a
    script) must keep printing help instead of trying to render a full-screen
    app into something that is not a terminal.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        # Closed or replaced streams: treat as non-interactive.
        return False


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        # huggingface_hub emits noisy rate-limit / token hints on every unauthenticated
        # call; hide them unless the user explicitly asks for verbose output.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    if ctx.invoked_subcommand is not None:
        return

    if _is_interactive_terminal():
        _launch_ui()
        raise typer.Exit(code=0)

    typer.echo(ctx.get_help())
    raise typer.Exit(code=0)


def _launch_ui() -> None:
    # Local import: Textual is heavy and only needed when the UI is invoked.
    from jaull.tui.app import run as run_ui

    run_ui()


@app.command("scan", help="Detect and display local hardware resources.")
def scan_command() -> None:
    raise typer.Exit(code=run_scan())


@app.command("inspect", help="Inspect a Hugging Face model by repo_id or URL.")
def inspect_command(
    reference: str = typer.Argument(
        ..., metavar="MODEL", help="Hugging Face repo_id or URL."
    ),
) -> None:
    raise typer.Exit(code=run_inspect(reference))


@app.command("doctor", help="Verify the environment can run jaull.")
def doctor_command() -> None:
    raise typer.Exit(code=run_doctor())


@app.command("ui", help="Launch the interactive terminal UI (Textual).")
def ui_command() -> None:
    _launch_ui()


@app.command("estimate", help="Estimate the memory footprint of a model for inference.")
def estimate_command(
    reference: str = typer.Argument(
        ..., metavar="MODEL", help="Hugging Face repo_id or URL."
    ),
    quantization: str | None = typer.Option(
        None, "--quantization", "-q", help="GGUF quantization variant to use."
    ),
    dtype: WeightPrecision | None = typer.Option(
        None,
        "--dtype",
        "-d",
        case_sensitive=False,
        help="Weight precision (transformers repos only).",
    ),
    context: int | None = typer.Option(
        None,
        "--context",
        "-c",
        min=1,
        help="Context length in tokens (defaults to max_position_embeddings or 4096).",
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", "-b", min=1, help="Batch size in sequences."
    ),
    device: TargetDevice = typer.Option(
        TargetDevice.AUTO,
        "--device",
        case_sensitive=False,
        help="Target device: auto, gpu or cpu.",
    ),
    kv_dtype: WeightPrecision = typer.Option(
        WeightPrecision.FLOAT16,
        "--kv-dtype",
        case_sensitive=False,
        help="Precision assumed for the KV cache.",
    ),
    safety_margin_percent: float = typer.Option(
        SAFETY_MARGIN_DEFAULT_PERCENT,
        "--safety-margin-percent",
        min=0.0,
        max=100.0,
        help="Extra percentage added on top of the estimated subtotal.",
    ),
    device_reserve_gib: float = typer.Option(
        _DEFAULT_RESERVE_GIB,
        "--device-reserve-gib",
        min=0.0,
        help="Fixed GPU/CPU reserve to leave free (GiB).",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit stable JSON to stdout instead of a Rich report."
    ),
    no_resolve_base_model: bool = typer.Option(
        False,
        "--no-resolve-base-model",
        help=(
            "Skip base-model resolution and GGUF-header enrichment. "
            "Falls back to weights-only estimates for GGUF repositories."
        ),
    ),
    no_runtime_recommendation: bool = typer.Option(
        False,
        "--no-runtime-recommendation",
        help="Skip the runtime recommendation section (llama.cpp / Transformers / vLLM).",
    ),
) -> None:
    options = EstimateOptions(
        quantization=quantization,
        dtype=dtype,
        context=context,
        batch_size=batch_size,
        device=device,
        kv_dtype=kv_dtype,
        safety_margin_percent=safety_margin_percent,
        device_reserve_gib=device_reserve_gib,
        as_json=as_json,
        resolve_base_model=not no_resolve_base_model,
        recommend_runtime=not no_runtime_recommendation,
    )
    raise typer.Exit(code=run_estimate(reference, options))


if __name__ == "__main__":
    app()
