from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from jaull.application.serving.policies import (
    SERVE_DEFAULT_CONTEXT_PER_USER,
    SERVE_DEFAULT_USERS,
)
from jaull.cli.cases import (
    CaseOptions,
    CreateCaseOptions,
    ExportCaseOptions,
    run_create_case,
    run_export_case,
    run_list_cases,
    run_show_case,
    run_validate_case,
    run_validate_case_bundle,
)
from jaull.cli.doctor import run_doctor
from jaull.cli.estimate import EstimateOptions, run_estimate
from jaull.cli.experiments import ReevaluateOptions, run_reevaluate
from jaull.cli.inspect import run_inspect
from jaull.cli.quality import run_quality, run_quality_compare, run_quality_setup
from jaull.cli.run import RunOptions, run_model
from jaull.cli.runtime import run_runtime_install
from jaull.cli.scan import run_scan
from jaull.cli.serve import ServeOptions, run_serve_start, run_serve_status, run_serve_stop
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import TargetDevice, WeightPrecision
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    RAM_RESERVE_DEFAULT_BYTES,
)
from jaull.runtime.quality_eval_runner import QualityProfile, QualityRunRequest

_DEFAULT_RESERVE_GIB = DEVICE_RESERVE_DEFAULT_BYTES / (1024**3)
_DEFAULT_RAM_RESERVE_GIB = RAM_RESERVE_DEFAULT_BYTES / (1024**3)

app = typer.Typer(
    add_completion=False,
    # No `no_args_is_help`: the callback decides between launching the UI and
    # printing help, depending on whether there is a terminal to draw on.
    invoke_without_command=True,
    help="Analyze local hardware and inspect Hugging Face model repositories.",
)
experiments_app = typer.Typer(
    add_completion=False,
    help="Inspect stored experiment records.",
)
app.add_typer(experiments_app, name="experiments")
case_app = typer.Typer(
    add_completion=False,
    help="Group experiment and benchmark evidence into one experimental case.",
)
experiments_app.add_typer(case_app, name="case")
quality_app = typer.Typer(
    add_completion=False,
    help="Run the pinned local quality pilot or compare stored diagnostics (no ranking changes).",
)
app.add_typer(quality_app, name="quality")
case_bundle_app = typer.Typer(
    add_completion=False,
    help="Inspect a portable experimental case bundle offline.",
)
case_app.add_typer(case_bundle_app, name="bundle")
runtime_app = typer.Typer(
    add_completion=False,
    help="Install local runtimes managed by Jaull.",
)
app.add_typer(runtime_app, name="runtime")
serve_app = typer.Typer(
    add_completion=False,
    help="Run a model as a local OpenAI-compatible server (llama-server).",
)
app.add_typer(serve_app, name="serve")


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
    safety_margin_percent: float | None = typer.Option(
        None,
        "--safety-margin-percent",
        min=0.0,
        max=100.0,
        help="Extra percentage added on top of the estimated subtotal.",
    ),
    device_reserve_gib: float = typer.Option(
        _DEFAULT_RESERVE_GIB,
        "--device-reserve-gib",
        min=0.0,
        help="GPU reserve to leave free (GiB); GGUF keeps at least 1 GiB of VRAM.",
    ),
    ram_reserve_gib: float = typer.Option(
        _DEFAULT_RAM_RESERVE_GIB,
        "--ram-reserve-gib",
        min=0.0,
        help="RAM reserve for CPU execution (GiB); independent of the GPU reserve.",
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
    context_sweep: bool = typer.Option(
        False,
        "--context-sweep",
        help=(
            "Estimate the model at several contexts and report the largest one that fits "
            "(replaces the single estimate; not combinable with --context)."
        ),
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
        ram_reserve_gib=ram_reserve_gib,
        as_json=as_json,
        resolve_base_model=not no_resolve_base_model,
        recommend_runtime=not no_runtime_recommendation,
        context_sweep=context_sweep,
    )
    raise typer.Exit(code=run_estimate(reference, options))


@app.command("run", help="Run a resolved GGUF artifact with llama-cli.")
def run_command(
    reference: str = typer.Option(
        ..., "--model", "-m", help="Hugging Face repo_id or URL."
    ),
    quantization: str | None = typer.Option(
        None, "--quantization", "-q", help="GGUF quantization variant to use."
    ),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to send to llama-cli."),
    revision: str | None = typer.Option(
        None, "--revision", "-r", help="Hugging Face revision to resolve."
    ),
    llama_cli: Path | None = typer.Option(
        None,
        "--llama-cli",
        help="Path to llama-cli. Defaults to resolving llama-cli from PATH.",
    ),
    context_size: int | None = typer.Option(
        None,
        "--ctx-size",
        min=1,
        help="Context length passed to llama-cli. Omit for the automatic default (4096).",
    ),
    n_gpu_layers: int | None = typer.Option(
        None,
        "--n-gpu-layers",
        min=-1,
        help="Layers to offload to GPU. Omit for automatic; 0 forces CPU-only; -1 is all.",
    ),
    timeout_seconds: float = typer.Option(
        300.0, "--timeout-seconds", min=0.1, help="llama-cli timeout."
    ),
    full_verify: bool = typer.Option(
        False, "--full-verify", help="Recompute SHA-256 before running."
    ),
) -> None:
    options = RunOptions(
        quantization=quantization,
        prompt=prompt,
        revision=revision,
        llama_cli_path=llama_cli,
        context_size=context_size,
        n_gpu_layers=n_gpu_layers,
        timeout_seconds=timeout_seconds,
        full_verify=full_verify,
    )
    raise typer.Exit(code=run_model(reference, options))


@experiments_app.command(
    "reevaluate",
    help="Re-evaluate a stored experiment with the current prediction logic.",
)
def reevaluate_experiment_command(
    experiment_id: str = typer.Argument(..., metavar="EXPERIMENT_ID"),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(
        code=run_reevaluate(
            experiment_id,
            ReevaluateOptions(as_json=as_json),
        )
    )


@case_app.command(
    "create",
    help="Group one experiment, its benchmarks and its raw evidence as one case.",
)
def create_case_command(
    experiment: str = typer.Option(..., "--experiment", metavar="EXPERIMENT_ID"),
    benchmark: list[str] = typer.Option(
        [],
        "--benchmark",
        metavar="BENCHMARK_ID",
        help="Repeat to attach several benchmark records.",
    ),
    evidence: list[str] = typer.Option(
        [],
        "--evidence",
        metavar="PATH:ROLE",
        help=(
            "Raw evidence file, relative to the current directory. "
            "Roles: prediction, report, runtime_log, timing, other."
        ),
    ),
    label: str | None = typer.Option(None, "--label", help="Human name for the case."),
    note: list[str] = typer.Option([], "--note", help="Repeat to add several notes."),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(
        code=run_create_case(
            CreateCaseOptions(
                experiment_id=experiment,
                benchmark_ids=tuple(benchmark),
                evidence=tuple(evidence),
                label=label,
                note=tuple(note),
                as_json=as_json,
            )
        )
    )


@case_app.command("show", help="Show a stored case and its current consistency.")
def show_case_command(
    case_id: str = typer.Argument(..., metavar="CASE_ID"),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(code=run_show_case(case_id, CaseOptions(as_json=as_json)))


@case_app.command(
    "validate",
    help="Check that the records and files a case names agree with each other.",
)
def validate_case_command(
    case_id: str = typer.Argument(..., metavar="CASE_ID"),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(code=run_validate_case(case_id, CaseOptions(as_json=as_json)))


@case_app.command("export", help="Export a case, its records and evidence as a portable directory.")
def export_case_command(
    case_id: str = typer.Argument(..., metavar="CASE_ID"),
    destination: Path = typer.Argument(..., metavar="DESTINATION"),
    evidence_root: Path = typer.Option(
        Path(),
        "--evidence-root",
        metavar="PATH",
        help="Root used to resolve evidence paths stored in the case.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(
        code=run_export_case(
            case_id,
            ExportCaseOptions(
                destination=destination,
                evidence_root=evidence_root,
                as_json=as_json,
            ),
        )
    )


@case_bundle_app.command(
    "validate",
    help="Validate a portable case bundle without local experiment stores.",
)
def validate_case_bundle_command(
    root: Path = typer.Argument(..., metavar="BUNDLE_DIRECTORY"),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(
        code=run_validate_case_bundle(root, CaseOptions(as_json=as_json))
    )


@case_app.command("list", help="List stored case ids.")
def list_cases_command(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit stable JSON to stdout instead of a Rich report.",
    ),
) -> None:
    raise typer.Exit(code=run_list_cases(CaseOptions(as_json=as_json)))


@runtime_app.command(
    "install", help="Download, verify and install the pinned official llama.cpp build.",
)
def runtime_install_command(
    backend: ComputeBackend = typer.Option(
        ..., "--backend", case_sensitive=False, help="vulkan, cuda or cpu.",
    ),
    latest: bool = typer.Option(
        False, "--latest",
        help="Install the newest upstream build instead (SHA-256 checked, not validated).",
    ),
) -> None:
    raise typer.Exit(code=run_runtime_install(backend, latest=latest))


@serve_app.command(
    "start",
    help="Start llama-server for a model, sized for N users, on this PC or the local network.",
)
def serve_start_command(
    model_ref: str = typer.Argument(..., help="Hugging Face repo id or URL of a GGUF model."),
    quantization: str | None = typer.Option(None, "--quantization", "-q"),
    revision: str | None = typer.Option(None, "--revision"),
    users: int = typer.Option(
        SERVE_DEFAULT_USERS, "--users", min=1, help="People using the server at the same time.",
    ),
    context: int = typer.Option(
        SERVE_DEFAULT_CONTEXT_PER_USER, "--context", min=1,
        help="Largest context one user may reach, in tokens.",
    ),
    lan: bool = typer.Option(
        False, "--lan", help="Listen on the local network (default: this PC only).",
    ),
    port: int | None = typer.Option(None, "--port", min=1, max=65535),
    backend: ComputeBackend | None = typer.Option(
        None, "--backend", case_sensitive=False, help="Override the detected backend.",
    ),
    llama_server: Path | None = typer.Option(None, "--llama-server"),
    tls_cert: Path | None = typer.Option(None, "--tls-cert", help="PEM certificate for HTTPS."),
    tls_key: Path | None = typer.Option(None, "--tls-key", help="PEM private key for HTTPS."),
    web_ui: bool = typer.Option(
        False, "--web-ui", help="Also serve llama.cpp's chat page (it asks for the API key).",
    ),
    shared_context: bool = typer.Option(
        False, "--shared-context",
        help="One pool of --context tokens for everyone: one person can use all of it.",
    ),
) -> None:
    raise typer.Exit(code=run_serve_start(model_ref, ServeOptions(
        quantization=quantization, users=users, context_per_user=context, lan=lan, port=port,
        backend=backend, revision=revision, llama_server_path=llama_server,
        tls_cert=tls_cert, tls_key=tls_key, web_ui=web_ui, shared_context=shared_context,
    )))


@serve_app.command("stop", help="Stop the server started by `jaull serve start`.")
def serve_stop_command() -> None:
    raise typer.Exit(code=run_serve_stop())


@serve_app.command("status", help="Show whether the Jaull server is running and ready.")
def serve_status_command() -> None:
    raise typer.Exit(code=run_serve_status())


@quality_app.command("setup", help="Check local evaluator; optionally prepare its pinned dataset.")
def quality_setup_command(
    dataset_file: Path | None = typer.Option(None, "--dataset-file"),
    llama_server: Path | None = typer.Option(None, "--llama-server"),
    pilot_root: Path | None = typer.Option(None, "--pilot-root"),
    image: str | None = typer.Option(None, "--image"),
    allow_dataset_download: bool = typer.Option(False, "--allow-dataset-download"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    raise typer.Exit(code=run_quality_setup(
        dataset_file=dataset_file, llama_server=llama_server, pilot_root=pilot_root,
        image=image, allow_dataset_download=allow_dataset_download, as_json=as_json,
    ))


@quality_app.command(
    "run",
    help="Evaluate explicit local GGUFs sequentially; never download/build.",
)
def quality_run_command(
    artifact: list[Path] = typer.Option(
        ...,
        "--artifact",
        help="ModelArtifact JSON. Repeat for sequential evaluation.",
    ),
    dataset_file: Path = typer.Option(
        ...,
        "--dataset-file",
        help="Pinned local validation Parquet.",
    ),
    llama_server: Path = typer.Option(
        ...,
        "--llama-server",
        help="Audited host llama-server binary.",
    ),
    image: str = typer.Option(..., "--image", help="Already-built evaluator image (never pulled)."),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Fresh campaign directory for logs and bundles.",
    ),
    pilot_root: Path = typer.Option(
        Path(),
        "--pilot-root",
        help="Trusted Jaull source-checkout root.",
    ),
    profile: QualityProfile = typer.Option(
        QualityProfile.SMOKE,
        "--profile",
        help="Fixed smoke, limited 100-sample or full IFEval chat profile.",
    ),
    backend: ComputeBackend | None = typer.Option(
        None,
        "--backend",
        help="Evaluation placement: cpu, cuda or vulkan; default auto. No explicit-GPU fallback.",
    ),
    external_server: Path | None = typer.Option(
        None, "--external-server", help="Prepared external Windows Vulkan IFEval manifest.",
    ),
    server_endpoint: str | None = typer.Option(
        None, "--server-endpoint",
        help="WSL loopback HTTP origin, also used by the host-network container.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit completion ids or an explicit failure.",
    ),
) -> None:
    raise typer.Exit(
        code=run_quality(
            [
                QualityRunRequest(
                    artifact_json=path,
                    dataset_file=dataset_file,
                    llama_server=llama_server,
                    image=image,
                    output=output / f"run-{index:02d}",
                    pilot_root=pilot_root,
                    profile=profile,
                    backend=backend,
                    external_server=external_server,
                    server_endpoint=server_endpoint,
                )
                for index, path in enumerate(artifact, start=1)
            ],
            as_json=as_json,
        )
    )


@quality_app.command(
    "compare", help="Compare stored records; incompatible protocols stay separate.",
)
def quality_compare_command(
    left_id: str = typer.Argument(..., metavar="LEFT_ID"),
    right_id: str = typer.Argument(..., metavar="RIGHT_ID"),
    as_json: bool = typer.Option(False, "--json", help="Emit per-task comparison with provenance."),
) -> None:
    raise typer.Exit(code=run_quality_compare(left_id, right_id, as_json=as_json))


if __name__ == "__main__":
    app()
