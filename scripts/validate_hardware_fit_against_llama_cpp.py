#!/usr/bin/env python
"""Check the Hardware Fit Analyzer's prediction against a real llama.cpp run.

    uv run python scripts/validate_hardware_fit_against_llama_cpp.py \
        --model /path/to/model.gguf --context 4096
    uv run python scripts/validate_hardware_fit_against_llama_cpp.py \
        --model /path/to/model.gguf --context 4096 --sweep --kv-offload-test

This is a measurement harness, not a test. It never recalibrates anything: the
prediction is computed and frozen *before* llama.cpp is launched, and the report
only states the error.

## What it does

1. Asks Jaull — `AdvisorService` for the hardware scan and the memory estimate,
   `estimator.hardware_fit.analyze_estimate` for the placement — so the
   prediction comes from production code rather than a re-derivation here.
2. Runs llama.cpp at `-ngl 0` for a CPU baseline, then at the predicted layer
   count, sampling VRAM throughout.
3. Optionally sweeps `-ngl` around the prediction to find the real maximum that
   still starts, and optionally repeats one run with `--no-kv-offload`.

## Measurement caveats, stated once

* **VRAM is device-wide.** `nvidia-smi --query-compute-apps` reports nothing
  under WSL, so the harness samples total `memory.used` and subtracts an idle
  baseline taken before the run. Anything else touching the GPU (a desktop
  compositor) lands in that baseline, and a change mid-run lands in the result.
* **The CUDA context is not modelled by Jaull.** A few hundred MiB of the
  measured delta is driver context that no `HardwareFitResult` field predicts.
  The report therefore also carries llama.cpp's own buffer accounting, which is
  the closer comparison.
* **RSS is not `ram_required_bytes`.** Peak RSS includes the mmap'd weights
  that were never faulted in and excludes memory the driver holds. It is
  reported as an observation, not as the same quantity.

Nothing is downloaded. The script exits 2 (not 1) when the machine simply
cannot run the validation — no GPU, no llama.cpp, no model — so a caller can
tell "cannot run here" from "ran and failed".
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaull.advisor.service import AdvisorService
from jaull.domain.estimation import HardwareFitResult, MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.estimator import hardware_fit
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from jaull.runtime.locator import RuntimeLocator

MIB = 1024**2
GIB = 1024**3

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CANNOT_RUN = 2

# llama.cpp startup lines worth keeping verbatim. Matching is deliberately
# loose — the wording drifts between builds, and a missed line should cost a
# row in the report, never the run.
_LOG_KEEP = re.compile(
    r"(offloa\w+|buffer size|n_ctx\s|flash_attn|KV self|model size|graph nodes"
    r"|system_info|device_info|CUDA\d|out of memory|failed|error)",
    re.IGNORECASE,
)
_OFFLOADED = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", re.IGNORECASE)
_BUFFER = re.compile(
    r"^(?P<what>.+?)\s+(?P<kind>model|KV|compute|output)\s+buffer size\s*=\s*"
    r"(?P<mib>[\d.]+)\s*MiB",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prediction — computed once, before anything runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    repo_id: str
    quantization: str
    context_length: int
    hardware: HardwareProfile
    estimate: MemoryEstimate
    fit: HardwareFitResult

    def as_dict(self) -> dict[str, Any]:
        fit = self.fit
        estimate = self.estimate
        gpu = _first_gpu(self.hardware)
        return {
            "repo_id": self.repo_id,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "hardware": {
                "gpu": gpu.name if gpu else None,
                "vram_total_bytes": gpu.vram_total_bytes if gpu else None,
                "vram_available_bytes": gpu.vram_available_bytes if gpu else None,
                "ram_total_bytes": self.hardware.memory.total_bytes,
                "ram_available_bytes": self.hardware.memory.available_bytes,
            },
            "estimate": {
                "weights_bytes": estimate.weights.component.bytes,
                "kv_cache_bytes": estimate.kv_cache.component.bytes,
                "kv_cache_layers": estimate.kv_cache.layers,
                "kv_cache_formula": estimate.kv_cache.formula,
                "runtime_overhead_bytes": estimate.runtime_overhead.component.bytes,
                "device_reserve_bytes": estimate.device_reserve.bytes,
                "safety_margin_bytes": (
                    estimate.safety_margin.bytes if estimate.safety_margin else 0
                ),
                "total_bytes": estimate.total_bytes,
                "status": estimate.assessment.status.value,
                "confidence": estimate.assessment.confidence.value,
            },
            "fit": {
                "mode": fit.mode.value,
                "memory_topology": fit.memory_topology.value,
                "placement_method": fit.placement_method.value,
                "gpu_layers": fit.gpu_layers,
                "total_layers": fit.total_layers,
                "gpu_required_bytes": fit.gpu_required_bytes,
                "ram_required_bytes": fit.ram_required_bytes,
                "gpu_weight_bytes": fit.gpu_weight_bytes,
                "ram_weight_bytes": fit.ram_weight_bytes,
                "gpu_overhead_bytes": fit.gpu_overhead_bytes,
                "ram_overhead_bytes": fit.ram_overhead_bytes,
                "gpu_safety_margin_bytes": fit.gpu_safety_margin_bytes,
                "ram_safety_margin_bytes": fit.ram_safety_margin_bytes,
                "reason": fit.reason,
                "warnings": list(fit.warnings),
            },
        }


def predict(
    advisor: AdvisorService,
    *,
    repo_id: str,
    quantization: str,
    context_length: int,
    concurrent_users: int,
) -> Prediction:
    hardware = advisor.scan_hardware()
    analysis = advisor.inspect_model(repo_id)
    inference_cfg = InferenceConfiguration(
        context_length=context_length,
        batch_size=1,
        target_device=TargetDevice.AUTO,
        quantization=quantization,
        safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
        device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
        concurrent_users=concurrent_users,
    )
    estimate = advisor.estimate_model(
        analysis, hardware, inference_cfg, recommend_runtime=False
    )
    fit = hardware_fit.analyze_estimate(estimate, hardware)
    if fit is None:
        raise SystemExit(
            "The estimate is missing a memory component, so no placement can be "
            "analyzed. Nothing was run."
        )
    return Prediction(
        repo_id=repo_id,
        quantization=quantization,
        context_length=context_length,
        hardware=hardware,
        estimate=estimate,
        fit=fit,
    )


# ---------------------------------------------------------------------------
# Observation — llama.cpp
# ---------------------------------------------------------------------------


@dataclass
class VramSampler:
    """Device-wide VRAM sampling, because WSL reports no per-process figure."""

    interval_seconds: float = 0.2
    samples: list[int] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    @staticmethod
    def read_used_mib() -> int | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
        first = out.splitlines()[0].strip() if out else ""
        return int(first) if first.isdigit() else None

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self.read_used_mib()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def peak_mib(self) -> int | None:
        return max(self.samples) if self.samples else None


@dataclass
class Run:
    gpu_layers: int | str
    command: list[str]
    exit_code: int
    seconds: float
    started: bool
    offloaded: tuple[int, int] | None
    buffers_mib: dict[str, float]
    vram_baseline_mib: int | None
    vram_peak_mib: int | None
    peak_rss_bytes: int | None
    log_path: str
    log_excerpt: list[str]
    oom: bool

    @property
    def vram_delta_mib(self) -> int | None:
        if self.vram_peak_mib is None or self.vram_baseline_mib is None:
            return None
        return max(0, self.vram_peak_mib - self.vram_baseline_mib)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_layers": self.gpu_layers,
            "command": " ".join(self.command),
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 2),
            "started": self.started,
            "oom": self.oom,
            "offloaded_layers": (
                f"{self.offloaded[0]}/{self.offloaded[1]}" if self.offloaded else None
            ),
            "buffers_mib": self.buffers_mib,
            "vram_baseline_mib": self.vram_baseline_mib,
            "vram_peak_mib": self.vram_peak_mib,
            "vram_delta_mib": self.vram_delta_mib,
            "peak_rss_bytes": self.peak_rss_bytes,
            "log": self.log_path,
        }


def run_llama(
    *,
    llama_cli: str,
    model: Path,
    gpu_layers: int | str,
    context_length: int,
    tokens: int,
    prompt: str,
    log_dir: Path,
    timeout_seconds: int,
    extra_args: tuple[str, ...] = (),
    tag: str = "",
) -> Run:
    """One llama.cpp run, instrumented.

    ``--fit off`` is not optional: left on, llama.cpp silently adjusts unset
    arguments to fit device memory, which would quietly answer a different
    question than the one being validated. ``-st`` makes the run single-turn
    (without it llama-cli blocks on stdin), and ``-v`` is required because the
    buffer-size lines are above the default verbosity threshold.
    """
    label = tag or f"ngl{gpu_layers}"
    log_path = log_dir / f"llama-{label}.log"
    time_path = log_dir / f"time-{label}.txt"
    command = [
        llama_cli,
        "-m",
        str(model),
        "-ngl",
        str(gpu_layers),
        "-c",
        str(context_length),
        "-n",
        str(tokens),
        "--fit",
        "off",
        "-no-cnv",
        "-st",
        "-v",
        "-p",
        prompt,
        *extra_args,
    ]
    wrapped = command
    gnu_time = shutil.which("time") or "/usr/bin/time"
    if Path(gnu_time).exists():
        wrapped = [gnu_time, "-v", "-o", str(time_path), *command]

    sampler = VramSampler()
    baseline = VramSampler.read_used_mib()
    sampler.start()
    started_at = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as handle, open(
            "/dev/null", "rb"
        ) as devnull:
            completed = subprocess.run(
                wrapped,
                stdin=devnull,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1
    finally:
        elapsed = time.monotonic() - started_at
        sampler.stop()

    text = log_path.read_text(encoding="utf-8", errors="replace")
    return Run(
        gpu_layers=gpu_layers,
        command=command,
        exit_code=exit_code,
        seconds=elapsed,
        started=_started_generating(text),
        offloaded=_parse_offloaded(text),
        buffers_mib=_parse_buffers(text),
        vram_baseline_mib=baseline,
        vram_peak_mib=sampler.peak_mib(),
        peak_rss_bytes=_parse_peak_rss(time_path),
        log_path=str(log_path),
        log_excerpt=_log_excerpt(text),
        oom=_looks_like_oom(text),
    )


def _started_generating(text: str) -> bool:
    """Did the model actually load and produce a context?"""
    return "llama_context:" in text and not _looks_like_oom(text)


def _looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    return any(
        needle in lowered
        for needle in (
            "out of memory",
            "cudamalloc failed",
            "failed to allocate",
            "unable to allocate",
        )
    )


def _parse_offloaded(text: str) -> tuple[int, int] | None:
    """The last report wins: llama.cpp prints it once per model load."""
    matches = _OFFLOADED.findall(text)
    if not matches:
        return None
    offloaded, total = matches[-1]
    return int(offloaded), int(total)


def _parse_buffers(text: str) -> dict[str, float]:
    buffers: dict[str, float] = {}
    for line in text.splitlines():
        match = _BUFFER.search(line)
        if match is None:
            continue
        where = match.group("what").split()[-1]
        key = f"{where} {match.group('kind').lower()}"
        buffers[key] = max(buffers.get(key, 0.0), float(match.group("mib")))
    return buffers


def _parse_peak_rss(time_path: Path) -> int | None:
    if not time_path.exists():
        return None
    for line in time_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size" in line:
            digits = re.search(r"(\d+)", line)
            if digits:
                return int(digits.group(1)) * 1024
    return None


def _log_excerpt(text: str, limit: int = 30) -> list[str]:
    kept = [line.strip() for line in text.splitlines() if _LOG_KEEP.search(line)]
    return kept[:limit]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _first_gpu(hardware: HardwareProfile) -> Any:
    return hardware.gpus[0] if hardware.gpus else None


def find_llama_cli(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    installation = RuntimeLocator().resolve_llama_cpp()
    if installation.llama_cli and Path(installation.llama_cli).exists():
        return installation.llama_cli
    return shutil.which("llama-cli")


_QUANT_IN_NAME = re.compile(
    r"(iq\d[a-z_]*|q\d_k_[sml]|q\d_k|q\d_\d|bf16|fp16|f16|f32)", re.IGNORECASE
)


def quantization_from(model: Path, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    match = _QUANT_IN_NAME.search(model.stem)
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct_error(observed: float | None, predicted: float | None) -> float | None:
    if not observed or predicted is None:
        return None
    return abs(observed - predicted) / observed * 100.0


def _cuda_buffers_mib(run: Run) -> float | None:
    """Sum only the device-side buffers llama.cpp reports.

    ``CUDA_Host`` is pinned *host* memory despite the name, so it is excluded
    along with the ``CPU`` and ``CPU_Mapped`` buffers.
    """
    total = sum(
        value
        for name, value in run.buffers_mib.items()
        if name.upper().startswith("CUDA") and not name.upper().startswith("CUDA_HOST")
    )
    return total or None


def build_report(
    prediction: Prediction,
    baseline: Run,
    predicted_run: Run,
    sweep: list[Run],
    kv_runs: list[Run],
    llama_version: str,
) -> dict[str, Any]:
    fit = prediction.fit
    predicted_vram_mib = (
        fit.gpu_required_bytes / MIB if fit.gpu_required_bytes is not None else None
    )
    observed_vram_mib = predicted_run.vram_delta_mib
    # Only the CUDA0 buffers live in VRAM. CPU_Mapped / CPU / CUDA_Host ones are
    # host memory, and summing all of them would compare a VRAM prediction
    # against a figure that is partly system RAM.
    observed_buffers_mib = _cuda_buffers_mib(predicted_run)

    starting = [run for run in sweep if run.started]
    real_max_layers = max(
        (int(run.gpu_layers) for run in starting if isinstance(run.gpu_layers, int)),
        default=None,
    )
    predicted_layers = fit.gpu_layers

    return {
        "llama_cpp_version": llama_version,
        "prediction": prediction.as_dict(),
        "runs": {
            "cpu_baseline": baseline.as_dict(),
            "at_predicted_layers": predicted_run.as_dict(),
            "sweep": [run.as_dict() for run in sweep],
            "kv_offload": [run.as_dict() for run in kv_runs],
        },
        "metrics": {
            "predicted_vram_mib": predicted_vram_mib,
            "observed_vram_device_delta_mib": observed_vram_mib,
            "observed_vram_llama_buffers_mib": observed_buffers_mib,
            "vram_error_pct_vs_device_delta": _pct_error(
                observed_vram_mib, predicted_vram_mib
            ),
            "vram_error_pct_vs_llama_buffers": _pct_error(
                observed_buffers_mib, predicted_vram_mib
            ),
            "observed_vram_llama_buffers_note": (
                "CUDA0 device buffers only; CPU/CPU_Mapped/CUDA_Host are host memory"
            ),
            "predicted_gpu_layers": predicted_layers,
            "real_max_gpu_layers": real_max_layers,
            "layer_difference": (
                real_max_layers - predicted_layers
                if real_max_layers is not None and predicted_layers is not None
                else None
            ),
            "predicted_ram_bytes": fit.ram_required_bytes,
            "observed_peak_rss_bytes": predicted_run.peak_rss_bytes,
            "cpu_baseline_peak_rss_bytes": baseline.peak_rss_bytes,
        },
    }


def render(report: dict[str, Any]) -> str:
    prediction = report["prediction"]
    fit = prediction["fit"]
    estimate = prediction["estimate"]
    metrics = report["metrics"]
    lines: list[str] = [
        "Hardware fit prediction vs llama.cpp",
        "=" * 78,
        f"  model        {prediction['repo_id']}  {prediction['quantization']}",
        f"  context      {prediction['context_length']}",
        f"  llama.cpp    {report['llama_cpp_version']}",
        f"  GPU          {prediction['hardware']['gpu']}",
        f"  VRAM avail   {_mib(prediction['hardware']['vram_available_bytes'])}",
        f"  RAM avail    {_mib(prediction['hardware']['ram_available_bytes'])}",
        "",
        "PREDICTION (frozen before any run)",
        f"  mode              {fit['mode']}",
        f"  placement         {fit['placement_method']}",
        f"  gpu_layers        {fit['gpu_layers']} / {fit['total_layers']}",
        f"  gpu_required      {_mib(fit['gpu_required_bytes'])}",
        f"  ram_required      {_mib(fit['ram_required_bytes'])}",
        f"  weights           {_mib(estimate['weights_bytes'])}",
        f"  kv_cache          {_mib(estimate['kv_cache_bytes'])}",
        f"  runtime_overhead  {_mib(estimate['runtime_overhead_bytes'])}",
        f"  device_reserve    {_mib(estimate['device_reserve_bytes'])}",
        f"  safety_margin     {_mib(estimate['safety_margin_bytes'])}",
        f"  status/conf       {estimate['status']} / {estimate['confidence']}",
        "",
        "OBSERVED",
    ]
    lines.append(
        f"  {'-ngl':>6}  {'started':>7}  {'offloaded':>10}  "
        f"{'VRAM d':>9}  {'CUDA buf':>9}  {'RSS':>10}"
    )
    runs = [report["runs"]["cpu_baseline"], report["runs"]["at_predicted_layers"]]
    runs.extend(report["runs"]["sweep"])
    runs.extend(report["runs"]["kv_offload"])
    seen: set[str] = set()
    for run in runs:
        key = f"{run['gpu_layers']}|{run['command']}"
        if key in seen:
            continue
        seen.add(key)
        buffers = _cuda_only(run["buffers_mib"])
        lines.append(
            f"  {run['gpu_layers']!s:>6}  {run['started']!s:>7}  "
            f"{run['offloaded_layers'] or '-'!s:>10}  "
            f"{_mib_num(run['vram_delta_mib']):>9}  {_mib_num(buffers):>9}  "
            f"{_mib(run['peak_rss_bytes']):>10}"
        )
    lines.extend(
        [
            "",
            "METRICS",
            f"  predicted VRAM              {_mib_num(metrics['predicted_vram_mib'])}",
            f"  observed VRAM (device Δ)    {_mib_num(metrics['observed_vram_device_delta_mib'])}",
            f"  observed VRAM (llama bufs)  {_mib_num(metrics['observed_vram_llama_buffers_mib'])}",
            f"  VRAM error vs device Δ      {_pct(metrics['vram_error_pct_vs_device_delta'])}",
            f"  VRAM error vs llama bufs    {_pct(metrics['vram_error_pct_vs_llama_buffers'])}",
            f"  predicted gpu_layers        {metrics['predicted_gpu_layers']}",
            f"  real max gpu_layers         {metrics['real_max_gpu_layers']}",
            f"  layer difference            {metrics['layer_difference']}",
            f"  predicted RAM               {_mib(metrics['predicted_ram_bytes'])}",
            f"  observed peak RSS           {_mib(metrics['observed_peak_rss_bytes'])}",
            "",
            "  RSS is not ram_required_bytes: it counts mmap'd weight pages that",
            "  were faulted in and excludes driver-side allocations.",
        ]
    )
    return "\n".join(lines)


def _cuda_only(buffers: dict[str, float]) -> float | None:
    total = sum(
        value
        for name, value in buffers.items()
        if name.upper().startswith("CUDA") and not name.upper().startswith("CUDA_HOST")
    )
    return total or None


def _mib(value: int | None) -> str:
    return "-" if value is None else f"{value / MIB:.1f} MiB"


def _mib_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    model = Path(args.model).expanduser()
    if not model.exists():
        print(f"Model not found: {model}. Nothing is downloaded.", file=sys.stderr)
        return EXIT_CANNOT_RUN

    llama_cli = find_llama_cli(args.llama_cli)
    if llama_cli is None:
        print(
            "No llama-cli found (checked --llama-cli, Jaull's runtime locator and "
            "PATH). Nothing was run.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN

    quantization = quantization_from(model, args.quantization)
    if quantization is None:
        print(
            f"Cannot infer a quantization from {model.name!r}; pass --quantization.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN

    advisor = AdvisorService.default(llama_cli_path=llama_cli)
    prediction = predict(
        advisor,
        repo_id=args.repo_id,
        quantization=quantization,
        context_length=args.context,
        concurrent_users=args.concurrent_users,
    )

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the prediction on disk before a single token is generated, so it
    # cannot be retro-fitted to the observation.
    (log_dir / "prediction.json").write_text(
        json.dumps(prediction.as_dict(), indent=2), encoding="utf-8"
    )
    print(json.dumps(prediction.as_dict(), indent=2))
    print("\nPrediction frozen. Launching llama.cpp...\n", file=sys.stderr)

    version = _llama_version(llama_cli)
    common = {
        "llama_cli": llama_cli,
        "model": model,
        "context_length": args.context,
        "tokens": args.tokens,
        "prompt": args.prompt,
        "log_dir": log_dir,
        "timeout_seconds": args.timeout,
    }

    baseline = run_llama(gpu_layers=0, **common)  # type: ignore[arg-type]
    predicted_layers = prediction.fit.gpu_layers
    target = predicted_layers if predicted_layers is not None else 0
    predicted_run = run_llama(gpu_layers=target, tag="predicted", **common)  # type: ignore[arg-type]

    sweep: list[Run] = []
    if args.sweep:
        total = prediction.fit.total_layers or target
        candidates = sorted(
            {
                value
                for value in (target - 3, target, target + 1, target + 3, total + 1)
                if 0 <= value <= total + 1
            }
        )
        for value in candidates:
            sweep.append(run_llama(gpu_layers=value, **common))  # type: ignore[arg-type]

    kv_runs: list[Run] = []
    if args.kv_offload_test:
        kv_runs.append(
            run_llama(
                gpu_layers=target,
                extra_args=("--no-kv-offload",),
                tag="nokvoffload",
                **common,  # type: ignore[arg-type]
            )
        )

    report = build_report(prediction, baseline, predicted_run, sweep, kv_runs, version)
    (log_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\nArtifacts: {log_dir}")
    return EXIT_OK


def _llama_version(llama_cli: str) -> str:
    try:
        out = subprocess.run(
            [llama_cli, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = (out.stdout + out.stderr).strip().splitlines()
    return text[0] if text else "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a local .gguf file.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo the artifact came from, for Jaull's estimate.",
    )
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--concurrent-users", type=int, default=1)
    parser.add_argument("--quantization", help="Default: inferred from the filename.")
    parser.add_argument("--llama-cli", help="Default: Jaull's locator, then PATH.")
    parser.add_argument(
        "--prompt", default="Explain briefly what a local language model is."
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--log-dir", default="./hardware-fit-validation")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Search for the real maximum -ngl around the prediction.",
    )
    parser.add_argument(
        "--kv-offload-test",
        action="store_true",
        help="Repeat the predicted run with --no-kv-offload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
