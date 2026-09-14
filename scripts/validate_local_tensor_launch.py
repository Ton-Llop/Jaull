"""Freeze a local tensor launch budget, then optionally measure it and controls.

Uses an existing experiment's weight/KV inputs, not a reconstructed model.
Offline mode retains recorded hardware. --execute replaces only the available
VRAM launch budget with detected free VRAM, recording both inputs separately.
HFA predictions and historical observations are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from jaull.application.execution.planner import plan_launch
from jaull.domain.benchmarks import BenchmarkGpuLayers, BenchmarkRequest
from jaull.domain.execution import ExecutionRequest
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import LlamaCppRuntimeCapability, RuntimeName
from jaull.execution.errors import ExecutionError
from jaull.execution.host import HostExecutionBackend
from jaull.hardware.detector import detect_hardware
from jaull.observability.provenance import capture_git_commit
from jaull.runtime.backend_selection import select_runtime_backend
from jaull.runtime.llama_bench_capability import inspect_llama_bench
from jaull.runtime.llama_bench_parser import parse_llama_bench_output
from jaull.runtime.llama_bench_runner import build_llama_bench_command
from jaull.runtime.llama_cpp_capability import inspect_llama_cpp_runtime
from jaull.runtime.llama_cpp_launch_policy import pick_gpu_layers
from jaull.runtime.llama_cpp_tensor_policy import (
    inspect_tensor_context,
    select_tensor_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--llama-bench", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    original = args.record.read_bytes()
    record = ExperimentRecord.model_validate_json(original)
    hardware = record.hardware
    capability = record.preflight.runtime_capability
    estimate = record.prediction
    backend = HostExecutionBackend()
    if args.execute:
        if args.llama_cli is None or args.llama_bench is None:
            parser.error("--execute requires --llama-cli and --llama-bench")
        hardware = detect_hardware()
        capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=args.llama_cli)
        if len(hardware.gpus) != 1 or hardware.gpus[0].vram_available_bytes is None:
            parser.error("A single GPU with available VRAM is required")
        estimate = estimate.model_copy(
            update={
                "assessment": estimate.assessment.model_copy(
                    update={"available_vram_bytes": hardware.gpus[0].vram_available_bytes}
                )
            }
        )
    if not isinstance(capability, LlamaCppRuntimeCapability):
        parser.error("A recorded llama.cpp capability is required")
    selection = select_runtime_backend(hardware)
    bench_capability = None
    if args.execute:
        bench_capability = inspect_llama_bench(
            backend=backend,
            llama_bench_path=args.llama_bench,
            allow_empty_workload_probe=True,
        )
    context = inspect_tensor_context(
        record.artifact,
        hardware,
        selection,
        capability,
        benchmark_capability=bench_capability,
    )
    budget = select_tensor_budget(estimate, context)
    if isinstance(budget, str) or context.index is None:
        parser.error(f"Tensor refinement unavailable: {budget}")
    launch = plan_launch(
        runtime=RuntimeName.LLAMA_CPP, estimate=estimate, hardware=hardware, tensor_context=context
    )
    automatic = budget.requested_units
    commands: list[tuple[str, tuple[str, ...]]] = []
    if args.execute:
        commands.append(
            (
                "launch-smoke",
                (
                    str(args.llama_cli),
                    "-m",
                    str(record.artifact.local_path),
                    "--ctx-size",
                    str(estimate.inference_configuration.context_length),
                    "--n-gpu-layers",
                    str(automatic),
                    "--n-predict",
                    "32",
                    "--single-turn",
                    "--simple-io",
                    "--no-display-prompt",
                    "--color",
                    "off",
                    "--prompt",
                    record.workload.prompt if record.workload else "Explain local AI briefly.",
                ),
            )
        )
        for units in dict.fromkeys([automatic, 24, 28, -1]):
            request = BenchmarkRequest(
                artifact=record.artifact,
                runtime=launch,
                backend=ComputeBackend.CUDA,
                device="CUDA0",
                gpu_layers=(
                    BenchmarkGpuLayers.full()
                    if units == -1
                    else BenchmarkGpuLayers.count_layers(units)
                ),
                prefill_sizes=(512,),
                generation_sizes=(128,),
                repetitions=3,
            )
            commands.append(
                (f"bench-{units}", build_llama_bench_command(str(args.llama_bench), request))
            )
    # A new directory and exclusive writes prevent overwriting previous evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "local_tensor_launch_validation_v1",
        "source_record": str(args.record),
        "source_record_sha256": hashlib.sha256(original).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jaull_git_commit": capture_git_commit(),
        "source_sha256": {
            path: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / path).read_bytes()
            ).hexdigest()
            for path in (
                "src/jaull/runtime/llama_cpp_tensor_policy.py",
                "src/jaull/runtime/llama_cpp_launch_policy.py",
                "src/jaull/metadata/gguf_reader.py",
                "src/jaull/runtime/policies.py",
            )
        },
        "hardware": hardware.model_dump(mode="json"),
        "runtime_capability": capability.model_dump(mode="json"),
        "benchmark_capability": bench_capability.model_dump(mode="json")
        if bench_capability
        else None,
        "estimate_input": estimate.model_dump(mode="json"),
        "original_available_vram_bytes": record.prediction.assessment.available_vram_bytes,
        "aggregate_units": pick_gpu_layers(estimate).n_gpu_layers,
        "launch": launch.model_dump(mode="json"),
        "budget": {**asdict(budget), "required_bytes": budget.required_bytes},
        "tensors": [asdict(t) for t in context.index.tensors],
        "artifact": record.artifact.model_dump(mode="json"),
        "commands": [{"label": label, "command": command} for label, command in commands],
        "limitations": [
            "GPU tensor bytes are stored payload plus alignment, not CUDA measurements.",
            "Original weight/KV inputs are reused; "
            "only launch VRAM availability changes in live mode.",
            "Artifact hash is carried from the record, not reverified by this script.",
            "pp512/tg128 are microbenchmarks, not context-4096 workloads.",
            "llama-bench default warmup; three repetitions, one session, sequential controls.",
            "The desktop shares the GPU; memory occupancy and thermals may drift.",
        ],
    }
    with (args.output / "prediction.json").open("x") as stream:
        json.dump(snapshot, stream, indent=2)
    print(json.dumps(snapshot["budget"], indent=2), flush=True)
    failed = False
    for label, command in commands:
        print(f"Running {label}: {command}", flush=True)
        try:
            result = backend.execute(ExecutionRequest(command=command, timeout_seconds=600))
        except ExecutionError as exc:
            failed = True
            result = getattr(exc, "result", None)
            with (args.output / f"{label}-error.txt").open("x") as stream:
                stream.write(str(exc))
            if result is None:
                continue
        with (args.output / f"{label}.json").open("x") as stream:
            stream.write(result.model_dump_json(indent=2))
        for name, output in (("stdout", result.stdout), ("stderr", result.stderr)):
            with (args.output / f"{label}-{name}.log").open("x") as stream:
                stream.write(output)
        if label.startswith("bench-") and result.exit_code == 0:
            measurements = parse_llama_bench_output(result.stdout, repetitions=3)
            with (args.output / f"{label}-measurements.json").open("x") as stream:
                json.dump([m.model_dump(mode="json") for m in measurements], stream, indent=2)
            print([(m.kind.value, m.mean_tokens_per_second) for m in measurements], flush=True)
    assert args.record.read_bytes() == original
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
