"""Run a fixed GGUF smoke or limited evaluation; own and clean up its processes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from pilot.quality_eval.evaluate import (
    CONTEXT,
    PROFILES,
    SERVER_SHA256,
    validate_artifact_identity,
    write_json,
)
from pilot.quality_eval.ifeval import CONTEXT as IFEVAL_CONTEXT
from pilot.quality_eval.setup import docker_host_network, inspect_runtime, prepare_infrastructure

from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import HardwareFitMode, MemoryEstimate
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.domain.runtime import ExecutionReadinessStatus
from jaull.estimator.policies import DEVICE_RESERVE_DEFAULT_BYTES, SAFETY_MARGIN_DEFAULT_PERCENT
from jaull.estimator.service import estimate_memory
from jaull.evaluation.quality_records import digest
from jaull.evaluation.quality_runtime import (
    QUALITY_RUNTIME_PROTOCOL,
    fingerprint_from_version,
    validate_quality_memory,
    validate_runtime_execution,
)
from jaull.execution.host import HostExecutionBackend
from jaull.hardware.detector import detect_hardware
from jaull.metadata.config_merger import merge
from jaull.metadata.gguf_reader import parse_header
from jaull.runtime.backend_selection import select_runtime_backend
from jaull.runtime.device_matching import match_runtime_device
from jaull.runtime.llama_cpp_capability import (
    evaluate_execution_readiness,
    inspect_llama_cpp_runtime,
)
from jaull.runtime.quality_execution import (
    quality_backend_selection,
    quality_memory_hardware,
)


class OfflineResolver:
    def resolve(self, repo_id, *, quantization=None, revision=None):
        raise RuntimeError("The quality smoke never resolves or downloads a model")


def embedded_chat_template(raw_kv: dict[str, object]) -> str:
    template = raw_kv.get("tokenizer.chat_template")
    # llama.cpp substitutes built-in ChatML for missing/empty/'chatml' metadata.
    if not isinstance(template, str) or not template.strip() or template == "chatml":
        raise ValueError("IFEval requires an embedded GGUF chat template, not a runtime fallback")
    return template


def chat_template_evidence(path: Path) -> dict[str, str]:
    """Digests the importer compares with the template the server reports.

    llama-server was observed serving the embedded template with trailing
    whitespace stripped, so both forms are recorded; nothing else is accepted.
    """
    with path.open("rb") as model:
        header = parse_header(model.read(32 * 1024**2))
    if header is None:
        raise ValueError("Local GGUF header could not be read")
    template = embedded_chat_template(header.raw_kv)
    return {
        "source": "gguf",
        "embedded_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "embedded_stripped_sha256": hashlib.sha256(template.strip().encode()).hexdigest(),
    }


def artifact_memory_estimate(
    artifact: ModelArtifact,
    hardware: HardwareProfile,
    *,
    context: int = CONTEXT,
    require_chat_template: bool = False,
    backend: ComputeBackend | None = None,
) -> MemoryEstimate:
    assert artifact.local_path is not None and artifact.size_bytes is not None
    # ponytail: retain the bounded 32 MiB metadata parser; larger headers fail closed.
    with artifact.local_path.open("rb") as model:
        header = parse_header(model.read(32 * 1024**2))
    enriched = merge(header, None)
    if enriched is None or header is None or (header.context_length or 0) < context:
        raise ValueError("Local GGUF lacks the required configuration/context")
    if require_chat_template:
        embedded_chat_template(header.raw_kv)
    if (
        type(header.raw_kv.get("split.count", 1)) is not int
        or header.raw_kv.get("split.count", 1) != 1
        or type(header.raw_kv.get("split.no", 0)) is not int
        or header.raw_kv.get("split.no", 0) != 0
    ):
        raise ValueError("Multipart GGUF quality evaluation is unsupported")
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id=artifact.repo_id),
        config=enriched.config,
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            gguf_variants=[GgufVariant(
                quantization=artifact.quantization,
                files=[ModelFile(path=artifact.filename, size_bytes=artifact.size_bytes)],
                total_bytes=artifact.size_bytes,
            )],
        ),
    )
    estimate = estimate_memory(
        analysis,
        hardware,
        InferenceConfiguration(
            context_length=context,
            target_device=TargetDevice.CPU if backend is ComputeBackend.CPU else TargetDevice.AUTO,
            quantization=artifact.quantization,
            device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
            safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
        ),
        OfflineResolver(),
        resolve_base_model=False,
        recommend_runtime=False,
    )
    return estimate.model_copy(update={"configuration_sources": enriched.sources})


def server_command(
    binary: Path,
    model: Path,
    *,
    n_gpu_layers: int = -1,
    device: str | None = "CUDA0",
    threads: int = 4,
    context: int = CONTEXT,
    chat: bool = False,
    port: int = 18083,
) -> list[str]:
    # The defaults are the pilot's own launch, so its recorded flags do not move.
    # Only the placement replay passes anything else, and only these three.
    return [
        str(binary),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context),
        "--parallel",
        "1",
        "--n-gpu-layers",
        str(n_gpu_layers),
        *(["--device", device] if device is not None else []),
        "--threads",
        str(threads),
        "--seed",
        "0",
        "--no-context-shift",
        "--no-cache-prompt",
        # Chat suites: the GGUF's own template, and reasoning off (verified per reply).
        *(["--jinja", "--reasoning", "off"] if chat else []),
        "--verbosity",
        "4",
    ]


def wait_for_server(server: subprocess.Popen[bytes], port: int, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited before readiness: {server.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except URLError:
            time.sleep(0.25)
    raise TimeoutError(f"Server did not become ready within {timeout:g} seconds")


def check_server_port(port: int) -> None:
    # TIME_WAIT from our preceding run is reusable; an active listener is not.
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))


def docker_command(
    image: str,
    name: str,
    output: Path,
    artifact: Path,
    dataset: Path,
    *,
    profile: str = "smoke",
    host_network: bool = False,
) -> list[str]:
    if profile not in PROFILES:
        raise ValueError("Unknown fixed evaluation profile")
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--network",
        "host" if host_network else "bridge",
        "--read-only",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,size=128m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--mount",
        f"type=bind,source={output},target=/output",
        "--mount",
        f"type=bind,source={artifact},target=/artifact/model.gguf,readonly",
        "--mount",
        f"type=bind,source={dataset},target={_dataset_target(profile)},readonly",
        image,
        "--base-url",
        "http://127.0.0.1:18083" if host_network else "http://host.docker.internal:18083",
        # Preserve the original smoke's recorded command exactly.
        *(["--profile", profile] if profile != "smoke" else []),
    ]


def _suite(profile: str) -> str:
    return str(PROFILES[profile].get("suite", "hellaswag"))


def _dataset_target(profile: str) -> str:
    return "/dataset/ifeval.jsonl" if _suite(profile) == "ifeval" else "/dataset/validation.parquet"


# Owned-process bounds. Full IFEval is 541 generations of up to 1280 tokens each.
TIMEOUTS = {"smoke": 300, "hellaswag100": 600, "ifeval-smoke": 900, "ifeval": 6 * 3600}
CPU_TIMEOUTS = {"smoke": 1800, "hellaswag100": 6 * 3600}


def run(args: argparse.Namespace) -> None:
    if getattr(args, "external_server", None) is not None:
        from pilot.quality_eval.external_server import run_external

        run_external(args)
        return
    profile = PROFILES[args.profile]
    suite = _suite(args.profile)
    context = IFEVAL_CONTEXT if suite == "ifeval" else CONTEXT
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    server = None
    container_name = "jaull-quality-smoke-" + uuid.uuid4().hex
    container_attempted = False
    service = ArtifactService(OfflineResolver(), ArtifactStorage())
    artifact = None
    artifact_verified = False
    observable = getattr(args, "backend", None) is not None
    runtime_sha = None if observable else SERVER_SHA256
    try:
        if observable and suite != "hellaswag":
            raise ValueError(
                "Observable CPU/CUDA/Vulkan execution currently supports HellaSwag only; "
                "IFEval retains its audited CUDA protocol."
            )
        artifact = ModelArtifact.model_validate_json(args.artifact_json.read_text())
        validate_artifact_identity(artifact.model_dump(mode="json"))
        artifact = service.verify(artifact, full=True)
        artifact_verified = True
        write_json(output / "verified-artifact.json", artifact.model_dump(mode="json"))
        image = prepare_infrastructure(
            args.dataset_file,
            args.llama_server,
            args.image,
            suite=suite,
            **({"observable_runtime": True} if observable else {}),
        )
        host_network = docker_host_network()
        hardware = detect_hardware()
        selection = None
        execution = None
        device = "CUDA0"
        if observable:
            requested = None if args.backend == "auto" else ComputeBackend(args.backend)
            selection = quality_backend_selection(hardware, requested)
            runtime_sha, capability = inspect_runtime(args.llama_server)
            readiness = evaluate_execution_readiness(
                selection=selection, runtime_capability=capability
            )
            if readiness.status is not ExecutionReadinessStatus.READY:
                raise ValueError(f"Evaluation runtime not ready: {readiness.message}")
            selected = selection.selected_backend
            device = None
            device_name = "cpu"
            if selected is not ComputeBackend.CPU:
                backend_capability = next(
                    item for item in capability.backend_capabilities if item.backend is selected
                )
                matched = match_runtime_device(selection, backend_capability)
                if matched.device is None:
                    raise ValueError("Evaluation device is ambiguous or unavailable; no fallback.")
                device, device_name = matched.device.runtime_id, matched.device.name or "unknown"
            execution = {
                "protocol": QUALITY_RUNTIME_PROTOCOL,
                "backend": selected.value,
                "device": device or "none",
                "device_name": device_name,
                "version_text": capability.version_text,
                "server_sha256": runtime_sha,
                "fingerprint": fingerprint_from_version(capability.version_text),
                "capability_sha256": digest(capability.model_dump(mode="json")),
            }
            write_json(output / "runtime-capability.json", capability.model_dump(mode="json"))
        estimate = artifact_memory_estimate(
            artifact,
            quality_memory_hardware(hardware, selection) if observable else hardware,
            context=context,
            require_chat_template=suite == "ifeval",
            **({"backend": selection.selected_backend} if observable else {}),
        )
        write_json(output / "memory-preflight.json", estimate.model_dump(mode="json"))
        if suite == "ifeval":
            assert artifact.local_path is not None
            write_json(output / "chat-template.json", chat_template_evidence(artifact.local_path))
        if not observable:
            selection = select_runtime_backend(hardware)
            capability = inspect_llama_cpp_runtime(
                backend=HostExecutionBackend(),
                llama_cli_path=args.llama_server,
            )
            readiness = evaluate_execution_readiness(
                selection=selection, runtime_capability=capability
            )
        write_json(output / "hardware.json", hardware.model_dump(mode="json"))
        write_json(output / "readiness.json", readiness.model_dump(mode="json"))
        if not observable and (
            selection.selected_backend is not ComputeBackend.CUDA
            or readiness.status is not ExecutionReadinessStatus.READY
            or len(hardware.gpus) != 1
        ):
            raise ValueError("The fixed CUDA smoke requires confirmed readiness on one GPU")
        if observable:
            validate_quality_memory(estimate, selection.selected_backend)
        elif (
            estimate.hardware_fit is None
            or estimate.hardware_fit.mode is not HardwareFitMode.GPU_RESIDENT
        ):
            raise ValueError("The explicit full-offload smoke requires a full-device memory fit")
        write_json(output / "image.json", {"id": image["Id"], "digests": image["RepoDigests"]})
        assert artifact.local_path is not None
        launch = server_command(
            args.llama_server.resolve(),
            artifact.local_path,
            context=context,
            chat=suite == "ifeval",
            **(
                {
                    "n_gpu_layers": 0 if selection.selected_backend is ComputeBackend.CPU else -1,
                    "device": device,
                }
                if observable
                else {}
            ),
        )
        if execution is not None:
            validate_runtime_execution(
                execution,
                {
                    "server_sha256": runtime_sha,
                    "fingerprint": execution["fingerprint"],
                    "backend_flags": launch[3:],
                },
            )
        evaluate = docker_command(
            image["Id"],
            container_name,
            output,
            artifact.local_path,
            args.dataset_file.resolve(),
            profile=args.profile,
            host_network=host_network,
        )
        write_json(
            output / "commands.json",
            {
                "server": launch,
                "evaluator": evaluate,
                "server_sha256": runtime_sha,
                **({"execution": execution} if execution else {}),
            },
        )
        # Never attach to or stop a pre-existing listener.
        check_server_port(18083)
        with (
            (output / "server.log").open("x") as server_log,
            (output / "evaluator.log").open("x") as eval_log,
        ):
            server = subprocess.Popen(launch, stdout=server_log, stderr=subprocess.STDOUT)
            wait_for_server(server, 18083, **({"timeout": 180} if observable else {}))
            container_attempted = True
            subprocess.run(
                evaluate,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                timeout=(
                    CPU_TIMEOUTS[args.profile]
                    if observable and selection.selected_backend is ComputeBackend.CPU
                    else TIMEOUTS[args.profile]
                ),
                check=True,
            )
            container_attempted = False  # Successful --rm already removed our container.
        if json.loads((output / "smoke-status.json").read_text()) != {
            "status": profile["status"],
            "quality_evidence": False,
        }:
            raise RuntimeError("Evaluator exited without its completed profile status")
    except Exception as exc:
        write_json(output / "runner-error.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if container_attempted:
            # This UUID belongs to this invocation; --rm may already have removed it.
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "--force", container_name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if cleanup.returncode and "No such container" not in cleanup.stderr:
                    write_json(output / "cleanup-error.json", {"error": cleanup.stderr})
            except (OSError, subprocess.TimeoutExpired) as exc:
                # A Docker cleanup error must still stop our host server.
                with contextlib.suppress(OSError):
                    write_json(output / "cleanup-error.json", {"error": str(exc)})
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if artifact_verified:
            try:
                after = service.verify(artifact, full=True)
            except Exception as exc:
                write_json(output / "artifact-after-error.json", {"error": str(exc)})
                raise
            write_json(output / "artifact-after.json", after.model_dump(mode="json"))
        if observable and runtime_sha is not None:
            with args.llama_server.open("rb") as runtime:
                if hashlib.file_digest(runtime, "sha256").hexdigest() != runtime_sha:
                    raise ValueError("Runtime changed during evaluation; record withheld.")
    if (output / "cleanup-error.json").exists():
        raise RuntimeError("Owned container cleanup could not be confirmed; see cleanup-error.json")
    write_json(
        output / "runner-status.json",
        {
            "status": profile["status"],
            "quality_evidence": False,
            "server_exit_code": server.returncode,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-json", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "vulkan"))
    parser.add_argument("--external-server", type=Path)
    parser.add_argument("--server-endpoint")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
