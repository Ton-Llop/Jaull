"""Prepare evidence, then evaluate an operator-owned native Windows server from WSL2.

Never starts/stops the supplied server or tunnel. Preparation must happen before
the model is loaded, so --list-devices observes a usable pre-load memory budget.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import uuid
from pathlib import Path
from urllib.request import urlopen

from pilot.quality_eval.evaluate import PROFILES, validate_artifact_identity, write_json
from pilot.quality_eval.setup import inspect_runtime, prepare_infrastructure

from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    HardwareProfile,
)
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    SelectedAcceleratorRef,
)
from jaull.evaluation.quality_external import (
    EXTERNAL_PROTOCOL,
    matched_device,
    validate_endpoint,
    validate_external_observation,
)
from jaull.evaluation.quality_records import digest
from jaull.evaluation.quality_runtime import fingerprint_from_version, validate_quality_memory
from jaull.hardware.detector import detect_hardware
from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness


def require_wsl_exe(server: Path) -> None:
    if os.name != "posix" or "microsoft" not in platform.release().casefold():
        raise ValueError("External Windows evaluation requires Jaull on WSL2, not native Windows.")
    with server.open("rb") as handle:
        magic = handle.read(2)
    if server.suffix.casefold() != ".exe" or magic != b"MZ":
        raise ValueError("External runtime must be a native Windows PE .exe.")


def selection_for(name: str) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=ComputeBackend.VULKAN,
        selected_accelerator=SelectedAcceleratorRef(
            name=name, vendor=AcceleratorVendor.UNKNOWN, type=AcceleratorType.DEDICATED,
        ),
        reason=RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
    )


def memory_hardware(
    hardware: HardwareProfile, capability: LlamaCppRuntimeCapability,
    selection: RuntimeBackendSelection,
) -> HardwareProfile:
    device = matched_device(capability, selection)
    total, budget = device.memory_total_bytes, device.memory_free_bytes
    if type(total) is not int or type(budget) is not int or not 0 < budget <= total:
        raise ValueError("Native Vulkan memory budget unknown; no evaluation attempted.")
    return hardware.model_copy(update={
        "gpus": [], "accelerators": [AcceleratorProfile(
            name=device.name or "", type=AcceleratorType.DEDICATED,
            dedicated_memory_bytes=total, available_memory_bytes=budget,
            detection_sources=["external-windows-llama-server-list-devices-before-load"],
            backends=[ComputeBackendInfo(
                backend=ComputeBackend.VULKAN, availability=BackendAvailability.AVAILABLE,
                source="native-windows-llama-server-list-devices",
            )],
        )],
    })


def prepare(args: argparse.Namespace) -> None:
    from scripts.quality_eval_smoke import OfflineResolver, server_command

    binary = args.llama_server.expanduser().resolve()
    require_wsl_exe(binary)
    if args.output.exists() or args.server_log.exists():
        raise ValueError("Use fresh manifest and server log paths; stop the old server first.")
    artifact = ModelArtifact.model_validate_json(args.artifact_json.read_text(encoding="utf-8"))
    validate_artifact_identity(artifact.model_dump(mode="json"))
    artifact = ArtifactService(OfflineResolver(), ArtifactStorage()).verify(artifact, full=True)
    sha, capability = inspect_runtime(binary)
    selected = selection_for(args.device_name)
    device = matched_device(capability, selected)
    memory_hardware(detect_hardware(), capability, selected)
    readiness = evaluate_execution_readiness(selection=selected, runtime_capability=capability)
    if readiness.status is not ExecutionReadinessStatus.READY:
        raise ValueError(readiness.message)
    model_path = subprocess.check_output(
        ["wslpath", "-w", str(artifact.local_path)], text=True, timeout=15,
    ).strip()
    command = server_command(binary, Path(model_path), device=device.runtime_id,
                             context=4096, chat=True, port=args.port)
    manifest = {
        "protocol": EXTERNAL_PROTOCOL, "server_sha256": sha,
        "artifact_sha256": artifact.sha256, "model_path": model_path,
        "device_name": args.device_name, "server_log": str(args.server_log.resolve()),
        "capability": capability.model_dump(mode="json"), "command": command, "port": args.port,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Prepared only; server is NOT started. In a separate WSL terminal run:")
    print(shlex.join(command) + " > " + shlex.quote(str(args.server_log.resolve())) + " 2>&1")


def run_external(args: argparse.Namespace) -> None:
    from scripts.quality_eval_smoke import (
        TIMEOUTS,
        OfflineResolver,
        artifact_memory_estimate,
        chat_template_evidence,
        docker_command,
    )

    if args.profile != "ifeval" or args.backend != "vulkan":
        raise ValueError("External server requires full IFEval and explicit Vulkan; no fallback.")
    validate_endpoint(args.server_endpoint)
    binary = args.llama_server.resolve()
    require_wsl_exe(binary)
    manifest = json.loads(args.external_server.read_text(encoding="utf-8"))
    capability = LlamaCppRuntimeCapability.model_validate(manifest["capability"])
    selection = selection_for(manifest["device_name"])
    device = matched_device(capability, selection)
    if capability.binary_path != str(binary):
        raise ValueError("Supplied server path differs from the prepared Windows executable.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = "jaull-external-ifeval-" + uuid.uuid4().hex
    service = ArtifactService(OfflineResolver(), ArtifactStorage())
    artifact = None
    attempted = False
    try:
        artifact = ModelArtifact.model_validate_json(args.artifact_json.read_text(encoding="utf-8"))
        validate_artifact_identity(artifact.model_dump(mode="json"))
        artifact = service.verify(artifact, full=True)
        native_path = subprocess.check_output(
            ["wslpath", "-w", str(artifact.local_path)], text=True, timeout=15,
        ).strip()
        if native_path != manifest["model_path"] or artifact.sha256 != manifest["artifact_sha256"]:
            raise ValueError("External Windows model path/hash differs from the verified GGUF.")
        write_json(output / "verified-artifact.json", artifact.model_dump(mode="json"))
        image = prepare_infrastructure(args.dataset_file, binary, args.image, suite="ifeval",
                                       observable_runtime=True, external_runtime=True)
        sha, live = inspect_runtime(binary)
        if sha != manifest["server_sha256"] or live.version_text != capability.version_text:
            raise ValueError("Prepared runtime binary/build changed.")
        hardware = memory_hardware(detect_hardware(), capability, selection)
        readiness = evaluate_execution_readiness(selection=selection, runtime_capability=live)
        if readiness.status is not ExecutionReadinessStatus.READY:
            raise ValueError(readiness.message)
        estimate = artifact_memory_estimate(
            artifact, hardware, context=4096, require_chat_template=True,
            backend=ComputeBackend.VULKAN,
        )
        validate_quality_memory(estimate, ComputeBackend.VULKAN)
        execution = {
            "protocol": EXTERNAL_PROTOCOL, "backend": "vulkan", "device": device.runtime_id,
            "device_name": device.name, "server_sha256": sha, "version_text": live.version_text,
            "fingerprint": fingerprint_from_version(live.version_text or ""),
            "capability_sha256": digest(live.model_dump(mode="json")),
            "external_server": {
                "ownership": "externally_supplied", "host_os": "windows",
                "endpoint": args.server_endpoint, "manifest_sha256": digest(manifest),
                "model_path": manifest["model_path"], "port": manifest["port"],
            },
        }
        runtime = {"execution": execution, "server_sha256": sha,
                   "fingerprint": execution["fingerprint"],
                   "backend_flags": manifest["command"][3:]}
        evaluate = docker_command(image["Id"], name, output, artifact.local_path,
                                  args.dataset_file.resolve(), profile="ifeval", host_network=True)
        evaluate[evaluate.index("--base-url") + 1] = args.server_endpoint
        data = {
            "external-server.json": manifest,
            "runtime-capability.json": live.model_dump(mode="json"),
            "memory-preflight.json": estimate.model_dump(mode="json"),
            "hardware.json": hardware.model_dump(mode="json"),
            "readiness.json": readiness.model_dump(mode="json"),
            "chat-template.json": chat_template_evidence(artifact.local_path),
            "image.json": {"id": image["Id"], "digests": image["RepoDigests"]},
            "commands.json": {"server": manifest["command"], "evaluator": evaluate,
                              "server_sha256": sha, "execution": execution},
        }
        for filename, value in data.items():
            write_json(output / filename, value)
        log = Path(manifest["server_log"])
        initial_log = log.read_bytes()
        for filename in ("server-before.log", "server.log"):
            (output / filename).write_bytes(initial_log)
        with urlopen(args.server_endpoint + "/props", timeout=15) as response:
            write_json(output / "server-props-before.json", json.load(response))

        def verify_observation(*, before: bool = False) -> None:
            raw = {path.name: path.read_bytes().decode("utf-8", "backslashreplace")
                   for path in output.iterdir()
                   if path.is_file() and path.suffix in {".json", ".log"}}
            if before:
                # Preflight uses in-memory aliases, not files owned by the evaluator
                # or post-run probes. All output files use exclusive creation.
                raw["server-props.json"] = raw["server-props-before.json"]
                raw["runtime-after.json"] = raw["runtime-capability.json"]
                raw["runtime-after-sha256.json"] = json.dumps({"server_sha256": sha})
            validate_external_observation(runtime, raw)

        verify_observation(before=True)  # Confirm full offload before any scored request.
        with (output / "evaluator.log").open("x", encoding="utf-8") as handle:
            attempted = True
            subprocess.run(evaluate, stdout=handle, stderr=subprocess.STDOUT,
                           timeout=TIMEOUTS["ifeval"], check=True)
            attempted = False
        after_sha, after = inspect_runtime(binary)
        if after_sha != sha or after.version_text != live.version_text:
            raise ValueError("Runtime changed during evaluation; record withheld.")
        write_json(output / "runtime-after.json", after.model_dump(mode="json"))
        write_json(output / "runtime-after-sha256.json", {"server_sha256": after_sha})
        (output / "server.log").write_bytes(log.read_bytes())
        verify_observation()
        if json.loads((output / "smoke-status.json").read_text()) != {
            "status": PROFILES["ifeval"]["status"], "quality_evidence": False,
        }:
            raise ValueError("Evaluator did not complete the full IFEval profile.")
        write_json(output / "runner-status.json", {
            "status": PROFILES["ifeval"]["status"], "quality_evidence": False,
            "server_ownership": "externally_supplied",
        })
    except BaseException as exc:
        write_json(output / "runner-error.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    finally:
        if attempted:
            cleanup = subprocess.run(["docker", "rm", "--force", name], capture_output=True,
                                     text=True, timeout=15, check=False)
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                write_json(output / "cleanup-error.json", {"error": cleanup.stderr})
        if artifact is not None:
            try:
                after_artifact = service.verify(artifact, full=True)
                write_json(output / "artifact-after.json", after_artifact.model_dump(mode="json"))
            except BaseException as exc:
                write_json(output / "artifact-after-error.json", {"error": str(exc)})
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-json", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
