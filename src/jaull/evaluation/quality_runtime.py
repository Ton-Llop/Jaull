"""Observable quality runtime v2 identity, separate from historical CUDA pins."""

from __future__ import annotations

import json
import re
from typing import Any

from jaull.domain.estimation import CompatibilityStatus, HardwareFitMode, MemoryEstimate
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
)
from jaull.runtime.device_matching import (
    match_runtime_device,
    names_agree,
    runtime_device_name_matches,
)
from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness
from jaull.runtime.llama_cpp_memory_report import observed_llama_cpp_device

QUALITY_RUNTIME_PROTOCOL = "observable-runtime-v2"
QUALITY_RUNTIME_LABEL = "io.jaull.quality.runtime-contract"
QUALITY_RUNTIME_SCHEMA = 3


def validate_runtime_observation(runtime: dict[str, Any], raw: dict[str, str]) -> None:
    """Schema 3 must retain positive execution evidence, not just requested flags."""
    # quality_records calls this validator, so import its digest after module loading.
    from jaull.evaluation.quality_records import digest

    try:
        execution = runtime["execution"]
        capability_data = json.loads(raw["runtime-capability.json"])
        capability = LlamaCppRuntimeCapability.model_validate(capability_data)
        selection = RuntimeBackendSelection.model_validate(
            json.loads(raw["readiness.json"])["selection"]
        )
        commands = json.loads(raw["commands.json"])
        props = json.loads(raw["server-props.json"])
        log = raw["server.log"]
        if (
            digest(capability_data) != execution["capability_sha256"]
            or capability.version_text != execution["version_text"]
            or commands["execution"] != execution
            or commands["server_sha256"] != runtime["server_sha256"]
            or commands["server"][0] != capability.binary_path
            or commands["server"][3:] != runtime["backend_flags"]
            or commands["server"][1:3] != ["--model", props["model_path"]]
            or selection.selected_backend.value != execution["backend"]
            or props["total_slots"] != 1
            or props["default_generation_settings"]["n_ctx"] != 2048
            or props["default_generation_settings"]["params"]["seed"] != 0
            or props["default_generation_settings"]["params"]["post_sampling_probs"] is not False
            or {int(value) for value in re.findall(r"\bn_threads\s*=\s*(\d+)", log)} != {4}
        ):
            raise ValueError("Observable runtime identity/configuration contradicts its evidence")
        readiness = evaluate_execution_readiness(selection=selection, runtime_capability=capability)
        if readiness.status is not ExecutionReadinessStatus.READY:
            raise ValueError("Observable runtime capability does not confirm this execution")
        expected_device = "CPU"
        if execution["backend"] != "cpu":
            backend = readiness.selected_backend_capability
            if not isinstance(backend, LlamaCppBackendCapability):
                raise ValueError("Missing GPU capability")
            matched = match_runtime_device(selection, backend).device
            if (
                matched is None or matched.runtime_id != execution["device"]
                or matched.name is None or not names_agree(matched.name, execution["device_name"])
            ):
                raise ValueError("Observable runtime device does not match its capability")
            expected_device = matched.runtime_id
            if not runtime_device_name_matches(log, expected_device, execution["device_name"]):
                raise ValueError("Execution device name differs or is missing from its own log")
            offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
            if not offload or any(
                int(done) != int(total) or int(total) <= 0 for done, total in offload
            ):
                raise ValueError("Full GPU offload was not confirmed")
        if observed_llama_cpp_device(log) != expected_device:
            raise ValueError(
                "Effective backend/device is missing, ambiguous or contradicts the record"
            )
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Missing observable runtime execution evidence") from exc


def validate_quality_memory(estimate: MemoryEstimate, backend: ComputeBackend) -> None:
    confirmed = {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }
    if backend is ComputeBackend.CPU:
        available = estimate.assessment.available_ram_bytes
        if (
            estimate.assessment.effective_device is TargetDevice.CPU
            and estimate.assessment.status in confirmed
            and estimate.total_bytes is not None
            and available is not None
            and 0 < estimate.total_bytes <= available
            and all(
                component.bytes is not None
                for component in (
                    estimate.weights.component,
                    estimate.kv_cache.component,
                    estimate.runtime_overhead.component,
                )
            )
        ):
            return
        raise ValueError("Evaluation RAM fit unconfirmed; no execution attempted.")
    if (
        estimate.hardware_fit is None
        or estimate.hardware_fit.mode is not HardwareFitMode.GPU_RESIDENT
        or estimate.assessment.status not in confirmed
    ):
        raise ValueError(
            "Evaluation full-device memory fit / VRAM budget unconfirmed; no execution attempted."
        )


def fingerprint_from_version(version: str) -> str:
    matches = re.findall(r"(?:version:\s*|\bb)(\d+)\s*(?:\(|-)([0-9a-f]{7,40})\)?", version)
    if len(set(matches)) != 1:
        raise ValueError("Runtime build/commit could not be confirmed from --version.")
    build, commit = matches[0]
    return f"b{build}-{commit}"


def validate_runtime_execution(execution: dict[str, Any], runtime: dict[str, Any]) -> None:
    if (
        set(execution)
        != {
            "protocol",
            "backend",
            "device",
            "version_text",
            "server_sha256",
            "fingerprint",
            "capability_sha256",
            "device_name",
        }
        or execution["protocol"] != QUALITY_RUNTIME_PROTOCOL
    ):
        raise ValueError("Unknown observable runtime contract")
    if execution["backend"] not in {"cpu", "cuda", "vulkan"}:
        raise ValueError("Unsupported quality execution backend")
    if (
        execution["server_sha256"] != runtime["server_sha256"]
        or execution["fingerprint"] != runtime["fingerprint"]
        or fingerprint_from_version(execution["version_text"]) != runtime["fingerprint"]
        or not isinstance(execution["device_name"], str)
        or not execution["device_name"].strip()
    ):
        raise ValueError("Runtime hash/build/fingerprint identity mismatch")
    for key in ("server_sha256", "capability_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", execution[key]) is None:
            raise ValueError("Runtime identity requires exact SHA256 digests")
    flags = runtime["backend_flags"]
    values: dict[str, str] = {}
    index = 0
    while index < len(flags):
        name = flags[index]
        if name in values:
            raise ValueError("Ambiguous quality server flags")
        if name in {"--no-context-shift", "--no-cache-prompt"}:
            values[name] = "true"
            index += 1
        else:
            if index + 1 >= len(flags):
                raise ValueError("Incomplete quality server flags")
            values[name] = flags[index + 1]
            index += 2
    fixed = {
        "--host": "127.0.0.1",
        "--port": "18083",
        "--ctx-size": "2048",
        "--parallel": "1",
        "--threads": "4",
        "--seed": "0",
        "--no-context-shift": "true",
        "--no-cache-prompt": "true",
        "--verbosity": "4",
    }
    if any(values.get(key) != value for key, value in fixed.items()) or (
        set(values) - set(fixed) - {"--n-gpu-layers", "--device"}
    ):
        raise ValueError("Unsupported effective quality server configuration")
    if execution["backend"] == "cpu":
        if (
            execution["device"] != "none"
            or "--device" in flags
            or values.get("--n-gpu-layers") != "0"
        ):
            raise ValueError("CPU evaluation must use zero GPU layers and no GPU device")
    elif (
        not isinstance(execution["device"], str)
        or re.fullmatch(
            r"CUDA\d+" if execution["backend"] == "cuda" else r"Vulkan\d+",
            execution["device"],
        )
        is None
        or values.get("--device") != execution["device"]
        or values.get("--n-gpu-layers") != "-1"
    ):
        raise ValueError("GPU evaluation requires its confirmed device and full offload")
