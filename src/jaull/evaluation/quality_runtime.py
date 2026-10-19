"""Observable quality runtime v2 identity, separate from historical CUDA pins."""

from __future__ import annotations

import re
from typing import Any

from jaull.domain.estimation import CompatibilityStatus, HardwareFitMode, MemoryEstimate
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import TargetDevice

QUALITY_RUNTIME_PROTOCOL = "observable-runtime-v2"
QUALITY_RUNTIME_LABEL = "io.jaull.quality.runtime-contract"
QUALITY_RUNTIME_SCHEMA = 3


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
