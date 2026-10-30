"""IFEval's external Windows Vulkan contract, distinct from audited CUDA."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
)
from jaull.evaluation.quality_runtime import (
    fingerprint_from_version,
    validate_quality_memory,
    validate_runtime_execution,
    validate_runtime_observation,
)
from jaull.runtime.device_matching import match_runtime_device

EXTERNAL_PROTOCOL = "ifeval-external-vulkan-v1"
EXTERNAL_SCHEMA = 4
EXTERNAL_LABEL = "io.jaull.quality.ifeval-runtime-contract"


def validate_endpoint(endpoint: str) -> None:
    url = urlsplit(endpoint)
    if (
        url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
        or url.port is None or url.username or url.password
        or url.path or url.query or url.fragment
    ):
        raise ValueError("External endpoint must be an explicit loopback HTTP origin with port.")


def matched_device(
    capability: LlamaCppRuntimeCapability, selection: RuntimeBackendSelection,
) -> LlamaCppRuntimeDevice:
    if selection.selected_backend is not ComputeBackend.VULKAN:
        raise ValueError("External IFEval requires an explicit Vulkan GPU; no fallback.")
    backends = [item for item in capability.backend_capabilities
                if item.backend is ComputeBackend.VULKAN]
    if len(backends) != 1:
        raise ValueError("External Vulkan capability missing or ambiguous.")
    device = match_runtime_device(selection, backends[0]).device
    if device is None:
        raise ValueError("External Vulkan device missing, software-rendered or ambiguous.")
    return device


def validate_external_execution(execution: dict[str, Any], runtime: dict[str, Any]) -> None:
    external = execution.get("external_server")
    if not isinstance(external, dict) or set(external) != {
        "ownership", "host_os", "endpoint", "manifest_sha256", "model_path", "port",
    }:
        raise ValueError("Missing external server identity.")
    validate_endpoint(external["endpoint"])
    if (
        execution["protocol"] != EXTERNAL_PROTOCOL or execution["backend"] != "vulkan"
        or external["ownership"] != "externally_supplied"
        or external["host_os"] != "windows"
        or type(external["port"]) is not int or not 1 <= external["port"] <= 65535
        or re.fullmatch(r"[0-9a-f]{64}", external["manifest_sha256"]) is None
        or not isinstance(external["model_path"], str) or not external["model_path"]
    ):
        raise ValueError("Unsupported external Windows Vulkan identity.")
    base = {key: value for key, value in execution.items() if key != "external_server"}
    base["protocol"] = "observable-runtime-v2"
    validate_runtime_execution(base, runtime, context=4096, port=external["port"], chat=True)


def validate_external_observation(runtime: dict[str, Any], raw: dict[str, str]) -> None:
    from jaull.evaluation.quality_records import digest

    try:
        execution = runtime["execution"]
        external = execution["external_server"]
        manifest = json.loads(raw["external-server.json"])
        before = LlamaCppRuntimeCapability.model_validate(manifest["capability"])
        after = LlamaCppRuntimeCapability.model_validate(json.loads(raw["runtime-after.json"]))
        selection = RuntimeBackendSelection.model_validate(
            json.loads(raw["readiness.json"])["selection"]
        )
        device = matched_device(before, selection)
        live = matched_device(after, selection)
        budget, total = device.memory_free_bytes, device.memory_total_bytes
        estimate = MemoryEstimate.model_validate(json.loads(raw["memory-preflight.json"]))
        props = json.loads(raw["server-props.json"])
        commands = json.loads(raw["commands.json"])
        recheck = json.loads(raw["runtime-after-sha256.json"])
        artifact = json.loads(raw["verified-artifact.json"])
        validate_quality_memory(estimate, ComputeBackend.VULKAN)
        if (
            digest(manifest) != external["manifest_sha256"]
            or manifest["protocol"] != EXTERNAL_PROTOCOL
            or manifest["server_sha256"] != runtime["server_sha256"]
            or recheck != {"server_sha256": runtime["server_sha256"]}
            or manifest["artifact_sha256"] != artifact["sha256"]
            or manifest["model_path"] != external["model_path"]
            or manifest["command"] != commands["server"]
            or manifest["capability"]["binary_path"] != commands["server"][0]
            or not commands["server"][0].lower().endswith(".exe")
            or props["model_path"] != external["model_path"]
            or device.runtime_id != execution["device"] or live.runtime_id != device.runtime_id
            or type(budget) is not int or type(total) is not int or not 0 < budget <= total
            or live.memory_total_bytes != total
            or estimate.hardware_fit is None
            or estimate.hardware_fit.available_vram_bytes != budget
            or estimate.hardware_fit.gpu_required_bytes is None
            or estimate.hardware_fit.gpu_required_bytes > budget
            or before.version_text != execution["version_text"]
            or after.version_text != before.version_text
            or after.binary_path != before.binary_path
            or fingerprint_from_version(after.version_text or "") != runtime["fingerprint"]
            or not raw["server.log"].startswith(raw["server-before.log"])
        ):
            raise ValueError("External runtime/model/budget evidence contradicts its identity.")
        validate_runtime_observation(runtime, raw, context=4096)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Missing external runtime evidence.") from exc
