"""Offline external-runtime contract tests, never physical B60 validation."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from pilot.quality_eval.external_server import memory_hardware, require_wsl_exe, selection_for
from pilot.quality_eval.records import validate_generation_http
from scripts.quality_eval_smoke import server_command

from jaull.domain.estimation import (
    CompatibilityStatus,
    HardwareFitMode,
    HardwareFitResult,
    HardwareMemoryTopology,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import LlamaCppRuntimeCapability
from jaull.evaluation.quality_comparison import compare_quality_records
from jaull.evaluation.quality_external import (
    EXTERNAL_PROTOCOL,
    matched_device,
    validate_endpoint,
    validate_external_execution,
)
from jaull.evaluation.quality_records import describe_record, digest, validate_record
from tests._execution_fixtures import qwen_ctx4096_estimate
from tests._workflow_fixtures import hardware
from tests.test_quality_cpu_runtime import _record as observable_record
from tests.test_quality_generation_records import SAMPLES, _exchange, record


def external_record():
    value = record()
    observed = observable_record("vulkan")
    raw = copy.deepcopy(observed["provenance"]["evidence"])
    capability = json.loads(raw["runtime-capability.json"])
    capability["binary_path"] = "/mnt/c/llama/llama-server.exe"
    capability["version_text"] = "version: 11889 (d854d98ba)"
    device = capability["backend_capabilities"][0]["devices"][0]
    device.update(memory_total_bytes=24 * 1024**3, memory_free_bytes=23 * 1024**3)
    command = server_command(PurePosixPath(capability["binary_path"]),
                             PurePosixPath("C:/models/model.gguf"),
                             device="Vulkan1", context=4096, chat=True, port=8080)
    manifest = {
        "protocol": EXTERNAL_PROTOCOL, "server_sha256": "a" * 64,
        "capability": capability, "artifact_sha256": value["identity"]["artifact_sha256"],
        "model_path": "C:/models/model.gguf", "command": command,
    }
    runtime = value["identity"]["runtime"]
    execution = copy.deepcopy(observed["identity"]["runtime"]["execution"])
    execution.update(
        protocol=EXTERNAL_PROTOCOL, version_text=capability["version_text"],
        fingerprint="b11889-d854d98ba", capability_sha256=digest(capability),
        external_server={
            "ownership": "externally_supplied", "host_os": "windows",
            "endpoint": "http://127.0.0.1:8081", "port": 8080,
            "model_path": "C:/models/model.gguf", "manifest_sha256": digest(manifest),
        },
    )
    runtime.update(execution=execution, server_sha256="a" * 64,
                   fingerprint=execution["fingerprint"], backend_flags=command[3:])
    props = json.loads(raw["server-props.json"])
    props["model_path"] = manifest["model_path"]
    props["default_generation_settings"]["n_ctx"] = 4096
    estimate = qwen_ctx4096_estimate(with_runtime_recommendation=False)
    fit = HardwareFitResult(mode=HardwareFitMode.GPU_RESIDENT,
                            memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
                            weights_bytes=estimate.weights.component.bytes,
                            kv_cache_bytes=estimate.kv_cache.component.bytes,
                            overhead_bytes=estimate.runtime_overhead.component.bytes,
                            available_vram_bytes=23 * 1024**3,
                            gpu_required_bytes=estimate.total_bytes, reason="synthetic fit")
    estimate = estimate.model_copy(update={"hardware_fit": fit,
        "assessment": estimate.assessment.model_copy(update={
            "status": CompatibilityStatus.COMPATIBLE,
        })})
    raw.update({
        "runtime-capability.json": json.dumps(capability),
        "runtime-after.json": json.dumps(capability),
        "runtime-after-sha256.json": json.dumps({"server_sha256": "a" * 64}),
        "external-server.json": json.dumps(manifest),
        "server-props.json": json.dumps(props),
        "server-before.log": raw["server.log"],
        "commands.json": json.dumps({"server": command, "execution": execution,
                                     "server_sha256": "a" * 64}),
        "verified-artifact.json": json.dumps({"sha256": value["identity"]["artifact_sha256"]}),
        "memory-preflight.json": estimate.model_dump_json(),
    })
    value.update(schema_version=4, identity_sha256=digest(value["identity"]))
    value["provenance"]["evidence"] = raw
    return value


def test_external_record_has_its_own_identity_and_preserves_jaull_warning():
    value = external_record()
    validate_record(value)
    description = describe_record(value)
    assert any("externally supplied" in item.lower() for item in description.limitations)
    assert any("not comparable with published IFEval" in item for item in description.limitations)
    assert compare_quality_records(value, record(), left_source="x", right_source="y")[
        "status"
    ] == "NOT_COMPARABLE"


@pytest.mark.parametrize("filename,mutate", [
    ("server.log", lambda text: text.replace("29/29", "28/29")),
    ("server.log", lambda text: text.replace("Vulkan1", "Vulkan0")),
    ("runtime-after-sha256.json", lambda text: text.replace("a" * 64, "b" * 64)),
    ("runtime-after.json", lambda text: text.replace("11889", "11890")),
    ("external-server.json", lambda text: text.replace("model.gguf", "other.gguf")),
    ("memory-preflight.json", lambda text: text.replace(str(23 * 1024**3), "null")),
])
def test_external_evidence_drift_is_rejected(filename, mutate):
    value = external_record()
    raw = value["provenance"]["evidence"]
    raw[filename] = mutate(raw[filename])
    with pytest.raises(ValueError):
        validate_record(value)


@pytest.mark.parametrize("mode", ["missing", "ambiguous", "software", "budget"])
def test_native_device_and_budget_are_mandatory(mode):
    value = external_record()
    capability = json.loads(value["provenance"]["evidence"]["runtime-capability.json"])
    devices = capability["backend_capabilities"][0]["devices"]
    if mode == "missing":
        devices.clear()
    elif mode == "ambiguous":
        devices.append(devices[0] | {"runtime_id": "Vulkan2"})
    elif mode == "software":
        devices[0]["name"] = "llvmpipe"
    else:
        devices[0]["memory_free_bytes"] = None
    cap = LlamaCppRuntimeCapability.model_validate(capability)
    with pytest.raises(ValueError):
        memory_hardware(hardware(), cap, selection_for("Synthetic GPU"))


def test_native_device_index_is_not_assumed_and_cpu_is_not_a_fallback():
    value = external_record()
    cap = LlamaCppRuntimeCapability.model_validate(json.loads(
        value["provenance"]["evidence"]["runtime-capability.json"],
    ))
    assert matched_device(cap, selection_for("Synthetic GPU")).runtime_id == "Vulkan1"
    with pytest.raises(ValueError, match="no fallback"):
        matched_device(cap, selection_for("Synthetic GPU").model_copy(update={
            "selected_backend": ComputeBackend.CPU,
        }))


@pytest.mark.parametrize("endpoint", [
    "http://remote:8081", "http://127.0.0.1", "http://user@localhost:8081",
    "http://127.0.0.1:8081/path", "http://127.0.0.1:8081?secret=x",
])
def test_only_the_explicit_loopback_tunnel_origin_is_accepted(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_windows_exe_requires_wsl_and_a_pe_binary(tmp_path, monkeypatch):
    import pilot.quality_eval.external_server as external

    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"MZtest")
    monkeypatch.setattr(external.platform, "release", lambda: "Linux-microsoft-WSL2")
    if external.os.name == "posix":
        require_wsl_exe(binary)
        binary.write_bytes(b"ELF")
        with pytest.raises(ValueError, match="PE"):
            require_wsl_exe(binary)
    monkeypatch.setattr(external.platform, "release", lambda: "Linux")
    with pytest.raises(ValueError, match="WSL2"):
        require_wsl_exe(binary)


def test_new_build_http_is_only_accepted_with_explicit_new_contract():
    text = "\n".join(_exchange(index) for index in range(2)).replace(
        "b10357-689e227db", "b11889-d854d98ba",
    )
    with pytest.raises(ValueError, match="runtime"):
        validate_generation_http(text, SAMPLES)
    assert validate_generation_http(text, SAMPLES, expected_fingerprint="b11889-d854d98ba")[0] == (
        "b11889-d854d98ba"
    )


def test_historical_cuda_identity_and_launch_are_byte_stable():
    assert record()["identity_sha256"] == (
        "2c6996771e7ad44d06df6ec9faddf2c2241750006ba994fdca6ec36e4d69a0b8"
    )
    assert server_command(Path("/server"), Path("/model"), chat=True, context=4096) == [
        str(Path("/server")), "--model", str(Path("/model")),
        "--host", "127.0.0.1", "--port", "18083",
        "--ctx-size", "4096", "--parallel", "1", "--n-gpu-layers", "-1", "--device", "CUDA0",
        "--threads", "4", "--seed", "0", "--no-context-shift", "--no-cache-prompt",
        "--jinja", "--reasoning", "off", "--verbosity", "4",
    ]


def test_external_execution_cannot_be_disguised_as_cuda():
    value = external_record()
    runtime = value["identity"]["runtime"]
    execution = copy.deepcopy(runtime["execution"])
    execution["backend"] = "cuda"
    with pytest.raises(ValueError):
        validate_external_execution(execution, runtime)


def test_external_bundle_snapshot_reconstructs_chat_http_not_hellaswag(tmp_path, monkeypatch):
    from pilot.quality_eval import records

    value = external_record()
    identity = value["identity"]
    raw = value["provenance"]["evidence"]
    evidence = {name: json.loads(text) for name, text in raw.items() if name.endswith(".json")}
    template = "{{ messages }}"
    template_sha = hashlib.sha256(template.encode()).hexdigest()
    dataset = b"synthetic dataset; not a physical evaluation"
    artifact = {"format": "gguf", "repo_id": "org/model", "filename": "model.gguf",
                "quantization": "Q4_K_M", "sha256": identity["artifact_sha256"],
                "local_path": "/mnt/c/models/model.gguf", "revision": "b" * 40, "size_bytes": 128}
    evidence.update({
        "verified-artifact.json": artifact,
        "lm-eval-results.json": value["result"],
        "evaluator-config.json": identity["evaluator"] | {"chat_template_sha256": template_sha},
        "dataset.json": identity["dataset"] | {"sha256": hashlib.sha256(dataset).hexdigest()},
        "hardware.json": value["provenance"]["hardware"],
        "image.json": {"id": identity["runtime"]["image_id"]},
    })
    evidence["server-props.json"]["chat_template"] = template
    raw.update({name: json.dumps(data) for name, data in evidence.items()})
    raw["http.jsonl"] = "\n".join(_exchange(index) for index in range(2)).replace(
        "b10357-689e227db", "b11889-d854d98ba",
    )
    extra = {
        "chat-template.json": {"source": "gguf", "embedded_sha256": template_sha,
                               "embedded_stripped_sha256": template_sha},
        "generation-summary.json": value["outcome"],
    }
    for name, data in extra.items():
        raw[name] = json.dumps(data)
    for name, text in raw.items():
        # Bytes, not write_text: on Windows write_text turns "\n" into "\r\n", and the
        # evidence reader keeps server-before.log byte-exact.
        (tmp_path / name).write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(records.ifeval, "verified_dataset_bytes", lambda path: dataset)
    profile = {"suite": "ifeval", "task": "jaull-ifeval-chat-v1",
               "sample_ids": [0, 1], "classification": "full"}
    monkeypatch.setattr(records, "profile_for_task", lambda task: profile)
    snapshot = records._snapshot_generation(tmp_path, evidence, raw, profile)
    assert snapshot["schema_version"] == 4
    assert snapshot["identity"]["runtime"]["execution"]["protocol"] == EXTERNAL_PROTOCOL
    assert snapshot["result"] == value["result"]


@pytest.mark.parametrize("backend", [None, ComputeBackend.CPU, ComputeBackend.CUDA])
def test_external_request_is_explicit_vulkan_only(tmp_path, backend):
    from jaull.runtime.quality_eval_runner import (
        QualityEvaluationError,
        QualityProfile,
        QualityRunRequest,
        prepare_quality_setup,
    )

    request = QualityRunRequest(
        artifact_json=None, dataset_file=tmp_path / "data", llama_server=tmp_path / "server.exe",
        image="img", output=tmp_path / "out", pilot_root=Path(__file__).resolve().parents[1],
        profile=QualityProfile.IFEVAL, backend=backend,
        external_server=tmp_path / "manifest.json", server_endpoint="http://127.0.0.1:8081",
    )
    with pytest.raises(QualityEvaluationError):
        prepare_quality_setup(request)


@pytest.mark.skipif(os.name != "posix", reason="The producer is WSL-only")
def test_external_producer_respects_exclusive_output_ownership(tmp_path, monkeypatch):
    import scripts.quality_eval_smoke as smoke
    from pilot.quality_eval import external_server as producer
    from pilot.quality_eval.evaluate import PROFILES

    from jaull.domain.estimation import MemoryEstimate

    value = external_record()
    raw = value["provenance"]["evidence"]
    manifest = json.loads(raw["external-server.json"])
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"MZsynthetic")
    manifest["capability"]["binary_path"] = str(binary)
    manifest["command"][0] = str(binary)
    manifest.update(device_name="Synthetic GPU", port=8080, server_log=str(tmp_path / "server.log"))
    Path(manifest["server_log"]).write_bytes(raw["server.log"].encode("utf-8"))
    manifest_file = tmp_path / "external.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_file = tmp_path / "artifact.json"
    artifact_file.write_text(json.dumps({
        "format": "gguf", "repo_id": "org/model", "filename": "model.gguf",
        "revision": "b" * 40, "quantization": "Q4_K_M", "sha256": "a" * 64,
        "local_path": str(tmp_path / "model.gguf"), "size_bytes": 128,
    }), encoding="utf-8")
    capability = LlamaCppRuntimeCapability.model_validate(manifest["capability"])
    monkeypatch.setattr(producer, "require_wsl_exe", lambda server: None)
    monkeypatch.setattr(producer, "ArtifactService", lambda *args: SimpleNamespace(
        verify=lambda artifact, **kwargs: artifact,
    ))
    monkeypatch.setattr(producer, "prepare_infrastructure", lambda *args, **kwargs: {
        "Id": "sha256:" + "3" * 64, "RepoDigests": [],
    })
    monkeypatch.setattr(producer, "inspect_runtime", lambda path: ("a" * 64, capability))
    monkeypatch.setattr(producer, "detect_hardware", hardware)
    monkeypatch.setattr(producer.subprocess, "check_output", lambda *args, **kwargs:
                        manifest["model_path"] + "\n")
    monkeypatch.setattr(producer, "urlopen", lambda *args, **kwargs:
                        io.BytesIO(raw["server-props.json"].encode()))
    estimate = MemoryEstimate.model_validate_json(raw["memory-preflight.json"])
    monkeypatch.setattr(smoke, "artifact_memory_estimate", lambda *args, **kwargs: estimate)
    monkeypatch.setattr(smoke, "chat_template_evidence", lambda path: {"source": "gguf"})
    output = tmp_path / "bundle"

    def evaluate(command, **kwargs):
        assert command[0] == "docker"
        assert command[command.index("--base-url") + 1] == "http://127.0.0.1:8081"
        assert not (output / "runtime-after.json").exists()
        assert not (output / "runtime-after-sha256.json").exists()
        producer.write_json(output / "server-props.json", json.loads(raw["server-props.json"]))
        producer.write_json(output / "smoke-status.json", {
            "status": PROFILES["ifeval"]["status"], "quality_evidence": False,
        })

    monkeypatch.setattr(producer.subprocess, "run", evaluate)
    producer.run_external(SimpleNamespace(
        profile="ifeval", backend="vulkan", server_endpoint="http://127.0.0.1:8081",
        llama_server=binary, external_server=manifest_file, output=output,
        artifact_json=artifact_file, dataset_file=tmp_path / "dataset.jsonl", image="synthetic",
    ))
    assert not list(output.glob("*-error.json"))
    assert (output / "artifact-after.json").is_file()
    assert (output / "runtime-after.json").is_file()
    assert json.loads((output / "runner-status.json").read_text())["server_ownership"] == (
        "externally_supplied"
    )
