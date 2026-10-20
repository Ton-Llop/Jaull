"""Observable quality v2 fixtures are synthetic, not physical GPU/CPU validation."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from pilot.quality_eval import evaluate, records, setup
from scripts.quality_eval_smoke import server_command

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
)
from jaull.domain.inference import TargetDevice
from jaull.evaluation.quality_comparison import compare_quality_records
from jaull.evaluation.quality_records import digest, validate_record
from jaull.evaluation.quality_runtime import (
    QUALITY_RUNTIME_PROTOCOL,
    fingerprint_from_version,
    validate_quality_memory,
    validate_runtime_execution,
)
from jaull.runtime.quality_execution import (
    quality_backend_selection,
    quality_memory_hardware,
)
from tests._workflow_fixtures import hardware
from tests.test_quality_candidates import SelectionAdvisor, _pool, prepare_candidates
from tests.test_quality_eval_records import synthetic_limited_record


def _record(backend="cpu", sha="a" * 64):
    record = synthetic_limited_record()
    runtime = record["identity"]["runtime"]
    device = "none" if backend == "cpu" else "CUDA1" if backend == "cuda" else "Vulkan1"
    runtime.update(
        {
            "server_sha256": sha,
            "fingerprint": "b12345-abcdef123",
            "backend_flags": server_command(
                Path("/runtime/server"),
                Path("/models/model.gguf"),
                device=None if device == "none" else device,
                n_gpu_layers=0 if backend == "cpu" else -1,
            )[3:],
            "execution": {
                "protocol": QUALITY_RUNTIME_PROTOCOL,
                "backend": backend,
                "device": device,
                "device_name": "cpu" if backend == "cpu" else "Synthetic GPU",
                "version_text": "version: 12345 (abcdef123)\nbuilt by synthetic compiler",
                "fingerprint": "b12345-abcdef123",
                "server_sha256": sha,
                "capability_sha256": "b" * 64,
            },
        }
    )
    record["schema_version"] = 3
    from jaull.domain.runtime import LlamaCppBinaryStatus, LlamaCppRuntimeCapability
    from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness
    from tests.test_runtime_device_matching import _capability, _device, _selection

    selection = _selection("Synthetic GPU").model_copy(
        update={"selected_backend": ComputeBackend(backend)}
    )
    gpu = _capability(_device(1, "Synthetic GPU")).model_copy(update={
        "backend": ComputeBackend(backend),
        "devices": [_device(1, "Synthetic GPU").model_copy(update={
            "backend": ComputeBackend(backend), "runtime_id": device,
        })] if backend != "cpu" else [],
    })
    capability = LlamaCppRuntimeCapability(
        binary_path="/runtime/server", binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text=runtime["execution"]["version_text"], backend_capabilities=[gpu],
    )
    raw_capability = capability.model_dump(mode="json")
    runtime["execution"]["capability_sha256"] = digest(raw_capability)
    label = "CPU" if backend == "cpu" else device
    record["provenance"]["evidence"] = {
        "runtime-capability.json": json.dumps(raw_capability),
        "readiness.json": evaluate_execution_readiness(
            selection=selection, runtime_capability=capability,
        ).model_dump_json(),
        "commands.json": json.dumps({
            "server": ["/runtime/server", "--model", "/models/model.gguf",
                       *runtime["backend_flags"]],
            "server_sha256": sha, "execution": runtime["execution"],
        }),
        "server-props.json": json.dumps({"model_path": "/models/model.gguf",
            "total_slots": 1, "default_generation_settings": {
                "n_ctx": 2048, "params": {"seed": 0, "post_sampling_probs": False},
            }}),
        "server.log": (
            ("" if backend == "cpu" else (
                "Device 1: Synthetic GPU, compute capability 7.5\n" if backend == "cuda" else
                "ggml_vulkan: 1 = Synthetic GPU (test driver) | uma: 0\n"
            )) +
            "system_info: n_threads = 4 (n_threads_batch = 4) / 8\n"
            f"load_tensors: offloaded {0 if backend == 'cpu' else 29}/29 layers to GPU\n"
            f"load_tensors: {label} model buffer size = 100.00 MiB\n"
            f"llama_context: {label} compute buffer size = 10.00 MiB\n"
        ),
    }
    prefix = {"url": "http://127.0.0.1:18083/tokenize",
              "request": {"content": "Synthetic prompt", "add_special": True},
              "response": {"tokens": [1]}}
    whole = deepcopy(prefix)
    whole["request"]["content"] += " choice"
    whole["response"]["tokens"] = [1, 2]
    score = {"url": "http://127.0.0.1:18083/v1/completions",
             "request": {"prompt": [1], "temperature": 0, "max_tokens": 1,
                         "logprobs": 2, "logit_bias": [[2, 100]], "id_slot": 0},
             "response": {"system_fingerprint": runtime["fingerprint"],
                          "usage": {"completion_tokens": 1, "prompt_tokens": 1,
                                    "prompt_tokens_details": {"cached_tokens": 0}},
                          "choices": [{"logprobs": {"content": [{"id": 2, "logprob": -1.0,
                                      "top_logprobs": [{"id": 2, "logprob": -1.0}]}]}}]}}
    record["provenance"]["evidence"]["http.jsonl"] = "".join(json.dumps(item) + "\n" for item in (
        prefix, whole, *[score for _ in record["identity"]["samples"]],
    ))
    record["identity_sha256"] = digest(record["identity"])
    return record


@pytest.mark.parametrize("damage", ["cuda_log", "another_device", "initialization_only",
                                   "mixed_devices", "missing", "threads", "capability",
                                   "renumbered_log", "command_runtime", "command_device",
                                   "empty_command", "props_context", "props_slots"])
def test_schema3_rejects_unproven_or_contradictory_execution(damage):
    record = _record("vulkan")
    raw = record["provenance"]["evidence"]
    if damage == "missing":
        del raw["server.log"]
    elif damage == "capability":
        capability = json.loads(raw["runtime-capability.json"])
        capability["backend_capabilities"] = []
        raw["runtime-capability.json"] = json.dumps(capability)
        record["identity"]["runtime"]["execution"]["capability_sha256"] = digest(capability)
        record["identity_sha256"] = digest(record["identity"])
    elif damage in {"command_runtime", "command_device", "empty_command"}:
        command = json.loads(raw["commands.json"])
        if damage == "empty_command":
            command["server"] = []
        elif damage == "command_runtime":
            command["server"][0] = "/different/llama-server"
        else:
            index = command["server"].index("--device")
            command["server"][index + 1] = "Vulkan0"
        raw["commands.json"] = json.dumps(command)
    elif damage.startswith("props_"):
        props = json.loads(raw["server-props.json"])
        if damage == "props_context":
            props["default_generation_settings"]["n_ctx"] = 4096
        else:
            props["total_slots"] = 2
        raw["server-props.json"] = json.dumps(props)
    else:
        raw["server.log"] = {
            "cuda_log": raw["server.log"].replace("Vulkan1", "CUDA0"),
            "another_device": raw["server.log"].replace("Vulkan1", "Vulkan0"),
            "initialization_only": "ggml_vulkan: Found 1 device\noffloaded 29/29 layers to GPU",
            "mixed_devices": raw["server.log"] + (
                "load_tensors: CUDA0 model buffer size = 10.00 MiB\n"
            ),
            "threads": raw["server.log"].replace("n_threads = 4", "n_threads = 8"),
            "renumbered_log": raw["server.log"].replace("1 = Synthetic GPU", "1 = Other GPU") + (
                "ggml_vulkan: 2 = Synthetic GPU (test driver) | uma: 0\n"
            ),
        }[damage]
    with pytest.raises(ValueError):
        validate_record(record)


def test_schema3_multiple_available_backends_do_not_prove_the_effective_one():
    record = _record("vulkan")
    raw = record["provenance"]["evidence"]
    cap = json.loads(raw["runtime-capability.json"])
    cuda = deepcopy(cap["backend_capabilities"][0])
    cuda["backend"] = "cuda"
    cuda["devices"][0].update(backend="cuda", runtime_id="CUDA0", name="Other GPU")
    cap["backend_capabilities"].append(cuda)
    raw["runtime-capability.json"] = json.dumps(cap)
    execution = record["identity"]["runtime"]["execution"]
    execution["capability_sha256"] = digest(cap)
    command = json.loads(raw["commands.json"])
    command["execution"] = execution
    raw["commands.json"] = json.dumps(command)
    record["identity_sha256"] = digest(record["identity"])
    raw["server.log"] += "ggml_cuda_init: Device 0: Other GPU, compute capability 7.5\n"
    validate_record(record)  # Both initialized; only Vulkan owns the model and compute.
    raw["server.log"] = raw["server.log"].replace("Vulkan1", "CUDA0")
    with pytest.raises(ValueError, match="Effective backend/device"):
        validate_record(record)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "vulkan"])
def test_schema3_snapshot_and_reload_enforce_observations(tmp_path, monkeypatch, backend):
    import hashlib

    record = _record(backend)
    raw = record["provenance"]["evidence"]
    profile = evaluate.PROFILES["hellaswag100"]
    data = b"Synthetic dataset, no benchmark was run"
    monkeypatch.setattr(records, "verified_dataset_bytes", lambda _: data)
    artifact = {
        "repo_id": "synthetic/model", "revision": "a" * 40,
        "filename": "model.gguf", "format": "gguf", "quantization": "Q4_K_M",
        "size_bytes": 128, "sha256": record["identity"]["artifact_sha256"],
        "local_path": "/models/model.gguf",
    }
    files = {name: {} for name in records.JSON_FILES} | {
        "verified-artifact.json": artifact, "artifact-after.json": artifact,
        "container-artifact.json": {"sha256": artifact["sha256"], "mount": "read-only"},
        "image.json": {"id": record["identity"]["runtime"]["image_id"]},
        "hardware.json": {"synthetic": True},
        "evaluator-config.json": record["identity"]["evaluator"],
        "dataset.json": record["identity"]["dataset"] | {
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "lm-eval-results.json": record["result"],
        "smoke-status.json": {"status": profile["status"], "quality_evidence": False},
        "runner-status.json": {"status": profile["status"], "quality_evidence": False,
                               "server_exit_code": 0},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    for name, text in raw.items():
        (tmp_path / name).write_text(text)
    (tmp_path / "evaluator.log").write_text("Synthetic, no evaluation happened")
    snapshot = records.snapshot_bundle(tmp_path)
    path = tmp_path / "record.json"
    records.save_record(path, snapshot)
    assert records.load_record(path) == snapshot
    before = path.read_bytes()
    (tmp_path / "server.log").write_text("ggml_vulkan: Found 1 device\n")
    with pytest.raises(ValueError):
        records.snapshot_bundle(tmp_path)
    assert path.read_bytes() == before
    snapshot["provenance"]["evidence"]["server.log"] = "ggml_cuda_init: found 1"
    damaged = tmp_path / "damaged.json"
    damaged.write_text(json.dumps({"record": snapshot, "record_sha256": digest(snapshot)}))
    with pytest.raises(ValueError):
        records.load_record(damaged)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "vulkan"])
def test_runtime_protocol_preserves_backend_device_build_and_distinct_binary_hashes(backend):
    record = _record(backend)
    validate_record(record)
    records.validate_record(record)
    assert record["identity"]["runtime"]["execution"]["backend"] == backend
    other = _record(backend, sha="c" * 64)
    validate_record(other)
    assert record["identity_sha256"] != other["identity_sha256"]
    result = compare_quality_records(record, other, left_source="left", right_source="right")
    assert result["status"] == "NOT_COMPARABLE"


@pytest.mark.parametrize("change", ["device", "layers", "context", "seed", "fingerprint", "sha"])
def test_runtime_contract_rejects_ambiguous_or_changed_effective_configuration(change):
    runtime = _record()["identity"]["runtime"]
    if change in {"layers", "context", "seed"}:
        flag, value = {
            "layers": ("--n-gpu-layers", "-1"),
            "context": ("--ctx-size", "4096"),
            "seed": ("--seed", "12"),
        }[change]
        runtime["backend_flags"][runtime["backend_flags"].index(flag) + 1] = value
    elif change == "device":
        runtime["backend_flags"].extend(["--device", "Vulkan0"])
    elif change == "sha":
        runtime["server_sha256"] = "c" * 64
    else:
        runtime["fingerprint"] = "b99999-abcdef123"
    with pytest.raises(ValueError):
        validate_runtime_execution(runtime["execution"], runtime)


def test_historical_cuda_records_remain_readable_but_not_equivalent_to_v2():
    legacy = synthetic_limited_record()
    before = deepcopy(legacy)
    validate_record(legacy)
    result = compare_quality_records(legacy, _record(), left_source="legacy", right_source="cpu")
    assert result["status"] == "NOT_COMPARABLE"
    assert not next(c["match"] for c in result["checks"] if c["field"] == "schema_version")
    assert legacy == before
    forged = _record()
    forged["classification"] = "full"
    with pytest.raises(ValueError, match="diagnostic"):
        validate_record(forged)


def test_transient_capability_snapshot_stays_auditable_without_changing_protocol_equivalence():
    first = _record("cuda")
    second = deepcopy(first)
    raw = second["provenance"]["evidence"]
    cap = json.loads(raw["runtime-capability.json"])
    cap["backend_capabilities"][0]["devices"][0]["memory_free_bytes"] = 123
    raw["runtime-capability.json"] = json.dumps(cap)
    execution = second["identity"]["runtime"]["execution"]
    execution["capability_sha256"] = digest(cap)
    commands = json.loads(raw["commands.json"])
    commands["execution"] = execution
    raw["commands.json"] = json.dumps(commands)
    second["identity_sha256"] = digest(second["identity"])
    assert first["identity_sha256"] != second["identity_sha256"]
    result = compare_quality_records(first, second, left_source="first", right_source="second")
    assert result["status"] == "COMPARABLE_LIMITED"
    second["identity"]["runtime"]["execution"]["device_name"] = "Different GPU"
    cap["backend_capabilities"][0]["devices"][0]["name"] = "Different GPU"
    raw["runtime-capability.json"] = json.dumps(cap)
    execution["capability_sha256"] = digest(cap)
    ready = json.loads(raw["readiness.json"])
    ready["selection"]["selected_accelerator"]["name"] = "Different GPU"
    raw["readiness.json"] = json.dumps(ready)
    raw["server.log"] = raw["server.log"].replace("Synthetic GPU", "Different GPU")
    commands["execution"] = execution
    raw["commands.json"] = json.dumps(commands)
    second["identity_sha256"] = digest(second["identity"])
    result = compare_quality_records(first, second, left_source="first", right_source="second")
    assert result["status"] == "NOT_COMPARABLE"


def test_cpu_selection_does_not_require_any_gpu_and_explicit_gpu_never_falls_back():
    hw = hardware(vram_gib=None)
    assert quality_backend_selection(hw).selected_backend is ComputeBackend.CPU
    assert quality_backend_selection(hw, ComputeBackend.CPU).selected_accelerator is None
    with pytest.raises(ValueError, match="no fallback"):
        quality_backend_selection(hw, ComputeBackend.VULKAN)
    cuda = quality_backend_selection(hardware(), ComputeBackend.CUDA)
    assert cuda.selected_backend is ComputeBackend.CUDA


def test_vulkan_selection_keeps_detected_budget_not_capacity():
    hw = hardware(vram_gib=None).model_copy(
        update={
            "accelerators": [
                AcceleratorProfile(
                    name="Synthetic Intel card",
                    vendor=AcceleratorVendor.INTEL,
                    type=AcceleratorType.DEDICATED,
                    dedicated_memory_bytes=24 * 1024**3,
                    available_memory_bytes=20 * 1024**3,
                    backends=[
                        ComputeBackendInfo(
                            backend=ComputeBackend.VULKAN,
                            availability=BackendAvailability.AVAILABLE,
                        )
                    ],
                )
            ]
        }
    )
    selection = quality_backend_selection(hw, ComputeBackend.VULKAN)
    exact = quality_memory_hardware(hw, selection)
    assert exact.accelerators[0].available_memory_bytes == 20 * 1024**3
    assert selection.selected_accelerator.name == "Synthetic Intel card"


def test_cpu_candidates_are_selectable_without_cuda_and_do_not_mutate_search(monkeypatch):
    from jaull.domain.estimation import HardwareFitMode

    class CpuAdvisor(SelectionAdvisor):
        def estimate_model(self, analysis, hw, cfg, **kwargs):
            estimate = super().estimate_model(analysis, hw, cfg, **kwargs)
            if cfg.target_device is TargetDevice.CPU:
                estimate = estimate.model_copy(
                    update={
                        "assessment": estimate.assessment.model_copy(
                            update={
                                "effective_device": TargetDevice.CPU,
                                "available_ram_bytes": hw.memory.available_bytes,
                            },
                        )
                    }
                )
            return estimate

    advisor = CpuAdvisor(1)
    advisor.fit = HardwareFitMode.CPU_RAM
    _pool(monkeypatch, advisor)
    state = advisor.state().model_copy(update={"hardware": hardware(vram_gib=None)})
    before = state.model_dump_json()
    result = prepare_candidates(advisor, state, backend=ComputeBackend.CPU)
    assert len(result.candidates) == 1
    assert "cpu evaluation fit" in result.candidates[0].reason
    assert state.model_dump_json() == before


def test_cpu_fit_uses_ram_and_rejects_insufficient_memory(tmp_path):
    from scripts.quality_eval_smoke import artifact_memory_estimate

    from jaull.domain.artifacts import ModelArtifact
    from tests._gguf_fixtures import build_header

    path = tmp_path / "model.gguf"
    path.write_bytes(
        build_header(
            {
                "general.architecture": "qwen2",
                "qwen2.context_length": 32768,
                "qwen2.embedding_length": 1536,
                "qwen2.block_count": 28,
                "qwen2.attention.head_count": 12,
                "qwen2.attention.head_count_kv": 2,
            }
        )
    )
    artifact = ModelArtifact(
        repo_id="synthetic/model",
        revision="a" * 40,
        filename="model.gguf",
        format="gguf",
        quantization="Q4_K_M",
        local_path=path,
        size_bytes=path.stat().st_size,
    )
    hw = hardware(vram_gib=None)
    estimate = artifact_memory_estimate(artifact, hw, backend=ComputeBackend.CPU)
    assert estimate.inference_configuration.target_device is TargetDevice.CPU
    validate_quality_memory(estimate, ComputeBackend.CPU)
    tiny = hw.model_copy(update={"memory": hw.memory.model_copy(update={"available_bytes": 1})})
    estimate = artifact_memory_estimate(artifact, tiny, backend=ComputeBackend.CPU)
    with pytest.raises(ValueError, match="RAM"):
        validate_quality_memory(estimate, ComputeBackend.CPU)


def test_unknown_build_metadata_is_not_a_generic_http_acceptance_rule():
    assert fingerprint_from_version("version: 10357 (689e227db)") == "b10357-689e227db"
    with pytest.raises(ValueError):
        fingerprint_from_version("some runtime")
    payload = {"prompt": [1], "logit_bias": [[2, 100]]}
    response = {
        "system_fingerprint": "b12345-abcdef123",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "id": 2,
                            "logprob": -1.0,
                            "top_logprobs": [{"id": 2, "logprob": -1.0}],
                        }
                    ]
                }
            }
        ],
    }
    evaluate.validate_scoring_response(payload, response, expected_fingerprint="b12345-abcdef123")
    with pytest.raises(ValueError):
        evaluate.validate_scoring_response(payload, response)
    response["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    with pytest.raises(ValueError):
        evaluate.validate_scoring_response(
            payload, response, expected_fingerprint="b12345-abcdef123"
        )


def test_missing_runtime_blocks_without_requiring_cuda(tmp_path):
    with pytest.raises(ValueError, match="Runtime unavailable"):
        setup.prepare_infrastructure(
            tmp_path / "data", tmp_path / "absent", "image", observable_runtime=True
        )


@pytest.mark.parametrize("backend", ["cpu", "cuda", "vulkan"])
def test_cli_forwards_explicit_evaluation_backend_without_executing(monkeypatch, tmp_path, backend):
    from importlib import import_module

    from typer.testing import CliRunner

    cli = import_module("jaull.cli.app")
    calls = []

    def run(requests, **kwargs):
        calls.extend(requests)
        return 0

    monkeypatch.setattr(cli, "run_quality", run)
    result = CliRunner().invoke(
        cli.app,
        [
            "quality", "run", "--artifact", str(tmp_path / "artifact.json"),
            "--dataset-file", str(tmp_path / "data.parquet"),
            "--llama-server", str(tmp_path / "server"),
            "--image", "synthetic:never-executed", "--output", str(tmp_path / "out"),
            "--backend", backend,
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0].backend is ComputeBackend(backend)
    assert not (tmp_path / "out").exists()


def test_tui_cpu_backend_is_preserved_without_implicitly_executing(tmp_path):
    import asyncio

    from textual.widgets import Select

    from jaull.tui.app import JaullApp
    from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen
    from tests.test_cli_quality import _request
    from tests.test_tui_quality import FakeAdvisor, _fill

    async def exercise():
        advisor = FakeAdvisor()
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(110, 40)) as pilot:
            screen = QualityEvaluationScreen()
            await app.push_screen(screen)
            await pilot.pause()
            _fill(screen, _request(tmp_path))
            screen.query_one("#quality-backend", Select).value = "cpu"
            await pilot.pause()
            assert screen._request(require_artifact=False).backend is ComputeBackend.CPU
            assert not advisor.calls and not advisor.preparations

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [None, "docker", "cancel", "memory"])
@pytest.mark.skipif(os.name != "posix", reason="Host pilot requires Linux/WSL paths and processes")
def test_cpu_producer_uses_zero_offload_and_cleans_up_owned_processes(
    tmp_path,
    monkeypatch,
    failure,
):
    import hashlib
    import json
    import subprocess
    from types import SimpleNamespace

    from scripts import quality_eval_smoke as smoke

    from jaull.artifacts.storage import ArtifactStorage
    from jaull.domain.runtime import (
        LlamaCppBackendCapability,
        LlamaCppBackendCapabilityState,
        LlamaCppBinaryStatus,
        LlamaCppCapabilityReason,
        LlamaCppRuntimeCapability,
    )
    from tests._gguf_fixtures import build_header
    from tests.test_quality_eval_smoke import synthetic_artifact_identity

    model, binary = tmp_path / "model.gguf", tmp_path / "llama-server"
    model.write_bytes(
        build_header(
            {
                "general.architecture": "qwen2",
                "qwen2.context_length": 32768,
                "qwen2.embedding_length": 1536,
                "qwen2.block_count": 28,
                "qwen2.attention.head_count": 12,
                "qwen2.attention.head_count_kv": 2,
            }
        )
    )
    binary.write_bytes(b"fake runtime; never executed")
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    ArtifactStorage().save_sha256(model, sha)
    manifest = tmp_path / "artifact.json"
    manifest.write_text(
        json.dumps(
            synthetic_artifact_identity()
            | {
                "local_path": str(model),
                "sha256": sha,
                "size_bytes": model.stat().st_size,
            }
        )
    )
    cap = LlamaCppRuntimeCapability(
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text="version: 12345 (abcdef123)",
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CPU,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
            )
        ],
    )
    monkeypatch.setattr(
        smoke,
        "inspect_runtime",
        lambda _: (
            hashlib.sha256(binary.read_bytes()).hexdigest(),
            cap,
        ),
    )
    hw = hardware(vram_gib=None)
    if failure == "memory":
        hw = hw.model_copy(update={"memory": hw.memory.model_copy(update={"available_bytes": 1})})
    monkeypatch.setattr(smoke, "detect_hardware", lambda: hw)
    monkeypatch.setattr(
        smoke,
        "prepare_infrastructure",
        lambda *a, **kw: {
            "Id": "sha256:" + "e" * 64,
            "RepoDigests": [],
        },
    )
    monkeypatch.setattr(smoke, "check_server_port", lambda _: None)
    monkeypatch.setattr(smoke, "docker_host_network", lambda: True)
    monkeypatch.setattr(smoke, "wait_for_server", lambda *a, **kw: None)
    launches, cleanups = [], []

    class Server:
        returncode = None

        def __init__(self, command, **kwargs):
            launches.append(command)

        def poll(self):
            return self.returncode

        def terminate(self):
            cleanups.append("server")
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    args = SimpleNamespace(
        artifact_json=manifest,
        dataset_file=tmp_path / "data.parquet",
        llama_server=binary,
        image="fake",
        profile="smoke",
        output=tmp_path / "bundle",
        backend="cpu",
    )

    def run(command, **kwargs):
        if command[1] == "rm":
            cleanups.append("container")
            return SimpleNamespace(returncode=0, stderr="")
        assert kwargs["timeout"] == smoke.CPU_TIMEOUTS["smoke"]
        if failure == "docker":
            raise subprocess.CalledProcessError(1, command)
        if failure == "cancel":
            raise KeyboardInterrupt
        (args.output / "smoke-status.json").write_text(
            json.dumps(
                {
                    "status": "plumbing_passed",
                    "quality_evidence": False,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke.subprocess, "Popen", Server)
    monkeypatch.setattr(smoke.subprocess, "run", run)
    if failure:
        error = {
            "memory": ValueError,
            "docker": subprocess.CalledProcessError,
            "cancel": KeyboardInterrupt,
        }[failure]
        with pytest.raises(error):
            smoke.run(args)
        assert not (args.output / "runner-status.json").exists()
    else:
        smoke.run(args)
        command = launches[0]
        assert command[command.index("--n-gpu-layers") + 1] == "0"
        assert "--device" not in command
        assert (
            json.loads((args.output / "commands.json").read_text())["execution"]["backend"] == "cpu"
        )
    if failure == "memory":
        assert not launches and not cleanups
    else:
        assert "server" in cleanups
        assert (args.output / "artifact-after.json").exists()
        if failure:
            assert "container" in cleanups


@pytest.mark.parametrize(
    "names,status",
    [
        (["Synthetic Intel card"], "ready"),
        (["Synthetic Intel card", "Synthetic Intel card"], "unknown"),
        (["Other device"], "not_ready"),
        (["llvmpipe"], "not_ready"),
    ],
)
def test_evaluation_vulkan_uses_d098_device_confirmation_not_a_guessed_index(names, status):
    from jaull.domain.runtime import (
        LlamaCppBackendCapability,
        LlamaCppBackendCapabilityState,
        LlamaCppBinaryStatus,
        LlamaCppCapabilityReason,
        LlamaCppRuntimeCapability,
    )
    from jaull.runtime.device_matching import match_runtime_device
    from jaull.runtime.llama_cpp_capability import (
        evaluate_execution_readiness,
        parse_llama_cpp_devices,
    )

    hw = hardware(vram_gib=None).model_copy(
        update={
            "accelerators": [
                AcceleratorProfile(
                    name="Synthetic Intel card",
                    vendor=AcceleratorVendor.INTEL,
                    type=AcceleratorType.DEDICATED,
                    backends=[
                        ComputeBackendInfo(
                            backend=ComputeBackend.VULKAN,
                            availability=BackendAvailability.AVAILABLE,
                        )
                    ],
                )
            ]
        }
    )
    selection = quality_backend_selection(hw, ComputeBackend.VULKAN)
    devices = parse_llama_cpp_devices(
        "Available devices:\n" + "\n".join(f"Vulkan{i + 1}: {name}" for i, name in enumerate(names))
    )
    backend = LlamaCppBackendCapability(
        backend=ComputeBackend.VULKAN,
        state=LlamaCppBackendCapabilityState.CONFIRMED,
        devices=devices,
        reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
    )
    capability = LlamaCppRuntimeCapability(
        binary_status=LlamaCppBinaryStatus.AVAILABLE, backend_capabilities=[backend]
    )
    readiness = evaluate_execution_readiness(selection=selection, runtime_capability=capability)
    assert readiness.status.value == status
    if status == "ready":
        device = match_runtime_device(selection, backend).device
        assert device.runtime_id == "Vulkan1"
        command = server_command(Path("/bin/server"), Path("/model.gguf"), device=device.runtime_id)
        assert command[command.index("--device") + 1] == "Vulkan1"
