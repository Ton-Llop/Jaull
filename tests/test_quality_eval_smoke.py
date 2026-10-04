"""Offline protocol and launch checks. All HTTP values here are synthetic."""

import errno
import hashlib
import json
import os
import socket
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from pilot.quality_eval.evaluate import (
    ARTIFACT_CONTRACT,
    ARTIFACT_CONTRACT_LABEL,
    CONTEXT,
    DATASET_REVISION,
    DATASET_URL,
    LIMITED_TASK,
    PROFILES,
    TASK,
    profile_for_task,
    validate_artifact_identity,
    validate_scoring_request,
    validate_scoring_response,
    validate_task,
    verified_dataset_bytes,
    write_json,
)
from pilot.quality_eval.setup import validate_evaluator_image
from scripts.quality_eval_smoke import (
    artifact_memory_estimate,
    check_server_port,
    docker_command,
    server_command,
)

from jaull.artifacts.errors import ArtifactVerificationError
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import HardwareFitMode
from tests._execution_fixtures import qwen_hardware
from tests._gguf_fixtures import build_header
from tests._quality_eval_fixtures import ARTIFACT_PINS


@pytest.mark.skipif(os.name != "posix", reason="Docker host command uses Linux/WSL UID/GID")
def test_quality_smoke_rejects_protocol_drift_and_preserves_previous_output(tmp_path: Path) -> None:
    payload = {
        "prompt": [1, 450],
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": 2,
        "logit_bias": [[265, 100]],
        "id_slot": 0,
    }
    response = {
        "system_fingerprint": "b10357-689e227db",
        "usage": {
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens": 1,
            "prompt_tokens": 2,
        },
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "id": 265,
                            "logprob": -13.8,
                            "top_logprobs": [{"id": 3681, "logprob": -0.96}],
                        }
                    ]
                }
            }
        ],
    }
    validate_scoring_request(payload)
    validate_scoring_response(payload, response)
    for change in (
        {"temperature": 0.5},
        {"top_p": 0.1},
        {"max_tokens": 8},
        {"id_slot": 1},
        {"prompt": [1] * CONTEXT},
        {"logit_bias": [[265, 10]]},
    ):
        with pytest.raises(ValueError):
            validate_scoring_request(payload | change)
    for key, value in (
        ("id", 266),
        ("logprob", float("nan")),
        ("logprob", float("inf")),
        ("logprob", 1),
        ("top_logprobs", []),
    ):
        bad = deepcopy(response)
        bad["choices"][0]["logprobs"]["content"][0][key] = value
        with pytest.raises(ValueError):
            validate_scoring_response(payload, bad)
    bad = deepcopy(response)
    bad["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    with pytest.raises(ValueError):
        validate_scoring_response(payload, bad)
    path = tmp_path / "result.json"
    write_json(path, {"status": "failed"})
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "passed"})
    assert '"failed"' in path.read_text()
    launch = server_command(Path("/server"), Path("/model.gguf"))
    assert "--no-cache-prompt" in launch and "--no-context-shift" in launch
    assert launch[launch.index("--parallel") + 1] == "1"
    command = docker_command(
        "sha256:synthetic", "owned-test", tmp_path, Path("/model.gguf"), Path("/data.parquet")
    )
    assert "type=bind,source=/model.gguf,target=/artifact/model.gguf,readonly" in command
    assert "type=bind,source=/data.parquet,target=/dataset/validation.parquet,readonly" in command
    assert "--gpus" not in command  # evaluator does not execute the model
    assert "--use_cache" not in command and "--gen_kwargs" not in command


def synthetic_task_config():
    def process_docs(docs):
        return docs

    # Synthetic stand-in; the container loads this function from the pinned harness.
    process_docs.__module__ = "lm_eval.tasks.hellaswag.utils"
    return {
        "task": TASK,
        "dataset_path": "parquet",
        "dataset_name": None,
        "dataset_kwargs": {"data_files": {"validation": DATASET_URL}},
        "output_type": "multiple_choice",
        "validation_split": "validation",
        "training_split": None,
        "test_split": None,
        "num_fewshot": 0,
        "process_docs": process_docs,
        "doc_to_text": "{{query}}",
        "doc_to_target": "{{label}}",
        "doc_to_choice": "choices",
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
        ],
        "metadata": {
            "version": 1.0,
            "dataset_repo": "Rowan/hellaswag",
            "dataset_revision": DATASET_REVISION,
        },
    }


def test_quality_smoke_rejects_changes_under_the_fixed_suite_name() -> None:
    task = synthetic_task_config()
    validate_task(task)
    for change in (
        {"output_type": "loglikelihood_rolling"},
        {"generation_kwargs": {"top_p": 1}},
        {"num_fewshot": 1},
        {"dataset_kwargs": {"revision": "main"}},
        {"doc_to_text": "Answer: {{query}}"},
        {"doc_to_target": "0"},
        {"doc_to_choice": ["a", "b"]},
        {"process_docs": lambda docs: docs},
        {"validation_split": None, "test_split": "validation"},
        {"metric_list": []},
        {"metadata": task["metadata"] | {"version": 2.0}},
        {"apply_chat_template": True},
    ):
        with pytest.raises(ValueError):
            validate_task(task | change)
    for key in task:
        with pytest.raises(ValueError):
            validate_task({name: value for name, value in task.items() if name != key})


def test_historical_artifacts_and_local_fit_fail_closed(tmp_path: Path) -> None:
    for digest, pin in ARTIFACT_PINS.items():
        identity = pin | {"sha256": digest, "local_path": "/synthetic/model.gguf"}
        validate_artifact_identity(identity)
        for change in ({"sha256": "invalid"}, {"revision": "main"}, {"quantization": None}):
            with pytest.raises(ValueError):
                validate_artifact_identity(identity | change)
    with pytest.raises(ValueError):
        validate_artifact_identity({})
    dataset = tmp_path / "bad.parquet"
    dataset.write_bytes(b"synthetic wrong dataset")
    with pytest.raises(ValueError, match="SHA256"):
        verified_dataset_bytes(dataset)

    # Synthetic header/hardware, using the pinned file size only as an estimate input.
    model = tmp_path / "model.gguf"
    model.write_bytes(build_header({
        "general.architecture": "qwen2",
        "qwen2.context_length": 32768,
        "qwen2.embedding_length": 1536,
        "qwen2.block_count": 28,
        "qwen2.attention.head_count": 12,
        "qwen2.attention.head_count_kv": 2,
    }))
    digest, pin = list(ARTIFACT_PINS.items())[1]
    artifact = ModelArtifact(**pin, sha256=digest, local_path=model)
    hardware = qwen_hardware()
    estimate = artifact_memory_estimate(artifact, hardware)
    assert estimate.weights.component.bytes == pin["size_bytes"]
    assert estimate.kv_cache.component.bytes == 56 * 1024**2
    assert estimate.hardware_fit.mode is HardwareFitMode.GPU_RESIDENT
    assert estimate.inference_configuration.context_length == CONTEXT
    assert estimate.device_reserve.bytes == 512 * 1024**2
    assert estimate.configuration_sources["num_hidden_layers"].value == "gguf_header"
    low_gpu = hardware.gpus[0].model_copy(update={"vram_available_bytes": 1024**2})
    low = artifact_memory_estimate(artifact, hardware.model_copy(update={"gpus": [low_gpu]}))
    assert low.hardware_fit.mode is not HardwareFitMode.GPU_RESIDENT
    model.write_bytes(build_header({"general.architecture": "qwen2"}))
    with pytest.raises(ValueError, match="configuration/context"):
        artifact_memory_estimate(artifact, hardware)


def synthetic_artifact_identity() -> dict:
    return {
        "repo_id": "synthetic/new-model", "revision": "a" * 40,
        "filename": "model.Q4_K_M.gguf", "format": "gguf", "quantization": "Q4_K_M",
        "sha256": "b" * 64, "size_bytes": 128, "local_path": "/synthetic/model.gguf",
    }


def test_quality_accepts_exact_unlisted_artifacts_and_rejects_incomplete_identity() -> None:
    identity = synthetic_artifact_identity()
    assert identity["sha256"] not in ARTIFACT_PINS
    validate_artifact_identity(identity)
    validate_artifact_identity(identity | {"repo_id": "another/fine-tune", "sha256": "c" * 64})
    for key in identity:
        with pytest.raises(ValueError):
            validate_artifact_identity({
                field: value for field, value in identity.items() if field != key
            })
    for change in (
        {"revision": "main"}, {"sha256": "f" * 63}, {"size_bytes": 0},
        {"size_bytes": True}, {"size_bytes": "128"}, {"format": "safetensors"},
        {"filename": "../model.gguf"}, {"filename": "/model.gguf"},
        {"filename": "model-00001-of-00003.gguf"}, {"filename": "model.bin"},
        {"quantization": None}, {"local_path": ""}, {"local_path": "relative/model.gguf"},
        {"repo_id": " "},
    ):
        with pytest.raises(ValueError):
            validate_artifact_identity(identity | change)


def test_quality_image_contract_is_explicit_and_requires_a_rebuild() -> None:
    validate_evaluator_image({"Config": {"Labels": {ARTIFACT_CONTRACT_LABEL: ARTIFACT_CONTRACT}}})
    for image in ({}, {"Config": {}}, {"Config": None}, {"Config": {"Labels": None}},
                  {"Config": {"Labels": "invalid"}},
                  {"Config": {"Labels": {ARTIFACT_CONTRACT_LABEL: "old"}}}):
        with pytest.raises(ValueError, match="rebuild"):
            validate_evaluator_image(image)
    dockerfile = Path("pilot/quality_eval/Dockerfile").read_text()
    assert f'LABEL {ARTIFACT_CONTRACT_LABEL}="{ARTIFACT_CONTRACT}"' in dockerfile


@pytest.mark.parametrize("split", [{"split.count": 3}, {"split.no": 1}, {"split.count": "1"}])
def test_quality_rejects_split_headers_even_with_a_single_file_name(tmp_path: Path, split: dict):
    model = tmp_path / "model.gguf"
    model.write_bytes(build_header({
        "general.architecture": "qwen2", "qwen2.context_length": 32768,
        "qwen2.embedding_length": 1536, "qwen2.block_count": 28,
        "qwen2.attention.head_count": 12, "qwen2.attention.head_count_kv": 2,
    } | split))
    artifact = ModelArtifact(**(synthetic_artifact_identity() | {
        "local_path": model, "size_bytes": model.stat().st_size,
    }))
    with pytest.raises(ValueError, match="Multipart"):
        artifact_memory_estimate(artifact, qwen_hardware())


@pytest.mark.parametrize("failure", ["revision", "size", "sha", "image", "fit", "metadata"])
@pytest.mark.skipif(os.name != "posix", reason="Host preflight requires Linux/WSL local paths")
def test_quality_generic_preflight_fails_before_launching_server_or_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Synthetic GGUF metadata/bytes; never starts a process or accesses Docker."""
    from pilot.quality_eval import setup
    from scripts import quality_eval_smoke as smoke

    model, server = tmp_path / "model.gguf", tmp_path / "server"
    metadata = {
        "general.architecture": "qwen2", "qwen2.context_length": 32768,
        "qwen2.embedding_length": 1536, "qwen2.block_count": 28,
        "qwen2.attention.head_count": 12, "qwen2.attention.head_count_kv": 2,
    }
    if failure == "metadata":
        metadata = {"general.architecture": "qwen2", "qwen2.context_length": 32768}
    model.write_bytes(build_header(metadata))
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    ArtifactStorage().save_sha256(model, sha)
    artifact = synthetic_artifact_identity() | {
        "local_path": str(model), "sha256": sha, "size_bytes": model.stat().st_size,
    }
    if failure == "revision":
        artifact["revision"] = "main"
    elif failure == "size":
        artifact["size_bytes"] += 1
    elif failure == "sha":
        model.write_bytes(model.read_bytes() + b"wrong bytes")
        artifact["size_bytes"] = model.stat().st_size  # Size matches, checksum does not.
    server.write_bytes(b"synthetic server, not executable")
    server.chmod(0o700)
    artifact_json = tmp_path / "artifact.json"
    artifact_json.write_text(json.dumps(artifact))
    monkeypatch.setattr(setup, "prepare_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "SERVER_SHA256", hashlib.sha256(server.read_bytes()).hexdigest())
    hardware = qwen_hardware()
    if failure == "fit":
        hardware = hardware.model_copy(update={"gpus": [
            hardware.gpus[0].model_copy(update={"vram_available_bytes": 1}),
        ]})
    monkeypatch.setattr(smoke, "detect_hardware", lambda: hardware)
    monkeypatch.setattr(smoke, "select_runtime_backend", lambda hardware: SimpleNamespace(
        selected_backend=smoke.ComputeBackend.CUDA,
    ))
    capability = SimpleNamespace()
    readiness = SimpleNamespace(status=smoke.ExecutionReadinessStatus.READY,
                                model_dump=lambda **kwargs: {})
    monkeypatch.setattr(smoke, "inspect_llama_cpp_runtime", lambda **kwargs: capability)
    monkeypatch.setattr(smoke, "evaluate_execution_readiness", lambda **kwargs: readiness)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("Preflight must not launch a server/container"))
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("Preflight must not execute a container"))

    def image_inspect(*args, **kwargs):
        assert failure in ("image", "fit", "metadata")
        labels = {} if failure == "image" else {ARTIFACT_CONTRACT_LABEL: ARTIFACT_CONTRACT}
        return json.dumps([{"Id": "sha256:" + "c" * 64, "RepoDigests": [],
                            "Config": {"Labels": labels}}])

    monkeypatch.setattr(smoke.subprocess, "check_output", image_inspect)
    args = SimpleNamespace(profile="smoke", output=tmp_path / "run", artifact_json=artifact_json,
                           dataset_file=tmp_path / "data.parquet", llama_server=server,
                           image="synthetic:old")
    reason = {"revision": "revision", "size": "Size mismatch", "sha": "SHA-256 mismatch",
              "image": "rebuild", "fit": "full-device memory fit",
              "metadata": "full-device memory fit"}[failure]
    with pytest.raises((ValueError, ArtifactVerificationError), match=reason):
        smoke.run(args)
    assert (args.output / "runner-error.json").is_file()
    assert not (args.output / "runner-status.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="Regression exercises Linux socket TIME_WAIT rules")
def test_port_probe_reuses_time_wait_and_rejects_an_active_listener() -> None:
    try:
        listener = socket.socket()
    except PermissionError:
        pytest.skip("Sandbox denies sockets; run this check with approved socket access")
    with listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Sandbox denies loopback bind; run this check with approved socket access")
        port = listener.getsockname()[1]
        listener.listen()
        with pytest.raises(OSError) as error:
            check_server_port(port)
        assert error.value.errno == errno.EADDRINUSE
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    with socket.socket() as probe, pytest.raises(OSError) as error:
        probe.bind(("127.0.0.1", port))
    assert error.value.errno == errno.EADDRINUSE  # TIME_WAIT reproduces the original failure.
    check_server_port(port)


@pytest.mark.skipif(os.name != "posix", reason="Docker host command uses Linux/WSL UID/GID")
def test_larger_profile_is_fixed_and_explicitly_forwarded_to_the_container(tmp_path: Path):
    profile = PROFILES["hellaswag100"]
    ids = profile["sample_ids"]
    assert profile["task"] == LIMITED_TASK and profile["classification"] == "limited"
    assert len(ids) == len(set(ids)) == 100 and ids == sorted(ids)
    assert all(0 <= index < 10042 for index in ids)
    # Freeze the selection independently of its sampling implementation.
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == (
        "efa126f5252a5e211a8f65c5fee35efbfcf0754e367123e129cd3ea192ef207f"
    )
    assert PROFILES["smoke"]["sample_ids"] == [0, 1, 2]
    assert profile_for_task(TASK) == PROFILES["smoke"]
    assert profile_for_task(LIMITED_TASK) == profile
    with pytest.raises(ValueError, match="Unknown"):
        profile_for_task("unversioned-task")
    command = docker_command("sha256:synthetic", "owned", tmp_path, Path("/model"),
                             Path("/dataset"), profile="hellaswag100")
    assert command[-2:] == ["--profile", "hellaswag100"]
    default = docker_command("sha256:synthetic", "owned", tmp_path, Path("/model"), Path("/data"))
    assert "--profile" not in default  # Legacy three-example launch stays unchanged.
    with pytest.raises(ValueError, match="Unknown"):
        docker_command("image", "owned", tmp_path, Path("/model"), Path("/data"), profile="full")


@pytest.mark.parametrize("profile_name", ["smoke", "hellaswag100"])
def test_selected_profile_reaches_the_harness_without_enabling_caches(tmp_path, monkeypatch,
                                                                     profile_name):
    """Synthetic harness/modules; no Docker, server, model inference or real results."""
    from pilot.quality_eval import evaluate

    profile = PROFILES[profile_name]
    source = tmp_path / "synthetic-gguf.py"
    source.write_bytes(b"synthetic backend")
    monkeypatch.setattr(evaluate, "BACKEND_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    model_bytes = b"synthetic model, not GGUF"
    sha = hashlib.sha256(model_bytes).hexdigest()
    pin = next(iter(ARTIFACT_PINS.values()))
    write_json(tmp_path / "verified-artifact.json", pin | {"sha256": sha,
                                                         "local_path": "/synthetic/model"})
    original_open = Path.open

    def open_model(path, *args, **kwargs):
        if path == Path("/artifact/model.gguf"):
            return BytesIO(model_bytes)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_model)
    monkeypatch.setattr(evaluate, "verified_dataset_bytes", lambda path: b"synthetic dataset")
    props = {"total_slots": 1, "model_path": "/synthetic/model",
             "default_generation_settings": {"n_ctx": CONTEXT,
                                               "params": {"post_sampling_probs": False}}}
    requests = SimpleNamespace(get=lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: props,
    ))

    class SyntheticGGUF:
        def __init__(self, **kwargs):
            pass

    def simple_evaluate(**kwargs):
        assert kwargs["samples"] == {profile["task"]: profile["sample_ids"]}
        assert kwargs["tasks"][0]["task"] == profile["task"]
        assert kwargs["use_cache"] is None and kwargs["cache_requests"] is False
        assert kwargs["apply_chat_template"] is False and kwargs["num_fewshot"] == 0
        return {"samples": {profile["task"]: [{"doc_id": i} for i in profile["sample_ids"]]},
                "n-samples": {profile["task"]: {"original": 10042,
                                                  "effective": len(profile["sample_ids"])}}}

    for name, module in {
        "requests": requests,
        "lm_eval": SimpleNamespace(simple_evaluate=simple_evaluate),
        "lm_eval.models": SimpleNamespace(gguf=SimpleNamespace(__file__=str(source),
                                                               GGUFLM=SyntheticGGUF)),
        "lm_eval.tasks": SimpleNamespace(TaskManager=lambda **kwargs: None),
        "lm_eval.tasks._yaml_loader": SimpleNamespace(load_yaml=lambda *args, **kwargs:
                                                      synthetic_task_config()),
        "lm_eval.utils": SimpleNamespace(handle_non_serializable=str),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    evaluate.run(tmp_path, "http://synthetic.invalid", profile_name)
    config = json.loads((tmp_path / "evaluator-config.json").read_text())
    assert config["sample_ids"] == profile["sample_ids"]
    status = json.loads((tmp_path / "smoke-status.json").read_text())
    assert status == {"status": profile["status"], "quality_evidence": False}
