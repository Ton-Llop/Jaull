"""Offline protocol and launch checks. All HTTP values here are synthetic."""

import errno
import socket
from copy import deepcopy
from pathlib import Path

import pytest
from pilot.quality_eval.evaluate import (
    ARTIFACT_PINS,
    CONTEXT,
    DATASET_REVISION,
    DATASET_URL,
    TASK,
    validate_artifact_identity,
    validate_scoring_request,
    validate_scoring_response,
    validate_task,
    verified_dataset_bytes,
    write_json,
)
from scripts.quality_eval_smoke import (
    artifact_memory_estimate,
    check_server_port,
    docker_command,
    server_command,
)

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import HardwareFitMode
from tests._execution_fixtures import qwen_hardware
from tests._gguf_fixtures import build_header


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


def test_quality_smoke_rejects_changes_under_the_fixed_suite_name() -> None:
    def process_docs(docs):
        return docs

    # Synthetic stand-in; the container loads this function from the pinned harness.
    process_docs.__module__ = "lm_eval.tasks.hellaswag.utils"
    task = {
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


def test_two_exact_artifacts_and_local_fit_fail_closed(tmp_path: Path) -> None:
    for digest, pin in ARTIFACT_PINS.items():
        identity = pin | {"sha256": digest}
        validate_artifact_identity(identity)
        for change in ({"sha256": "0" * 64}, {"revision": "main"}, {"quantization": "Q8_0"}):
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
