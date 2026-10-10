"""IFEval chat contract checks. Synthetic payloads; no Docker, server, model or harness."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from pilot.quality_eval import ifeval
from pilot.quality_eval.evaluate import PROFILES, profile_for_task
from pilot.quality_eval.records import snapshot_bundle
from pilot.quality_eval.setup import validate_evaluator_image
from scripts.quality_eval_smoke import docker_command, server_command

from jaull.domain.artifacts import ModelArtifact
from tests._execution_fixtures import qwen_hardware
from tests._gguf_fixtures import build_header
from tests._quality_eval_fixtures import ARTIFACT_PINS

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {
    "messages": [{"role": "user", "content": "Write a haiku in all lowercase."}],
    "model": ifeval.MODEL_NAME, "max_tokens": 1280, "temperature": 0, "stop": [], "seed": 1234,
}
RESPONSE = {
    "system_fingerprint": ifeval.RUNTIME_FINGERPRINT,
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "quiet pond at dusk"}}],
    "usage": {"prompt_tokens": 40, "completion_tokens": 6,
              "prompt_tokens_details": {"cached_tokens": 0}},
}


def _changed(base, path, value):
    changed = copy.deepcopy(base)
    target = changed
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return changed


def test_the_audited_request_passes() -> None:
    ifeval.validate_chat_request(REQUEST)


@pytest.mark.parametrize("path,value", [
    (("top_p",), 1.0),                                   # A key the protocol never sends.
    (("seed",), KeyError),
    (("stop",), ["\n\n"]),
    (("temperature",), 0.7),
    (("max_tokens",), 256),
    (("model",), "other"),
    (("messages",), [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]),
    (("messages", 0, "role"), "system"),
    (("messages", 0, "content"), "   "),
])
def test_any_other_request_is_refused(path, value) -> None:
    with pytest.raises(ValueError, match="audited IFEval protocol"):
        ifeval.validate_chat_request(_changed(REQUEST, path, value))


def test_a_clean_reply_reports_how_it_finished() -> None:
    assert ifeval.validate_chat_response(RESPONSE) == "stop"
    assert ifeval.validate_chat_response(
        _changed(RESPONSE, ("choices", 0, "finish_reason"), "length")) == "length"


@pytest.mark.parametrize("path,value,match", [
    (("system_fingerprint",), "b9999-0000000", "runtime"),
    (("choices", 0, "message", "reasoning_content"), "let me think", "Reasoning"),
    (("choices", 0, "message", "content"), "<think>hmm</think>quiet pond", "Reasoning"),
    (("choices", 0, "message", "content"), None, "assistant text"),
    (("choices", 0, "finish_reason"), "tool_calls", "finish_reason"),
    (("usage", "prompt_tokens_details", "cached_tokens"), 12, "cache"),
    (("usage", "prompt_tokens"), ifeval.CONTEXT - ifeval.MAX_GEN_TOKS + 1, "context"),
    (("usage",), KeyError, "usage"),
    (("usage", "prompt_tokens"), -1, "usage"),
    (("usage", "prompt_tokens"), 0, "usage"),
    (("usage", "completion_tokens"), -1, "usage"),
    (("usage", "completion_tokens"), True, "usage"),
])
def test_a_reply_that_breaks_the_contract_fails_the_run(path, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        ifeval.validate_chat_response(_changed(RESPONSE, path, value))


def test_truncated_replies_are_counted_and_kept() -> None:
    log = [{"finish_reason": "stop"}, {"finish_reason": "length"}, {"error": "x"}]
    assert ifeval.summarise_generations(log) == {"responses": 2, "truncated": 1}


def test_server_props_bind_the_template_the_gguf_carries() -> None:
    artifact = {"local_path": "/models/a.gguf"}
    props = {"total_slots": 1, "default_generation_settings": {"n_ctx": ifeval.CONTEXT},
             "model_path": "/models/a.gguf", "chat_template": "{{ messages }}"}
    assert ifeval.validate_server_props(props, artifact) == (
        hashlib.sha256(b"{{ messages }}").hexdigest()
    )
    for broken in ({"chat_template": ""}, {"total_slots": 2},
                   {"default_generation_settings": {"n_ctx": 2048}}):
        with pytest.raises(ValueError):
            ifeval.validate_server_props(props | broken, artifact)


@pytest.mark.parametrize("template", [None, "", "   ", "chatml", ["chatml"], "{{ messages }}"])
def test_ifeval_preflight_requires_a_real_embedded_template(tmp_path, template) -> None:
    from scripts.quality_eval_smoke import artifact_memory_estimate

    metadata = {
        "general.architecture": "qwen2", "qwen2.context_length": 32768,
        "qwen2.embedding_length": 1536, "qwen2.block_count": 28,
        "qwen2.attention.head_count": 12, "qwen2.attention.head_count_kv": 2,
    }
    if template is not None:
        metadata["tokenizer.chat_template"] = template
    model = tmp_path / "model.gguf"
    model.write_bytes(build_header(metadata))
    sha, pin = next(iter(ARTIFACT_PINS.items()))
    artifact = ModelArtifact(**(pin | {
        "sha256": sha, "local_path": model, "size_bytes": model.stat().st_size,
    }))
    # Raw HellaSwag does not require a template, including on historical artifacts.
    artifact_memory_estimate(artifact, qwen_hardware())
    if template == "{{ messages }}":
        estimate = artifact_memory_estimate(
            artifact, qwen_hardware(), context=ifeval.CONTEXT, require_chat_template=True,
        )
        assert estimate.inference_configuration.context_length == ifeval.CONTEXT
    else:
        with pytest.raises(ValueError, match="embedded GGUF chat template"):
            artifact_memory_estimate(
                artifact, qwen_hardware(), context=ifeval.CONTEXT, require_chat_template=True,
            )


def _named(module: str, name: str):
    def function(*args, **kwargs):
        return None

    function.__module__, function.__name__ = module, name
    return function


def _suite():
    utils = "lm_eval.tasks.ifeval.utils"
    agg = _named(utils, "agg_inst_level_acc")
    return {
        "task": "jaull-ifeval-chat-v1", "dataset_path": "json", "dataset_name": None,
        "dataset_kwargs": {"data_files": {"train": ifeval.DATASET_URL}},
        "output_type": "generate_until", "test_split": "train", "num_fewshot": 0,
        "doc_to_text": "prompt", "doc_to_target": 0,
        "generation_kwargs": {"until": [], "do_sample": False, "temperature": 0.0,
                              "max_gen_toks": 1280},
        "process_results": _named(utils, "process_results"),
        "metric_list": [
            {"metric": "prompt_level_strict_acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "inst_level_strict_acc", "aggregation": agg, "higher_is_better": True},
            {"metric": "prompt_level_loose_acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "inst_level_loose_acc", "aggregation": agg, "higher_is_better": True},
        ],
        "metadata": {"version": 1.0, "upstream_task_version": 4.0,
                     "dataset_repo": "google/IFEval", "dataset_revision": ifeval.DATASET_REVISION},
    }


def test_only_the_pinned_suite_definition_is_accepted() -> None:
    ifeval.validate_task(_suite())
    for change in ({"generation_kwargs": {"until": ["\n"], "do_sample": False,
                                          "temperature": 0.0, "max_gen_toks": 1280}},
                   {"num_fewshot": 3},
                   {"dataset_kwargs": {"data_files": {"train": "https://example.invalid/x"}}},
                   {"process_results": _named("somewhere.else", "process_results")}):
        with pytest.raises(ValueError, match="fixed zero-shot IFEval"):
            ifeval.validate_task(_suite() | change)


def test_the_suite_file_declares_what_the_validator_expects() -> None:
    text = (ROOT / "pilot/quality_eval/suite_ifeval.yaml").read_text()
    assert ifeval.DATASET_URL in text
    assert "max_gen_toks: 1280" in text and "until: []" in text
    assert "!function lm_eval.tasks.ifeval.utils.process_results" in text


def test_a_dataset_other_than_the_pinned_file_is_refused(tmp_path) -> None:
    path = tmp_path / "ifeval.jsonl"
    path.write_bytes(b'{"key": 1}\n')
    with pytest.raises(ValueError, match="SHA256"):
        ifeval.verified_dataset_bytes(path)


def test_profiles_keep_smoke_as_plumbing_and_only_the_full_split_reusable() -> None:
    assert PROFILES["ifeval-smoke"]["classification"] == "plumbing"
    assert PROFILES["ifeval"]["classification"] == "full"
    assert PROFILES["ifeval"]["sample_ids"] == list(range(ifeval.DATASET_SIZE))
    assert profile_for_task("jaull-ifeval-chat-smoke-v1")["suite"] == "ifeval"


def test_chat_suites_add_template_and_reasoning_flags_only_when_asked() -> None:
    default = server_command(Path("/server"), Path("/model.gguf"))
    assert "--jinja" not in default and "--reasoning" not in default
    chat = server_command(Path("/server"), Path("/model.gguf"), context=4096, chat=True)
    assert chat[chat.index("--reasoning") + 1] == "off"
    assert "--jinja" in chat
    assert chat[chat.index("--ctx-size") + 1] == "4096"
    # Everything else is the pilot's audited launch, unchanged.
    assert [f for f in chat if f not in {"--jinja", "--reasoning", "off", "4096"}] == [
        f for f in default if f != "2048"
    ]


@pytest.mark.skipif(os.name != "posix", reason="Docker host command uses Linux/WSL UID/GID")
def test_ifeval_mounts_its_own_dataset_and_names_its_profile() -> None:
    command = docker_command("img", "name", Path("/out"), Path("/m.gguf"), Path("/d.jsonl"),
                             profile="ifeval-smoke")
    assert "type=bind,source=/d.jsonl,target=/dataset/ifeval.jsonl,readonly" in command
    assert command[-2:] == ["--profile", "ifeval-smoke"]


def test_an_image_without_the_chat_suite_is_refused_for_ifeval_only() -> None:
    old = {"Config": {"Labels": {"io.jaull.quality.artifact-contract": "exact-local-gguf-v1"}}}
    validate_evaluator_image(old)  # HellaSwag images keep working.
    with pytest.raises(ValueError, match="predates the IFEval"):
        validate_evaluator_image(old, suite="ifeval")
    new = copy.deepcopy(old)
    new["Config"]["Labels"]["io.jaull.quality.suites"] = "hellaswag-v1,ifeval-chat-v1"
    validate_evaluator_image(new, suite="ifeval")


def test_the_image_pins_punkt_and_ships_the_ifeval_runner() -> None:
    dockerfile = (ROOT / "pilot/quality_eval/Dockerfile").read_text()
    assert ("--checksum=sha256:e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
            in dockerfile)
    assert "ifeval.py" in dockerfile and "suite_ifeval.yaml" in dockerfile
    assert "verify_installed_sources()" in dockerfile
    lock = (ROOT / "pilot/quality_eval/requirements.lock").read_text()
    assert "langdetect==1.0.9" in lock and "immutabledict==4.3.1" in lock


# llama.cpp cuts its tokenizer-merges preview mid-character for LFM2.5-1.2B; a
# real 541-prompt run once failed to import on exactly this server.log line.
SPLIT_UTF8_LOG = b'tokenizer.ggml.merges arr[str,63683] = ["\xc4\x8a \xc4\x8a", "\xc4...\n'


@pytest.mark.parametrize("server_log", [b"", SPLIT_UTF8_LOG])
def test_an_ifeval_bundle_is_never_imported_through_the_hellaswag_checks(
    tmp_path, server_log,
) -> None:
    from pilot.quality_eval.records import JSON_FILES, TEXT_FILES

    for name in JSON_FILES:
        (tmp_path / name).write_text("{}")
    for name in TEXT_FILES:
        (tmp_path / name).write_text("")
    (tmp_path / "server.log").write_bytes(server_log)
    (tmp_path / "verified-artifact.json").write_text(json.dumps({
        "format": "gguf", "repo_id": "org/m", "filename": "m.gguf", "quantization": "Q4_K_M",
        "local_path": "/models/m.gguf", "sha256": "a" * 64, "revision": "b" * 40,
        "size_bytes": 1,
    }))
    (tmp_path / "lm-eval-results.json").write_text(
        json.dumps({"samples": {"jaull-ifeval-chat-smoke-v1": []}}),
    )
    # Execution itself verified, so the profile decides which checks run next.
    artifact = json.loads((tmp_path / "verified-artifact.json").read_text())
    status = {"status": "ifeval_plumbing_passed", "quality_evidence": False}
    (tmp_path / "smoke-status.json").write_text(json.dumps(status))
    (tmp_path / "runner-status.json").write_text(json.dumps(status | {"server_exit_code": 0}))
    (tmp_path / "artifact-after.json").write_text(json.dumps(artifact))
    (tmp_path / "container-artifact.json").write_text(
        json.dumps({"sha256": artifact["sha256"], "mount": "read-only"}),
    )
    # The generation path asks for template evidence; HellaSwag's would ask for its parquet.
    with pytest.raises(OSError, match=r"chat-template\.json"):
        snapshot_bundle(tmp_path)
