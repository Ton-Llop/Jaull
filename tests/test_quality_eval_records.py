"""Synthetic offline records only; none of these fixtures is measured quality."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pilot.quality_eval.evaluate import ARTIFACT_PINS, TASK
from pilot.quality_eval.records import (
    JSON_FILES,
    digest,
    load_record,
    quality_lookup,
    save_record,
    snapshot_bundle,
    validate_record,
)


def synthetic_full_record():
    task = "synthetic-full-task-v1"
    sample = {
        "doc_id": 0, "doc_hash": "a" * 64, "prompt_hash": "b" * 64,
        "target_hash": "c" * 64, "arguments": [["Synthetic prompt", " choice"]],
    }
    identity = {
        "artifact_sha256": "d" * 64,
        "suite": {"name": task, "sha256": "e" * 64},
        "dataset": {"repo": "synthetic/dataset", "revision": "f" * 40,
                    "sha256": "0" * 64, "sample_ids": [0], "total_samples": 1},
        "samples": [sample],
        "evaluator": {
            "evaluator_commit": "1" * 40, "backend_sha256": "2" * 64,
            "suite_sha256": "e" * 64, "sample_ids": [0], "num_fewshot": 0,
            "context": 2048, "apply_chat_template": False, "system_instruction": None,
            "generation": None, "use_cache": None, "cache_requests": False,
            "seeds": [0, 1234, 1234, 1234], "server_seed": 0, "python": "3.12.12",
            "packages": {"lm_eval": "synthetic-version"},
        },
        "runtime": {
            "server_sha256": "3" * 64, "image_id": "sha256:" + "4" * 64,
            "fingerprint": "synthetic-build", "backend_flags": ["--parallel", "1"],
            "server_defaults_sha256": "6" * 64,
        },
        "protocol": {
            "mode": "loglikelihood", "tokenize_add_special": True,
            "temperature": 0, "max_tokens": 1, "logprobs": 2, "id_slot": 0,
            "target_logit_bias": 100, "cached_prompt_tokens": 0,
        },
    }
    return {
        "schema_version": 1, "status": "completed", "classification": "full",
        "identity": identity, "identity_sha256": digest(identity),
        "result": {
            "results": {task: {"acc,none": 1.0, "acc_norm,none": 1.0}},
            "configs": {task: {"task": task}},
            "n-samples": {task: {"original": 1, "effective": 1}},
            "samples": {task: [sample | {
                "doc": {"synthetic": True}, "resps": [[[-1.0, False]]],
                "filtered_resps": [[-1.0, False]], "acc": 1, "acc_norm": 1,
            }]},
        },
        "provenance": {"hardware": "synthetic hardware, no real measurement"},
    }


def test_quality_record_identity_reuse_and_immutable_full_results(tmp_path: Path) -> None:
    record = synthetic_full_record()
    path = tmp_path / "record.json"
    save_record(path, record)
    before = path.read_bytes()
    identity = record["identity"]
    assert quality_lookup(path, identity) == record["result"]
    changed = deepcopy(identity)
    changed["evaluator"]["seeds"][0] = False  # Python considers False == 0.
    assert changed == identity and digest(changed) != digest(identity)
    assert quality_lookup(path, changed) is None
    for field, subfield, value in (
        (None, "artifact_sha256", "5" * 64),
        ("runtime", "image_id", "sha256:" + "5" * 64),
        ("runtime", "server_sha256", "5" * 64),
        ("runtime", "server_defaults_sha256", "5" * 64),
        ("runtime", "backend_flags", ["--parallel", "2"]),
        ("dataset", "revision", "5" * 40),
        ("dataset", "sha256", "5" * 64),
        ("evaluator", "context", 4096),
        ("evaluator", "seeds", [1, 1234, 1234, 1234]),
        ("evaluator", "apply_chat_template", True),
        ("protocol", "temperature", 0.5),
    ):
        changed = deepcopy(identity)
        (changed if field is None else changed[field])[subfield] = value
        assert quality_lookup(path, changed) is None
    for section in identity:
        missing = deepcopy(identity)
        del missing[section]
        assert quality_lookup(path, missing) is None
    changed = deepcopy(identity)
    changed["runtime"]["fingerprint"] = None
    assert quality_lookup(path, changed) is None
    for section in ("suite", "dataset", "runtime", "evaluator"):
        for key in identity[section]:
            missing = deepcopy(identity)
            del missing[section][key]
            assert quality_lookup(path, missing) is None
    with pytest.raises(FileExistsError):
        save_record(path, record)
    assert path.read_bytes() == before
    # Hardware is outside the reuse identity; only full harness quality data is returned.
    relocated = deepcopy(record)
    relocated["provenance"]["hardware"] = "different synthetic hardware"
    relocated_path = tmp_path / "relocated.json"
    save_record(relocated_path, relocated)
    assert quality_lookup(relocated_path, identity) == record["result"]


def test_quality_record_rejects_incomplete_unknown_limited_and_tampered_results(tmp_path: Path):
    record = synthetic_full_record()
    identity = record["identity"]
    path = tmp_path / "record.json"
    for changed in (
        record | {"status": "failed"}, record | {"status": "partial"},
        record | {"classification": "plumbing"},
    ):
        path.write_text(json.dumps({"record": changed, "record_sha256": digest(changed)}))
        assert quality_lookup(path, identity) is None
    incomplete = deepcopy(record)
    incomplete["result"]["samples"][identity["suite"]["name"]] = []
    with pytest.raises(ValueError, match="Incomplete"):
        save_record(tmp_path / "incomplete.json", incomplete)
    assert not (tmp_path / "incomplete.json").exists()
    incomplete = deepcopy(record)
    del incomplete["result"]["samples"][identity["suite"]["name"]][0]["resps"]
    path.write_text(json.dumps({"record": incomplete, "record_sha256": digest(incomplete)}))
    assert quality_lookup(path, identity) is None
    incomplete = deepcopy(record)
    incomplete["result"]["samples"][identity["suite"]["name"]][0]["resps"] = [[]]
    path.write_text(json.dumps({"record": incomplete, "record_sha256": digest(incomplete)}))
    assert quality_lookup(path, identity) is None
    limited = deepcopy(record)
    limited["identity"]["dataset"]["total_samples"] = 2
    limited["result"]["n-samples"][identity["suite"]["name"]]["original"] = 2
    limited["identity_sha256"] = digest(limited["identity"])
    save_record(tmp_path / "limited.json", limited)
    assert quality_lookup(tmp_path / "limited.json", limited["identity"]) is None
    # Renaming a smoke as full must not make it quality evidence.
    smoke = deepcopy(record)
    old = smoke["identity"]["suite"]["name"]
    smoke["identity"]["suite"]["name"] = TASK
    smoke["identity_sha256"] = digest(smoke["identity"])
    for key in ("results", "configs", "n-samples", "samples"):
        smoke["result"][key][TASK] = smoke["result"][key].pop(old)
    save_record(tmp_path / "smoke.json", smoke)
    assert quality_lookup(tmp_path / "smoke.json", smoke["identity"]) is None
    save_record(tmp_path / "valid.json", record)
    envelope = json.loads((tmp_path / "valid.json").read_text())
    envelope["record"]["result"]["results"][old]["acc,none"] = 0
    path.write_text(json.dumps(envelope))
    assert quality_lookup(path, identity) is None
    for contents in ("{", "null", "{}"):
        path.write_text(contents)
        assert quality_lookup(path, identity) is None
    assert quality_lookup(tmp_path / "absent.json", identity) is None
    validate_record(record)


@pytest.mark.parametrize("metric", ["acc", "acc_norm"])
def test_inconsistent_aggregate_is_rejected_even_with_a_valid_checksum(tmp_path: Path, metric: str):
    record = synthetic_full_record()
    task = record["identity"]["suite"]["name"]
    record["result"]["results"][task][metric + ",none"] = 0
    path = tmp_path / "record.json"
    with pytest.raises(ValueError, match="Saved metric"):
        save_record(path, record)
    assert not path.exists()
    path.write_text(json.dumps({"record": record, "record_sha256": digest(record)}))
    before = path.read_bytes()
    assert quality_lookup(path, record["identity"]) is None
    assert path.read_bytes() == before


def test_snapshot_checks_actual_http_and_retains_full_raw_evidence(tmp_path: Path, monkeypatch):
    record = synthetic_full_record()
    task = record["identity"]["suite"]["name"]
    result = record["result"]
    for key in ("results", "configs", "n-samples", "samples"):
        result[key][TASK] = result[key].pop(task)
    sample = result["samples"][TASK][0]
    result["samples"][TASK] = [sample | {"doc_id": i} for i in range(3)]
    result["n-samples"][TASK] = {"original": 10042, "effective": 3}
    evaluator = record["identity"]["evaluator"] | {"sample_ids": [0, 1, 2]}
    sha, pin = next(iter(ARTIFACT_PINS.items()))
    artifact = pin | {"sha256": sha, "local_path": "/synthetic/model.gguf"}
    data = b"synthetic dataset, not parquet or measured data"
    # Only file verification is replaced; the evidence/parser/protocol guards run normally.
    monkeypatch.setattr("pilot.quality_eval.records.verified_dataset_bytes", lambda path: data)
    evidence = {name: {} for name in JSON_FILES} | {
        "verified-artifact.json": artifact, "artifact-after.json": artifact,
        "container-artifact.json": {"sha256": sha, "mount": "read-only"},
        "commands.json": {"server": ["/synthetic/server", "--model", artifact["local_path"],
                                     "--parallel", "1"], "server_sha256": "3" * 64},
        "image.json": {"id": "sha256:" + "4" * 64},
        "hardware.json": {"name": "synthetic GPU", "unavailable_attribution": None},
        "server-props.json": {
            "total_slots": 1, "model_path": artifact["local_path"],
            "default_generation_settings": {
                "n_ctx": 2048, "params": {"post_sampling_probs": False},
            },
        },
        "evaluator-config.json": evaluator,
        "dataset.json": {"repo": "synthetic/dataset", "revision": "f" * 40,
                         "sha256": hashlib.sha256(data).hexdigest(), "sample_ids": [0, 1, 2]},
        "lm-eval-results.json": result,
        "smoke-status.json": {"status": "plumbing_passed", "quality_evidence": False},
        "runner-status.json": {"status": "plumbing_passed", "quality_evidence": False,
                               "server_exit_code": 0},
    }
    for name, value in evidence.items():
        (tmp_path / name).write_text(json.dumps(value))
    for name in ("server.log", "evaluator.log"):
        (tmp_path / name).write_text("Synthetic log; no inference happened.\n")
    scoring = {
        "url": "http://host.docker.internal:18083/v1/completions",
        "request": {"prompt": [1], "temperature": 0, "max_tokens": 1,
                    "logprobs": 2, "logit_bias": [[2, 100]], "id_slot": 0},
        "response": {
            "system_fingerprint": "b10357-689e227db",
            "usage": {"completion_tokens": 1, "prompt_tokens": 1,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "choices": [{"logprobs": {"content": [{
                "id": 2, "logprob": -0.4, "top_logprobs": [{"logprob": -0.4}],
            }]}}],
        },
    }
    tokenization = {
        "url": "http://host.docker.internal:18083/tokenize",
        "request": {"content": "Synthetic prompt", "add_special": True},
        "response": {"tokens": [1]},
    }
    whole = deepcopy(tokenization)
    whole["request"]["content"] += " choice"
    whole["response"]["tokens"] = [1, 2, 3]
    second = deepcopy(scoring)
    second["request"]["prompt"] = [1, 2]
    second["request"]["logit_bias"] = [[3, 100]]
    second["response"]["usage"]["prompt_tokens"] = 2
    second["response"]["choices"][0]["logprobs"]["content"] = [{
        "id": 3, "logprob": -0.6, "top_logprobs": [{"logprob": -0.6}],
    }]
    # Tokenization is cached by text, but every selected choice must be scored.
    exchanges = [tokenization, whole, *([scoring, second] * 3)]
    http = tmp_path / "http.jsonl"
    http.write_text("".join(json.dumps(exchange) + "\n" for exchange in exchanges))
    snapshot = snapshot_bundle(tmp_path)
    save_record(tmp_path / "snapshot.json", snapshot)
    assert snapshot["result"] == result
    assert snapshot["provenance"]["evidence"]["http.jsonl"] == http.read_text()
    assert snapshot["provenance"]["hardware"] == evidence["hardware.json"]
    assert quality_lookup(tmp_path / "snapshot.json", snapshot["identity"]) is None
    assert load_record(tmp_path / "snapshot.json") == snapshot
    truncated = deepcopy(snapshot)
    truncated["provenance"]["evidence"]["http.jsonl"] = json.dumps(scoring) + "\n"
    damaged = tmp_path / "truncated-record.json"
    damaged.write_text(json.dumps({"record": truncated, "record_sha256": digest(truncated)}))
    with pytest.raises(ValueError, match="tokenizer evidence"):
        load_record(damaged)
    original = http.read_text()
    for incomplete in (
        [scoring],                         # The original one-response loophole.
        exchanges[1:],                     # Missing prefix tokenization.
        exchanges[:-1],                    # Missing final token score.
        [*exchanges, scoring],             # Unexpected duplicate scoring.
        [tokenization, whole, *([scoring] * 6)],  # Correct count, wrong coverage.
    ):
        http.write_text("".join(json.dumps(exchange) + "\n" for exchange in incomplete))
        with pytest.raises(ValueError, match=r"evidence|coverage"):
            snapshot_bundle(tmp_path)
    # Valid payload/response shapes do not establish that the saved score was measured.
    inconsistent = deepcopy(exchanges)
    inconsistent[-1]["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] = -0.7
    http.write_text("".join(json.dumps(exchange) + "\n" for exchange in inconsistent))
    with pytest.raises(ValueError, match="reconstruct"):
        snapshot_bundle(tmp_path)
    http.write_text(original)
    tokenization["request"]["add_special"] = False
    http.write_text("".join(json.dumps(exchange) + "\n" for exchange in exchanges))
    with pytest.raises(ValueError, match="tokenizer"):
        snapshot_bundle(tmp_path)
    (tmp_path / "runner-error.json").write_text('{"error":"synthetic failure"}')
    with pytest.raises(ValueError, match="Failed bundle"):
        snapshot_bundle(tmp_path)
