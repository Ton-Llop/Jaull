"""Schema 2 (chat generation) records. Synthetic data; not measured IFEval results."""

import copy
import hashlib
import json

import pytest
from pilot.quality_eval import ifeval
from pilot.quality_eval.records import validate_generation_http

from jaull.evaluation.quality_records import (
    GENERATION_PROTOCOL,
    GENERATION_SETTINGS,
    describe_record,
    digest,
    is_reusable_evidence,
    quality_lookup,
    save_record,
    validate_record,
)

FULL, SMOKE = "jaull-ifeval-chat-v1", "jaull-ifeval-chat-smoke-v1"
PROMPTS = ["Write in lowercase.", "Use no commas, then say yes."]
FOLLOWED = [[True], [True, False]]
REPLIES = ["all lowercase here", "no commas yes"]


def _sample(doc_id: int) -> dict:
    followed = FOLLOWED[doc_id]
    prompt = json.dumps([{"role": "user", "content": PROMPTS[doc_id]}])
    document = {
        "prompt": PROMPTS[doc_id],
        "instruction_id_list": [f"rule:{i}" for i in range(len(followed))],
    }
    return {
        "doc_id": doc_id,
        "doc_hash": hashlib.sha256(json.dumps(document, indent=2, ensure_ascii=False)
                                   .encode()).hexdigest(),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_hash": hashlib.sha256(b"0").hexdigest(),
        "arguments": [[[prompt], {"until": [], "do_sample": False, "temperature": 0.0,
                                  "max_gen_toks": 1280}]],
        "doc": document,
        "resps": [[REPLIES[doc_id]]], "filtered_resps": [REPLIES[doc_id]],
        "prompt_level_strict_acc": all(followed), "inst_level_strict_acc": list(followed),
        "prompt_level_loose_acc": all(followed), "inst_level_loose_acc": list(followed),
    }


def record(task: str = FULL, classification: str = "full", *, truncated: int = 0) -> dict:
    ids = [0, 1]
    samples = [_sample(i) for i in ids]
    identity = {
        "artifact_sha256": "a" * 64,
        "suite": {"name": task, "sha256": "b" * 64},
        "dataset": {"repo": "google/IFEval", "revision": "c" * 40, "sha256": "d" * 64,
                    "sample_ids": ids, "total_samples": 2},
        "samples": [{k: s[k] for k in ("doc_id", "doc_hash", "prompt_hash", "target_hash",
                                        "arguments")} for s in samples],
        "evaluator": {
            "evaluator_commit": "e" * 40, "source_sha256": {"lm_eval.x": "f" * 64},
            "suite_sha256": "b" * 64, "sample_ids": ids, "num_fewshot": 0, "context": 4096,
            "apply_chat_template": True, "chat_template_source": "gguf",
            "chat_template_sha256": "1" * 64, "reasoning": "off", "system_instruction": None,
            "generation": dict(GENERATION_SETTINGS), "langdetect_seed": 0, "use_cache": None,
            "cache_requests": False, "python": "3.12.12", "packages": {"lm_eval": "0.4.14.dev0"},
        },
        "runtime": {"server_sha256": "2" * 64, "image_id": "sha256:" + "3" * 64,
                    "fingerprint": "b10357-689e227db", "backend_flags": ["--jinja"],
                    "server_defaults_sha256": "4" * 64},
        "protocol": dict(GENERATION_PROTOCOL),
    }
    results = {
        "prompt_level_strict_acc,none": 0.5, "prompt_level_loose_acc,none": 0.5,
        "inst_level_strict_acc,none": 2 / 3, "inst_level_loose_acc,none": 2 / 3,
    }
    return {
        "schema_version": 2, "status": "completed", "classification": classification,
        "identity": identity, "identity_sha256": digest(identity),
        "result": {"n-samples": {task: {"original": 2, "effective": 2}},
                   "samples": {task: samples}, "results": {task: results},
                   "configs": {task: {"task": task}}},
        "outcome": {"responses": 2, "truncated": truncated},
        "provenance": {"hardware": {"gpus": [{"name": "Synthetic GPU"}]}},
    }


def _rehash(value: dict) -> dict:
    value["identity_sha256"] = digest(value["identity"])
    return value


def test_a_complete_generation_record_is_valid_and_counts_both_levels() -> None:
    validate_record(record())
    evidence = describe_record(record(truncated=1))
    counts = {m.name: (m.correct, m.samples) for m in evidence.metrics}
    # Prompts count prompts; instruction metrics count the three instructions.
    assert counts["prompt_level_strict_acc"] == (1, 2)
    assert counts["inst_level_strict_acc"] == (2, 3)
    assert any("own chat template" in item for item in evidence.limitations)
    assert any("1 of 2 replies reached the 1280-token cap" in item
               for item in evidence.limitations)
    assert evidence.reusable


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["result"]["results"][FULL].__setitem__("prompt_level_strict_acc,none", 1.0),
     "Saved metric differs"),
    (lambda r: r["result"]["samples"][FULL][1].__setitem__("prompt_level_strict_acc", True),
     "inconsistent instruction outcomes"),
    (lambda r: r["result"]["samples"][FULL][0].__setitem__("inst_level_strict_acc", [True, True]),
     "inconsistent instruction outcomes"),
    (lambda r: r["result"]["samples"][FULL][0].__setitem__("filtered_resps", ["other"]),
     "Incomplete sample responses"),
    (lambda r: r["result"]["samples"][FULL][0].__setitem__("resps", [["a", "b"]]),
     "Incomplete sample responses"),
    (lambda r: r.__setitem__("outcome", {"responses": 2, "truncated": 3}), "generation outcome"),
    (lambda r: r.__setitem__("outcome", {"responses": 1, "truncated": 0}), "generation outcome"),
])
def test_records_that_disagree_with_themselves_are_refused(mutate, match) -> None:
    broken = record()
    mutate(broken)
    with pytest.raises(ValueError, match=match):
        validate_record(broken)


@pytest.mark.parametrize("path,value", [
    (("evaluator", "reasoning"), "on"),
    (("evaluator", "chat_template_source"), "runtime"),
    (("evaluator", "apply_chat_template"), False),
    (("evaluator", "generation"), {**GENERATION_SETTINGS, "max_gen_toks": 256}),
    (("evaluator", "source_sha256"), {"lm_eval.x": "not-a-digest"}),
    (("evaluator", "chat_template_sha256"), "short"),
    (("evaluator", "system_instruction"), "Be brief."),
    (("protocol", "temperature"), 0.7),
])
def test_any_departure_from_the_chat_protocol_is_refused(path, value) -> None:
    broken = record()
    target = broken["identity"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_record(_rehash(broken))


def test_schema_versions_never_cross() -> None:
    as_v1 = record()
    as_v1["schema_version"] = 1
    with pytest.raises(ValueError, match="loglikelihood"):
        validate_record(as_v1)
    loglikelihood = record()
    loglikelihood["identity"]["protocol"] = {"mode": "loglikelihood"}
    with pytest.raises(ValueError):
        validate_record(_rehash(loglikelihood))


def test_a_smoke_is_plumbing_and_never_reusable(tmp_path) -> None:
    smoke = record(SMOKE, "plumbing")
    validate_record(smoke)
    assert not is_reusable_evidence(smoke)
    with pytest.raises(ValueError):
        validate_record(record(SMOKE, "full"))
    path = tmp_path / "smoke.json"
    save_record(path, smoke)
    assert quality_lookup(path, smoke["identity"]) is None


def test_a_full_record_round_trips_through_strict_lookup(tmp_path) -> None:
    full = record()
    path = tmp_path / "full.json"
    save_record(path, full)
    assert quality_lookup(path, full["identity"]) == full["result"]


@pytest.mark.parametrize("arguments", [None, [], [[[], {}]], [["not JsonChatStr", {}]],
                                       [[['not JSON'], {"until": []}]]])
def test_unknown_generation_arguments_cannot_become_reusable(arguments, tmp_path) -> None:
    broken = record()
    broken["identity"]["samples"][0]["arguments"] = arguments
    broken["result"]["samples"][FULL][0]["arguments"] = arguments
    _rehash(broken)
    with pytest.raises(ValueError, match="generation arguments"):
        validate_record(broken)
    with pytest.raises(ValueError):
        save_record(tmp_path / "invalid.json", broken)
    assert not (tmp_path / "invalid.json").exists()
    assert quality_lookup(tmp_path / "invalid.json", broken["identity"]) is None


@pytest.mark.parametrize("field,value", [
    ("prompt", "A completely different request"),
    ("instruction_id_list", ["a-different-rule"]),
    ("kwargs", [{"minimum": 1000}]),
])
def test_changed_documents_never_compare_under_the_old_identity(field, value) -> None:
    from jaull.evaluation.quality_comparison import compare_quality_records

    changed = record()
    changed["result"]["samples"][FULL][0]["doc"][field] = value
    with pytest.raises(ValueError, match="document differs"):
        compare_quality_records(record(), changed, left_source="l", right_source="r")


def test_rehashing_a_changed_document_does_not_bind_it_to_the_old_prompt() -> None:
    changed = record()
    sample = changed["result"]["samples"][FULL][0]
    sample["doc"]["prompt"] = "A completely different request"
    document_hash = hashlib.sha256(json.dumps(sample["doc"], indent=2, ensure_ascii=False)
                                  .encode()).hexdigest()
    sample["doc_hash"] = document_hash
    changed["identity"]["samples"][0]["doc_hash"] = document_hash
    with pytest.raises(ValueError, match="document differs"):
        validate_record(_rehash(changed))


@pytest.mark.parametrize("change", ["json", "role", "settings", "prompt_hash", "target_hash"])
def test_generation_prompt_identity_is_checked_not_just_digest_shape(change) -> None:
    changed = record()
    sample = changed["identity"]["samples"][0]
    if change == "json":
        sample["arguments"][0][0] = ["not JSON"]
    elif change == "role":
        sample["arguments"][0][0] = [json.dumps([{"role": "system", "content": "A prompt"}])]
    elif change == "settings":
        sample["arguments"][0][1]["temperature"] = 0.7
    else:
        sample[change] = "8" * 64
    changed["result"]["samples"][FULL][0].update(copy.deepcopy(sample))
    with pytest.raises(ValueError):
        validate_record(_rehash(changed))


def test_partial_generation_record_never_reaches_lookup(tmp_path) -> None:
    complete = record()
    path = tmp_path / "full.json"
    save_record(path, complete)
    partial = record()
    partial["status"] = "interrupted"
    with pytest.raises(ValueError, match="unfinished"):
        save_record(tmp_path / "partial.json", partial)
    partial_identity = copy.deepcopy(complete["identity"])
    partial_identity["dataset"]["sample_ids"] = [0]
    partial_identity["evaluator"]["sample_ids"] = [0]
    partial_identity["samples"] = partial_identity["samples"][:1]
    assert quality_lookup(path, partial_identity) is None


def _exchange(index: int, finish: str = "stop", reply: str | None = None) -> str:
    request = {"messages": [{"role": "user", "content": PROMPTS[index]}],
               "model": ifeval.MODEL_NAME, "max_tokens": 1280, "temperature": 0,
               "stop": [], "seed": 1234}
    response = {
        "system_fingerprint": ifeval.RUNTIME_FINGERPRINT,
        "choices": [{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": REPLIES[index] if reply is None else reply}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }
    return json.dumps({"url": "http://h/v1/chat/completions", "request": request,
                       "rendered_prompt": "<|user|>" + PROMPTS[index], "response": response})


SAMPLES = [_sample(0), _sample(1)]


def test_every_scored_reply_is_traced_to_one_audited_exchange() -> None:
    # Harness order differs from doc order; matching is by prompt, not position.
    text = "\n".join([_exchange(1, "length"), _exchange(0)])
    assert validate_generation_http(text, SAMPLES) == (
        ifeval.RUNTIME_FINGERPRINT, {"responses": 2, "truncated": 1},
    )


@pytest.mark.parametrize("lines,match", [
    ([0], "Missing generation evidence"),
    ([0, 1, 1], "Duplicate generation"),
    (["altered"], "Scored reply differs"),
])
def test_http_evidence_must_cover_exactly_what_was_scored(lines, match) -> None:
    text = "\n".join(
        _exchange(0, reply="something else") + "\n" + _exchange(1) if item == "altered"
        else _exchange(item) for item in lines
    )
    with pytest.raises(ValueError, match=match):
        validate_generation_http(text, copy.deepcopy(SAMPLES))


def _other_artifact(base: dict, template: str) -> dict:
    other = copy.deepcopy(base)
    other["identity"]["artifact_sha256"] = "9" * 64
    other["identity"]["evaluator"]["chat_template_sha256"] = template * 64
    sample = other["result"]["samples"][FULL][1]
    sample["inst_level_strict_acc"] = sample["inst_level_loose_acc"] = [True, True]
    sample["prompt_level_strict_acc"] = sample["prompt_level_loose_acc"] = True
    other["result"]["results"][FULL] = {
        "prompt_level_strict_acc,none": 1.0, "prompt_level_loose_acc,none": 1.0,
        "inst_level_strict_acc,none": 1.0, "inst_level_loose_acc,none": 1.0,
    }
    return _rehash(other)


def test_two_models_each_under_its_own_template_are_comparable() -> None:
    from jaull.evaluation.quality_comparison import compare_quality_records

    left, right = record(), _other_artifact(record(), "8")
    report = compare_quality_records(left, right, left_source="l", right_source="r")

    assert report["status"] == "COMPARABLE_DIAGNOSTIC"
    by_metric = {item["metric"]: item for item in report["per_task"]}
    assert by_metric["prompt_level_strict_acc"]["left"]["correct"] == 1
    assert by_metric["prompt_level_strict_acc"]["right"]["correct"] == 2
    assert by_metric["inst_level_strict_acc"]["left"]["samples"] == 3
    assert "clustered within prompts" in by_metric["inst_level_strict_acc"]["uncertainty"]["reason"]


def test_a_template_from_elsewhere_is_never_comparable() -> None:
    from jaull.evaluation.quality_comparison import compare_quality_records

    right = _other_artifact(record(), "8")
    right["identity"]["evaluator"]["python"] = "3.13.0"
    report = compare_quality_records(record(), _rehash(right), left_source="l", right_source="r")
    assert report["status"] == "NOT_COMPARABLE" and report["per_task"] == []


def test_loglikelihood_and_generation_records_never_compare() -> None:
    from jaull.evaluation.quality_comparison import compare_quality_records
    from tests.test_quality_eval_records import synthetic_full_record

    report = compare_quality_records(synthetic_full_record(), record(),
                                     left_source="l", right_source="r")
    assert report["status"] == "NOT_COMPARABLE"
    assert {"field": "schema_version", "match": False} in report["checks"]


def test_store_lookup_counts_identical_repeats_as_one_and_disagreeing_ones_as_none(
    tmp_path,
) -> None:
    from jaull.evaluation.quality_storage import QualityEvidenceStore

    first = record()
    repeat = copy.deepcopy(first)
    repeat["result"]["date"] = 1  # Same answers; only lm-eval's run date moved.
    representative = min(first, repeat, key=digest)["result"]
    for name, order in (("same", (first, repeat)), ("reversed", (repeat, first))):
        store = QualityEvidenceStore(tmp_path / name)
        for item in order:
            store.save(item)
        assert len(store.list_ids()) == 2  # Both runs are kept as written.
        # The shadow's representative, whatever order the runs were stored in.
        assert store.lookup(first["identity"]) == representative

    other = copy.deepcopy(first)
    sample = other["result"]["samples"][FULL][0]
    sample["resps"] = [["A different reply."]]
    sample["filtered_resps"] = ["A different reply."]
    store = QualityEvidenceStore(tmp_path / "different")
    store.save(first)
    store.save(other)
    assert store.lookup(first["identity"]) is None
