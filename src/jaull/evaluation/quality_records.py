"""Read-only contract for persisted quality-evaluation records.

This is product code on purpose. The container, the harness glue and the
runners stay in ``pilot/``, which does not ship in the wheel, but a record is
read by the advisor and shown in the interface, so everything needed to decide
whether a record may be *trusted* and *reused* lives here.

Two read paths, deliberately separate:

``quality_lookup`` is strict. It answers "is there reusable evidence for this
exact identity", and refuses a plumbing suite or a partial sample selection,
so a three-example smoke can never reach a ranking.

``describe_record`` is for display. It returns what was measured together with
its grade and its limitations, so the interface can show a diagnostic as a
diagnostic instead of hiding it or dressing it up as evidence.

An identity covers the artifact, the suite, the dataset, the samples, the
evaluator, the runtime *and* the protocol. The runtime part matters: the
placement replay (see ``docs/quality-evaluation-pilot.md``) measured that the
same artifact under a different offload split disagrees on every token
logprob, so an artifact digest alone is not a key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: A record check. The pilot supplies a stronger one than this module can.
RecordValidator = Callable[[dict[str, Any]], None]

#: The three tiers the pilot records. Only ``full`` is reusable evidence:
#: ``plumbing`` proves the protocol executes and ``limited`` explores, so a
#: `--limit 100` run can never become "this model is better".
CLASSIFICATIONS = ("plumbing", "limited", "full")
REUSABLE_CLASSIFICATION = "full"

#: Suites whose purpose is to prove the plumbing works.
PLUMBING_SUITES = frozenset({"jaull-quality-smoke-v1", "jaull-ifeval-chat-smoke-v1"})
#: Suites that by construction run a sample selection, never the whole split.
#: Belt and braces: a record claiming ``full`` on one of these contradicts
#: itself, and ``classification`` alone would already have refused it.
NON_REUSABLE_SUITES = PLUMBING_SUITES | frozenset({"jaull-hellaswag-100-v1"})

IDENTITY_FIELDS = {
    "artifact_sha256", "suite", "dataset", "samples", "evaluator", "runtime", "protocol",
}
EVALUATOR_FIELDS = {
    "evaluator_commit", "backend_sha256", "suite_sha256", "sample_ids", "num_fewshot",
    "context", "apply_chat_template", "system_instruction", "generation", "use_cache",
    "cache_requests", "seeds", "server_seed", "python", "packages",
}
RUNTIME_FIELDS = {
    "server_sha256", "image_id", "fingerprint", "backend_flags", "server_defaults_sha256",
}
DATASET_FIELDS = {"repo", "revision", "sha256", "sample_ids", "total_samples"}
SUPPORTED_PROTOCOL = {
    "mode": "loglikelihood", "tokenize_add_special": True,
    "temperature": 0, "max_tokens": 1, "logprobs": 2, "id_slot": 0,
    "target_logit_bias": 100, "cached_prompt_tokens": 0,
}
#: Schema 2: free-text generation scored by instruction checkers (IFEval).
#: Separate from loglikelihood on purpose; a record never switches between them.
GENERATION_SCHEMA_VERSION = 2
GENERATION_EVALUATOR_FIELDS = {
    "evaluator_commit", "source_sha256", "suite_sha256", "sample_ids", "num_fewshot",
    "context", "apply_chat_template", "chat_template_source", "chat_template_sha256",
    "reasoning", "system_instruction", "generation", "langdetect_seed", "use_cache",
    "cache_requests", "python", "packages",
}
GENERATION_SETTINGS = {"until": [], "temperature": 0, "max_gen_toks": 1280, "seed": 1234}
GENERATION_PROTOCOL = {
    "mode": "chat_generation", "chat_template": "gguf", "reasoning": "off",
    "temperature": 0, "max_tokens": 1280, "stop": [], "seed": 1234, "cached_prompt_tokens": 0,
}
#: Prompt-level metrics count prompts; instruction-level ones count instructions.
PROMPT_METRICS = ("prompt_level_strict_acc", "prompt_level_loose_acc")
INSTRUCTION_METRICS = ("inst_level_strict_acc", "inst_level_loose_acc")
#: Flags that select where the layers run. A record measured under one of these
#: does not describe a run under another.
PLACEMENT_FLAGS = ("--n-gpu-layers", "--device")


def digest(value: Any) -> str:
    """SHA256 over canonical JSON, so an identity has one stable digest."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def known(value: Any) -> bool:
    """Null/empty identity values are unknown, except explicitly checked absences."""
    if isinstance(value, dict):
        return bool(value) and all(known(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(known(item) for item in value)
    return value is not None and value != ""


def validate_identity(identity: dict[str, Any]) -> None:
    protocol = identity.get("protocol")
    if isinstance(protocol, dict) and protocol.get("mode") == GENERATION_PROTOCOL["mode"]:
        _validate_generation_identity(identity)
        return
    if set(identity) != IDENTITY_FIELDS:
        raise ValueError("Missing/unknown identity fields")
    evaluator = identity["evaluator"]
    if set(evaluator) != EVALUATOR_FIELDS:
        raise ValueError("Incomplete evaluator identity")
    absent = {"system_instruction", "generation", "use_cache"}
    if (
        any(evaluator[key] is not None for key in absent)
        or evaluator["apply_chat_template"] is not False
        or evaluator["cache_requests"] is not False
        or not known({key: value for key, value in evaluator.items() if key not in absent})
        or not known({key: value for key, value in identity.items() if key != "evaluator"})
    ):
        raise ValueError("Unknown identity or unsupported chat/generation/cache mode")
    _validate_shared_identity(identity, SUPPORTED_PROTOCOL, (evaluator["backend_sha256"],))


def _validate_generation_identity(identity: dict[str, Any]) -> None:
    """Chat generation: the GGUF's own template, reasoning off, greedy, full budget."""
    if set(identity) != IDENTITY_FIELDS:
        raise ValueError("Missing/unknown identity fields")
    evaluator = identity["evaluator"]
    if set(evaluator) != GENERATION_EVALUATOR_FIELDS:
        raise ValueError("Incomplete evaluator identity")
    absent = {"system_instruction", "use_cache"}
    # Compared with exact constants below; their empty stop lists are deliberate,
    # and `known` would read an empty list as unknown.
    exact = absent | {"generation"}
    sources = evaluator["source_sha256"]
    if (
        any(evaluator[key] is not None for key in absent)
        or evaluator["apply_chat_template"] is not True
        or evaluator["chat_template_source"] != "gguf"
        or evaluator["reasoning"] != "off"
        or evaluator["cache_requests"] is not False
        or evaluator["generation"] != GENERATION_SETTINGS
        or type(evaluator["langdetect_seed"]) is not int
        or evaluator["num_fewshot"] != 0
        or not isinstance(sources, dict) or not sources
        or not known({key: value for key, value in evaluator.items() if key not in exact})
        # Sample arguments carry the task's gen_kwargs (`until: []`); each sample is
        # checked structurally below and compared exactly with the saved results.
        or not known({key: value for key, value in identity.items()
                      if key not in {"evaluator", "protocol", "samples"}})
    ):
        raise ValueError("Unknown identity or unsupported chat/generation/cache mode")
    _validate_shared_identity(
        identity, GENERATION_PROTOCOL, (evaluator["chat_template_sha256"], *sources.values()),
    )
    for sample in identity["samples"]:
        _generation_messages(sample)


def _generation_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    arguments = sample["arguments"]
    if (
        not isinstance(arguments, list) or len(arguments) != 1
        or not isinstance(arguments[0], list) or len(arguments[0]) != 2
        # The pinned harness serializes JsonChatStr as a one-element list.
        or not isinstance(arguments[0][0], list) or len(arguments[0][0]) != 1
        or not isinstance(arguments[0][0][0], str)
        or arguments[0][1] != {
            "until": [], "do_sample": False, "temperature": 0.0, "max_gen_toks": 1280,
        }
    ):
        raise ValueError("Missing/inconsistent generation arguments")
    prompt = arguments[0][0][0]
    messages = json.loads(prompt)
    if (
        not isinstance(messages, list) or len(messages) != 1
        or not isinstance(messages[0], dict) or set(messages[0]) != {"role", "content"}
        or messages[0]["role"] != "user"
        or not isinstance(messages[0]["content"], str) or not messages[0]["content"].strip()
        or hashlib.sha256(prompt.encode()).hexdigest() != sample["prompt_hash"]
        or hashlib.sha256(b"0").hexdigest() != sample["target_hash"]
    ):
        raise ValueError("Unknown/mismatched generation prompt identity")
    return messages


def _validate_shared_identity(
    identity: dict[str, Any], protocol: dict[str, Any], evaluator_digests: tuple[Any, ...],
) -> None:
    evaluator = identity["evaluator"]
    suite, dataset, runtime = (identity[key] for key in ("suite", "dataset", "runtime"))
    if (
        set(suite) != {"name", "sha256"}
        or set(dataset) != DATASET_FIELDS
        or set(runtime) != RUNTIME_FIELDS
        or identity["protocol"] != protocol
    ):
        raise ValueError("Unsupported identity schema/protocol")
    image_id = runtime["image_id"]
    if not isinstance(image_id, str):
        raise ValueError("Unknown exact image digest")
    for sha in (identity["artifact_sha256"], suite["sha256"], dataset["sha256"],
                runtime["server_sha256"], *evaluator_digests,
                runtime["server_defaults_sha256"],
                image_id.removeprefix("sha256:")):
        if not isinstance(sha, str) or re.fullmatch("[0-9a-f]{64}", sha) is None:
            raise ValueError("Unknown exact digest")
    ids = dataset["sample_ids"]
    if (
        type(dataset["total_samples"]) is not int or dataset["total_samples"] < 1
        or not isinstance(ids, list) or not ids
        or any(type(item) is not int or not 0 <= item < dataset["total_samples"] for item in ids)
        or len(set(ids)) != len(ids) or evaluator["sample_ids"] != ids
        or evaluator["suite_sha256"] != suite["sha256"]
        or type(evaluator["context"]) is not int or evaluator["context"] < 1
        or re.fullmatch("[0-9a-f]{40}", evaluator["evaluator_commit"]) is None
        or re.fullmatch("[0-9a-f]{40}", dataset["revision"]) is None
        or not isinstance(evaluator["packages"], dict) or not evaluator["packages"]
        or not isinstance(runtime["backend_flags"], list)
        or any(not isinstance(flag, str) for flag in runtime["backend_flags"])
    ):
        raise ValueError("Incomplete sample/config identity")
    samples = identity["samples"]
    if not isinstance(samples, list) or [sample["doc_id"] for sample in samples] != ids:
        raise ValueError("Incomplete selected sample identity")
    for sample in samples:
        if set(sample) != {"doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments"}:
            raise ValueError("Unknown sample identity")
        for field_name in ("doc_hash", "prompt_hash", "target_hash"):
            if re.fullmatch("[0-9a-f]{64}", sample[field_name]) is None:
                raise ValueError("Unknown sample/prompt digest")


def validate_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") == GENERATION_SCHEMA_VERSION:
        _validate_generation_record(record)
        return
    if record["schema_version"] != 1 or record["status"] != "completed":
        raise ValueError("Unsupported/unfinished result record")
    if identity_mode(record["identity"]) != "loglikelihood":
        raise ValueError("A schema 1 record must describe loglikelihood scoring")
    identity = record["identity"]
    validate_identity(identity)
    if not record["provenance"]["hardware"]:
        raise ValueError("Missing hardware provenance")
    if record["identity_sha256"] != digest(identity):
        raise ValueError("Identity checksum mismatch")
    result, task = record["result"], identity["suite"]["name"]
    ids = identity["dataset"]["sample_ids"]
    if (
        record["classification"] not in CLASSIFICATIONS
        or result["n-samples"][task] != {
            "original": identity["dataset"]["total_samples"], "effective": len(ids),
        }
        or [sample["doc_id"] for sample in result["samples"][task]] != ids
        or not result["results"][task] or not result["configs"][task]
        or any(
            {key: sample[key] for key in expected} != expected
            for sample, expected in zip(result["samples"][task], identity["samples"], strict=True)
        )
    ):
        raise ValueError("Incomplete/mismatched raw results")
    for sample in result["samples"][task]:
        if (
            len(sample["resps"]) != len(sample["arguments"])
            or len(sample["filtered_resps"]) != len(sample["arguments"])
            or any(sample[metric] not in (0, 1) for metric in ("acc", "acc_norm"))
            or "error" in sample
        ):
            raise ValueError("Incomplete sample responses/metrics")
        for repeats in sample["resps"]:
            if not isinstance(repeats, list) or not repeats:
                raise ValueError("Missing continuation scores")
        for score in [value for repeats in sample["resps"] for value in repeats] + sample[
            "filtered_resps"
        ]:
            if (
                not isinstance(score, list) or len(score) != 2
                or type(score[0]) not in (int, float) or not math.isfinite(score[0])
                or score[0] > 0 or type(score[1]) is not bool
            ):
                raise ValueError("Invalid/missing continuation score")
    for metric in ("acc", "acc_norm"):
        value = result["results"][task][metric + ",none"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid/missing task metric")
        # A saved aggregate that disagrees with its own samples is not a record.
        expected = sum(sample[metric] for sample in result["samples"][task]) / len(ids)
        if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Saved metric differs from completed per-sample results")


def identity_mode(identity: dict[str, Any]) -> str | None:
    protocol = identity.get("protocol") if isinstance(identity, dict) else None
    mode = protocol.get("mode") if isinstance(protocol, dict) else None
    return mode if isinstance(mode, str) else None


def _validate_generation_record(record: dict[str, Any]) -> None:
    """Every aggregate is recomputed from per-sample checker outcomes, as in schema 1."""
    if record["status"] != "completed":
        raise ValueError("Unsupported/unfinished result record")
    identity = record["identity"]
    if identity_mode(identity) != GENERATION_PROTOCOL["mode"]:
        raise ValueError("A schema 2 record must describe chat generation")
    validate_identity(identity)
    if not record["provenance"]["hardware"]:
        raise ValueError("Missing hardware provenance")
    if record["identity_sha256"] != digest(identity):
        raise ValueError("Identity checksum mismatch")
    result, task = record["result"], identity["suite"]["name"]
    ids = identity["dataset"]["sample_ids"]
    samples = result["samples"][task]
    if (
        record["classification"] not in CLASSIFICATIONS
        or (task in PLUMBING_SUITES and record["classification"] != "plumbing")
        or result["n-samples"][task] != {
            "original": identity["dataset"]["total_samples"], "effective": len(ids),
        }
        or [sample["doc_id"] for sample in samples] != ids
        or not result["results"][task] or not result["configs"][task]
        or any(
            {key: sample[key] for key in expected} != expected
            for sample, expected in zip(samples, identity["samples"], strict=True)
        )
    ):
        raise ValueError("Incomplete/mismatched raw results")
    for sample in samples:
        messages = _generation_messages(sample)
        document = sample["doc"]
        # Match the pinned harness's document serialization, not just digest shape.
        document_hash = hashlib.sha256(
            json.dumps(document, indent=2, ensure_ascii=False).encode()
        ).hexdigest()
        if (
            document_hash != sample["doc_hash"]
            or document.get("prompt") != messages[0]["content"]
        ):
            raise ValueError("Generation document differs from its prompt identity")
        replies = sample["resps"]
        instructions = sample["doc"]["instruction_id_list"]
        if (
            "error" in sample
            or not isinstance(replies, list) or len(replies) != 1
            or not isinstance(replies[0], list) or len(replies[0]) != 1
            or not isinstance(replies[0][0], str)
            or sample["filtered_resps"] != [replies[0][0]]
            or not isinstance(instructions, list) or not instructions
        ):
            raise ValueError("Incomplete sample responses")
        for prompt_metric, instruction_metric in zip(
            PROMPT_METRICS, INSTRUCTION_METRICS, strict=True,
        ):
            followed = sample[instruction_metric]
            if (
                not isinstance(followed, list) or len(followed) != len(instructions)
                or any(type(item) is not bool for item in followed)
                or type(sample[prompt_metric]) is not bool
                or sample[prompt_metric] != all(followed)
            ):
                raise ValueError("Incomplete/inconsistent instruction outcomes")
    for name, expected in _generation_aggregates(samples).items():
        value = result["results"][task][name + ",none"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid/missing task metric")
        if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Saved metric differs from completed per-sample results")
    outcome = record["outcome"]
    if (
        set(outcome) != {"responses", "truncated"}
        or outcome["responses"] != len(ids)
        or type(outcome["truncated"]) is not int
        or not 0 <= outcome["truncated"] <= outcome["responses"]
    ):
        raise ValueError("Missing/inconsistent generation outcome")


def _generation_aggregates(samples: list[dict[str, Any]]) -> dict[str, float]:
    aggregates = {
        name: sum(sample[name] for sample in samples) / len(samples) for name in PROMPT_METRICS
    }
    for name in INSTRUCTION_METRICS:
        flat = [item for sample in samples for item in sample[name]]
        aggregates[name] = sum(flat) / len(flat)
    return aggregates


def save_record(
    path: Path, record: dict[str, Any], *, validate: RecordValidator = validate_record
) -> None:
    """Write once. A record that already exists is never silently replaced.

    ``validate`` exists because the pilot validates more than this contract can:
    it also pins its own audited profiles and reconstructs every continuation
    from the captured HTTP evidence. The producer passes its stronger check; a
    consumer reading from the store gets this generic one.
    """
    validate(record)
    envelope = {"record": record, "record_sha256": digest(record)}
    # A hard link publishes the fully flushed temporary file without replacing
    # an existing immutable record, including when another writer wins a race.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    if os.name == "posix":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def load_record(
    path: Path, *, validate: RecordValidator = validate_record
) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    record: dict[str, Any] = envelope["record"]
    validate(record)
    if envelope["record_sha256"] != digest(record):
        raise ValueError("Result record checksum mismatch")
    return record


def repeat_key(record: dict[str, Any]) -> str:
    """Equal for repeated runs that gave the same answers under the same identity.

    Identity plus every per-sample response and score; only lm-eval's run date
    is left out. The audited contract is deterministic, so a repeat that matches
    here adds no information and cannot be a choice between results.
    """
    result = record["result"]
    return digest({
        "identity_sha256": record["identity_sha256"],
        "samples": result["samples"], "results": result["results"],
        "outcome": record.get("outcome"),
    })


def is_reusable_evidence(record: dict[str, Any]) -> bool:
    """A complete run of a non-plumbing suite over the whole split, or nothing."""
    identity = record["identity"]
    dataset = identity["dataset"]
    return bool(
        record["classification"] == REUSABLE_CLASSIFICATION
        and identity["suite"]["name"] not in NON_REUSABLE_SUITES
        and dataset["sample_ids"] == list(range(dataset["total_samples"]))
    )


def quality_lookup(
    path: Path, identity: dict[str, Any], *, validate: RecordValidator = validate_record
) -> dict[str, Any] | None:
    """Read only. Return raw harness data; no speed evidence or hardware-based match."""
    try:
        validate_identity(identity)
        record = load_record(path, validate=validate)
        if record["identity_sha256"] != digest(identity) or not is_reusable_evidence(record):
            return None
        result: dict[str, Any] = record["result"]
        return result
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
        return None


def placement_of(record: dict[str, Any]) -> dict[str, str]:
    """The launch flags that decide where the layers ran, as a comparable mapping."""
    flags = list(record["identity"]["runtime"]["backend_flags"])
    placement: dict[str, str] = {}
    # llama.cpp applies the last occurrence of a repeated launch flag.
    for index, name in enumerate(flags):
        if name in PLACEMENT_FLAGS:
            placement.pop(name, None)
            if index + 1 < len(flags):
                placement[name] = flags[index + 1]
    return placement


@dataclass(frozen=True)
class QualityMetric:
    name: str
    value: float
    correct: int
    samples: int


@dataclass(frozen=True)
class QualityEvidence:
    """What a record measured, with the grade and the limits attached.

    Built for display. ``reusable`` is the same question ``quality_lookup``
    answers, carried alongside the numbers so a screen cannot show the metrics
    without the grade that qualifies them.
    """

    identity_sha256: str
    artifact_sha256: str
    suite: str
    dataset: str
    dataset_revision: str
    metrics: tuple[QualityMetric, ...]
    samples_used: int
    samples_available: int
    evaluated_at: datetime | None
    evaluator_commit: str
    runtime_fingerprint: str
    placement: dict[str, str]
    hardware: str | None
    classification: str
    reusable: bool
    limitations: tuple[str, ...] = field(default=())
    context_length: int | None = None
    #: ``quality_comparison.comparison_fields``: what must match for two runs to compare.
    comparison: tuple[tuple[str, str], ...] = field(default=())

    @property
    def summary(self) -> str:
        parts = [f"{metric.name} {metric.correct}/{metric.samples}" for metric in self.metrics]
        return ", ".join(parts) if parts else "no metrics"


def _hardware_label(provenance: dict[str, Any]) -> str | None:
    """A short label for the machine, from whatever shape the provenance holds.

    ``validate_record`` only requires the hardware provenance to be non-empty,
    so it may be a profile mapping or a plain note. Display copes with both
    rather than the contract narrowing to suit one screen.
    """
    hardware = provenance.get("hardware")
    if isinstance(hardware, str):
        return hardware or None
    if not isinstance(hardware, dict):
        return None
    gpus = hardware.get("gpus")
    if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict):
        name = gpus[0].get("name")
        if isinstance(name, str) and name:
            return name
    cpu = hardware.get("cpu")
    name = cpu.get("name") if isinstance(cpu, dict) else None
    return name if isinstance(name, str) and name else None


def _evaluated_at(result: dict[str, Any]) -> datetime | None:
    # lm-eval writes its own Unix timestamp; nothing here invents a date.
    stamp = result.get("date")
    if not isinstance(stamp, int | float) or not math.isfinite(stamp):
        return None
    try:
        return datetime.fromtimestamp(stamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def describe_record(record: dict[str, Any]) -> QualityEvidence:
    """Project a validated record onto what an interface may show about it."""
    # The comparator imports this module; the reverse import waits until here.
    from jaull.evaluation.quality_comparison import comparison_fields

    validate_record(record)
    identity = record["identity"]
    dataset, result = identity["dataset"], record["result"]
    task = identity["suite"]["name"]
    samples = result["samples"][task]
    generation = record["schema_version"] == GENERATION_SCHEMA_VERSION
    if generation:
        metrics = tuple(
            QualityMetric(
                name=name,
                value=float(result["results"][task][f"{name},none"]),
                correct=int(sum(flat)),
                samples=len(flat),
            )
            for name in (*PROMPT_METRICS, *INSTRUCTION_METRICS)
            for flat in ([
                item for sample in samples
                for item in (sample[name] if name in INSTRUCTION_METRICS else [sample[name]])
            ],)
        )
    else:
        metrics = tuple(
            QualityMetric(
                name=name,
                value=float(result["results"][task][f"{name},none"]),
                correct=int(sum(sample[name] for sample in samples)),
                samples=len(samples),
            )
            for name in ("acc", "acc_norm")
        )
    limitations: list[str] = []
    if generation:
        limitations.append(
            "Scored under this artifact's own chat template with reasoning off; whatever "
            "that template adds, such as a default system prompt, is part of the result."
        )
        truncated = record["outcome"]["truncated"]
        if truncated:
            limitations.append(
                f"{truncated} of {record['outcome']['responses']} replies reached the "
                "1280-token cap and are scored as written."
            )
        limitations.append(
            "Language checks use a seeded detector: Jaull's IFEval, not comparable with "
            "published IFEval figures."
        )
    if task in PLUMBING_SUITES:
        limitations.append(
            "Plumbing suite: this run proves the protocol executes, not model quality."
        )
    if dataset["sample_ids"] != list(range(dataset["total_samples"])):
        limitations.append(
            f"{len(dataset['sample_ids'])} of {dataset['total_samples']} samples. "
            "Too few samples to separate two models is not a tie, it is no answer."
        )
    if record["classification"] == "limited":
        limitations.append(
            "Limited run: exploration, not a verdict. Report a margin, not a point."
        )
    elif record["classification"] != REUSABLE_CLASSIFICATION:
        limitations.append(
            f"Classified {record['classification']!r}, so it is not reusable evidence."
        )
    limitations.append(
        "Measured under one placement. The same artifact under a different offload "
        "split produces different scores."
    )
    return QualityEvidence(
        identity_sha256=record["identity_sha256"],
        artifact_sha256=identity["artifact_sha256"],
        suite=task,
        dataset=dataset["repo"],
        dataset_revision=dataset["revision"],
        metrics=metrics,
        samples_used=len(dataset["sample_ids"]),
        samples_available=int(dataset["total_samples"]),
        evaluated_at=_evaluated_at(result),
        evaluator_commit=identity["evaluator"]["evaluator_commit"],
        runtime_fingerprint=identity["runtime"]["fingerprint"],
        placement=placement_of(record),
        hardware=_hardware_label(record["provenance"]),
        classification=record["classification"],
        reusable=is_reusable_evidence(record),
        limitations=tuple(limitations),
        context_length=identity["evaluator"]["context"],
        comparison=comparison_fields(record),
    )


__all__ = [
    "CLASSIFICATIONS",
    "NON_REUSABLE_SUITES",
    "PLACEMENT_FLAGS",
    "PLUMBING_SUITES",
    "REUSABLE_CLASSIFICATION",
    "QualityEvidence",
    "QualityMetric",
    "RecordValidator",
    "describe_record",
    "digest",
    "is_reusable_evidence",
    "known",
    "load_record",
    "placement_of",
    "quality_lookup",
    "save_record",
    "validate_identity",
    "validate_record",
]
