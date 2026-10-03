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
import re
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
PLUMBING_SUITES = frozenset({"jaull-quality-smoke-v1"})
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
    suite, dataset, runtime = (identity[key] for key in ("suite", "dataset", "runtime"))
    if (
        set(suite) != {"name", "sha256"}
        or set(dataset) != DATASET_FIELDS
        or set(runtime) != RUNTIME_FIELDS
        or identity["protocol"] != SUPPORTED_PROTOCOL
    ):
        raise ValueError("Unsupported identity schema/protocol")
    image_id = runtime["image_id"]
    if not isinstance(image_id, str):
        raise ValueError("Unknown exact image digest")
    for sha in (identity["artifact_sha256"], suite["sha256"], dataset["sha256"],
                runtime["server_sha256"], evaluator["backend_sha256"],
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
    if record["schema_version"] != 1 or record["status"] != "completed":
        raise ValueError("Unsupported/unfinished result record")
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
    # Serialize before exclusive creation; interrupted/truncated files are lookup misses.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        handle.write("\n")


def load_record(
    path: Path, *, validate: RecordValidator = validate_record
) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    record: dict[str, Any] = envelope["record"]
    validate(record)
    if envelope["record_sha256"] != digest(record):
        raise ValueError("Result record checksum mismatch")
    return record


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
    validate_record(record)
    identity = record["identity"]
    dataset, result = identity["dataset"], record["result"]
    task = identity["suite"]["name"]
    samples = result["samples"][task]
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
