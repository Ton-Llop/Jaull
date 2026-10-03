"""Synthetic comparison records only; these tests do not measure model quality."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pilot.quality_eval.compare import compare_records
from pilot.quality_eval.records import digest, save_record

from tests.test_quality_eval_records import synthetic_full_record, synthetic_limited_record


def test_diagnostic_comparison_keeps_artifacts_and_hardware_as_provenance(tmp_path: Path):
    left = synthetic_full_record() | {"classification": "plumbing"}
    right = deepcopy(left)
    right["identity"]["artifact_sha256"] = "5" * 64
    right["identity_sha256"] = digest(right["identity"])
    right["provenance"]["hardware"] = "different synthetic hardware; no speed evidence"
    paths = [tmp_path / "left.json", tmp_path / "right.json"]
    for path, record in zip(paths, (left, right), strict=True):
        save_record(path, record)
    before = [path.read_bytes() for path in paths]
    report = compare_records(*paths)
    assert report["status"] == "COMPARABLE_PLUMBING"
    assert all(check["match"] for check in report["checks"])
    assert [row["metric"] for row in report["per_task"]] == ["acc", "acc_norm"]
    assert report["per_task"][0]["left"] == {"value": 1.0, "correct": 1, "samples": 1}
    assert report["provenance"][0]["identity"]["artifact_sha256"] == "d" * 64
    assert report["provenance"][1]["identity"]["artifact_sha256"] == "5" * 64
    assert report["provenance"][0]["hardware"] != report["provenance"][1]["hardware"]
    assert any("plumbing" in text for text in report["limitations"])
    assert [path.read_bytes() for path in paths] == before


def test_comparison_withholds_metrics_for_each_changed_protocol_field(tmp_path: Path):
    record = synthetic_full_record()
    left = tmp_path / "left.json"
    save_record(left, record)
    for field, subfield, value in (
        ("suite", "sha256", "5" * 64),
        ("dataset", "revision", "5" * 40),
        ("samples", "prompt_hash", "5" * 64),
        ("evaluator", "context", 4096),
        ("evaluator", "seeds", [False, 1234, 1234, 1234]),
        ("runtime", "server_defaults_sha256", "5" * 64),
        ("runtime", "backend_flags", ["--parallel", "2"]),
    ):
        changed = deepcopy(record)
        if field == "samples":
            changed["identity"][field][0][subfield] = value
            task = changed["identity"]["suite"]["name"]
            changed["result"]["samples"][task][0][subfield] = value
        else:
            changed["identity"][field][subfield] = value
        if field == "suite":
            changed["identity"]["evaluator"]["suite_sha256"] = value
        changed["identity_sha256"] = digest(changed["identity"])
        right = tmp_path / f"{field}-{subfield}.json"
        save_record(right, changed)
        report = compare_records(left, right)
        assert report["status"] == "NOT_COMPARABLE" and report["per_task"] == []
        assert {"field": field, "match": False} in report["checks"]
        assert field + " differs; comparison withheld" in report["reasons"]


def test_comparison_fails_closed_on_corrupt_partial_unknown_or_inconsistent_results(tmp_path: Path):
    record = synthetic_full_record()
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    save_record(left, record)
    for mutation in ("checksum", "partial", "unknown", "aggregate"):
        changed = deepcopy(record)
        if mutation == "checksum":
            changed["provenance"]["hardware"] = "modified without checksum update"
        elif mutation == "partial":
            changed["status"] = "partial"
        elif mutation == "unknown":
            del changed["identity"]["runtime"]["server_defaults_sha256"]
        else:
            task = changed["identity"]["suite"]["name"]
            changed["result"]["results"][task]["acc,none"] = 0
        checksum = digest(record if mutation == "checksum" else changed)
        right.write_text(json.dumps({"record": changed, "record_sha256": checksum}))
        with pytest.raises((ValueError, KeyError)):
            compare_records(left, right)


def test_fixed_larger_subset_is_diagnostic_not_a_full_benchmark(tmp_path: Path):
    record = synthetic_limited_record()
    paths = [tmp_path / "left.json", tmp_path / "right.json"]
    for path in paths:
        save_record(path, record)
    report = compare_records(*paths)
    assert report["status"] == "COMPARABLE_LIMITED"
    assert all(check["match"] for check in report["checks"])
    assert report["per_task"][0]["left"]["samples"] == 100
    assert any("not a full benchmark" in text for text in report["limitations"])
    smoke = synthetic_full_record() | {"classification": "plumbing"}
    smoke_path = tmp_path / "smoke.json"
    save_record(smoke_path, smoke)
    assert compare_records(paths[0], smoke_path)["status"] == "NOT_COMPARABLE"
