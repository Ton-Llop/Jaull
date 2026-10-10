"""Offline checks for the product-side quality contract and store.

Every record here is synthetic. None of it measures a model, and the point of
most of these tests is that the store refuses to pretend otherwise.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jaull.advisor.service import AdvisorService
from jaull.bootstrap.container import ServiceContainer
from jaull.evaluation.quality_records import (
    describe_record,
    digest,
    is_reusable_evidence,
    placement_of,
)
from jaull.evaluation.quality_storage import (
    InvalidQualityIdError,
    QualityEvidenceStore,
    QualityRecordNotFoundError,
    QualityStoreError,
)
from tests.test_quality_eval_records import synthetic_full_record, synthetic_limited_record


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )


def test_a_record_round_trips_under_its_own_identity_digest(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    path = store.save(record)
    # The filename is the identity, not the model name: the same weights under a
    # different placement are a different row.
    assert path.name == f"{record['identity_sha256']}.json"
    assert store.list_ids() == [record["identity_sha256"]]
    assert store.load(record["identity_sha256"]) == record
    assert store.exists(record["identity_sha256"])


def test_repeated_runs_keep_distinct_records_without_overwriting(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    first = store.save(record)
    assert store.save(deepcopy(record)) == first

    repeated = deepcopy(record)
    repeated["result"]["date"] = 1790943344.0
    second = store.save(repeated)

    assert second != first
    assert second.stem == f"{record['identity_sha256']}-{digest(repeated)}"
    assert store.save(deepcopy(repeated)) == second
    assert store.load(first.stem) == record
    assert store.load(second.stem) == repeated
    assert store.list_ids() == sorted([first.stem, second.stem])
    # Same answers, only the run date moved: one result, not a choice.
    assert store.lookup(record["identity"]) == record["result"]

    differing_measurement = deepcopy(record)
    task = record["identity"]["suite"]["name"]
    differing_measurement["result"]["results"][task]["acc,none"] = 0.0
    differing_measurement["result"]["samples"][task][0]["acc"] = 0
    third = store.save(differing_measurement)
    assert third not in (first, second)
    assert store.load(third.stem) == differing_measurement
    assert len(store.evidence()) == 3
    assert store.lookup(record["identity"]) is None  # Disagreeing repeats: no result.


def test_concurrent_first_run_publication_falls_back_to_a_distinct_record_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jaull.evaluation import quality_storage

    store = QualityEvidenceStore(tmp_path)
    first = synthetic_full_record()
    second = deepcopy(first)
    second["result"]["date"] = 1790943344.0
    original_save = quality_storage.save_record

    def raced_save(path: Path, record: dict[str, object]) -> None:
        if path.stem == first["identity_sha256"]:
            original_save(path, first)
            raise FileExistsError(path)
        original_save(path, record)

    monkeypatch.setattr(quality_storage, "save_record", raced_save)
    second_path = store.save(second)

    assert second_path.stem == f"{first['identity_sha256']}-{digest(second)}"
    assert store.load(first["identity_sha256"]) == first
    assert store.load(second_path.stem) == second


def test_legacy_truncated_record_does_not_block_a_new_run(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    primary = store.path_for(record["identity_sha256"])
    primary.write_text("{", encoding="utf-8")

    saved = store.save(record)

    assert saved != primary
    assert primary.read_text(encoding="utf-8") == "{"
    assert store.load(saved.stem) == record
    assert store.lookup(record["identity"]) == record["result"]


def test_atomic_publication_keeps_existing_record_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jaull.evaluation import quality_records

    record = synthetic_full_record()
    path = tmp_path / "atomic.json"

    def fail_publication(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(quality_records.os, "link", fail_publication)
    with pytest.raises(OSError, match="interrupted publication"):
        quality_records.save_record(path, record)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_interrupted_json_write_never_publishes_a_partial_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jaull.evaluation import quality_records

    def partial_dump(payload: object, handle: object, **kwargs: object) -> None:
        handle.write("{")  # type: ignore[attr-defined]
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(quality_records.json, "dump", partial_dump)
    path = tmp_path / "partial.json"
    with pytest.raises(OSError, match="interrupted write"):
        quality_records.save_record(path, synthetic_full_record())
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_never_replaces_an_existing_record(tmp_path: Path) -> None:
    from jaull.evaluation.quality_records import save_record

    record = synthetic_full_record()
    path = tmp_path / "immutable.json"
    save_record(path, record)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        save_record(path, record)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_a_record_whose_digest_does_not_match_its_identity_is_refused(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    record["identity"]["artifact_sha256"] = "9" * 64
    with pytest.raises(QualityStoreError, match="does not match its identity"):
        store.save(record)


@pytest.mark.parametrize(
    "identity_sha256",
    ["", "../escape", "a" * 63, "A" * 64, "g" * 64, "a" * 64 + "/x", "."],
)
def test_an_identity_that_is_not_a_digest_never_becomes_a_path(
    tmp_path: Path, identity_sha256: str
) -> None:
    store = QualityEvidenceStore(tmp_path)
    with pytest.raises(InvalidQualityIdError):
        store.path_for(identity_sha256)


def test_unreadable_evidence_is_skipped_rather_than_guessed_at(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    good = synthetic_full_record()
    store.save(good)
    (tmp_path / f"{'b' * 64}.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "not-an-identity.json").write_text("{}", encoding="utf-8")

    # Listing shows the digest-shaped file; reading it fails loudly; the sweep
    # over every record drops it and keeps the readable one.
    assert store.list_ids() == sorted([good["identity_sha256"], "b" * 64])
    with pytest.raises(QualityStoreError):
        store.load("b" * 64)
    assert store.records() == [good]
    with pytest.raises(QualityRecordNotFoundError):
        store.load("c" * 64)


@pytest.mark.parametrize("image_id", [123, ["invalid"], {"invalid": True}])
def test_malformed_image_id_does_not_break_the_evidence_list(
    tmp_path: Path, image_id: object
) -> None:
    store = QualityEvidenceStore(tmp_path)
    good = synthetic_full_record()
    store.save(good)
    malformed = deepcopy(good)
    malformed["identity"]["runtime"]["image_id"] = image_id
    malformed["identity_sha256"] = digest(malformed["identity"])
    path = store.path_for(malformed["identity_sha256"])
    contents = json.dumps({"record": malformed, "record_sha256": digest(malformed)})
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(QualityStoreError, match="invalid"):
        store.load(malformed["identity_sha256"])
    assert store.records() == [good]
    assert [item.identity_sha256 for item in store.evidence()] == [good["identity_sha256"]]
    assert store.lookup(malformed["identity"]) is None
    assert path.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("unknown", [float("nan"), object()])
def test_lookup_of_a_non_serializable_identity_is_a_miss(
    tmp_path: Path, unknown: object
) -> None:
    store = QualityEvidenceStore(tmp_path / "quality")
    identity = synthetic_full_record()["identity"]
    identity["evaluator"]["seeds"][0] = unknown
    assert store.lookup(identity) is None
    assert not store.root.exists()


def test_lookup_serves_full_evidence_and_refuses_a_diagnostic(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    store.save(record)
    assert store.lookup(record["identity"]) == record["result"]

    # A limited run is exploration. Same protocol, same artifact, still not
    # something a ranking may consume.
    limited_store = QualityEvidenceStore(tmp_path / "limited")
    limited = synthetic_limited_record()
    limited_store.save(limited)
    assert not is_reusable_evidence(limited)
    assert limited_store.lookup(limited["identity"]) is None


def test_lookup_refuses_an_identity_that_is_not_the_one_stored(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    store.save(record)
    other = deepcopy(record["identity"])
    other["artifact_sha256"] = "7" * 64
    assert digest(other) != record["identity_sha256"]
    assert store.lookup(other) is None


def test_lookup_rejects_record_with_incorrect_record_digest_suffix(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    invalid_id = f"{record['identity_sha256']}-{'0' * 64}"
    invalid_path = store.path_for(invalid_id)
    invalid_path.write_text(
        json.dumps({"record": record, "record_sha256": digest(record)}), encoding="utf-8",
    )
    before = invalid_path.read_bytes()

    assert store.lookup(record["identity"]) is None
    assert store.records() == []
    assert invalid_path.read_bytes() == before


def test_lookup_ignores_bad_suffix_when_one_valid_record_exists(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    valid_path = store.save(record)
    invalid_id = f"{record['identity_sha256']}-{'0' * 64}"
    invalid_path = store.path_for(invalid_id)
    invalid_path.write_bytes(valid_path.read_bytes())

    assert store.lookup(record["identity"]) == record["result"]
    assert store.records() == [record]


def test_lookup_reuses_a_valid_repeat_when_primary_record_is_unreadable(
    tmp_path: Path,
) -> None:
    store = QualityEvidenceStore(tmp_path)
    record = synthetic_full_record()
    primary = store.save(record)
    repeated = deepcopy(record)
    repeated["result"]["date"] = 1790943344.0
    repeated_path = store.save(repeated)
    primary.write_text("{", encoding="utf-8")

    assert store.lookup(record["identity"]) == repeated["result"]
    assert primary.read_text(encoding="utf-8") == "{"
    assert store.load(repeated_path.stem) == repeated


def test_the_display_projection_carries_the_grade_and_the_limits() -> None:
    limited = synthetic_limited_record()
    limited["identity"]["runtime"]["backend_flags"] = [
        "--parallel", "1", "--n-gpu-layers", "0", "--threads", "4",
    ]
    limited["identity_sha256"] = digest(limited["identity"])
    evidence = describe_record(limited)

    assert evidence.reusable is False
    assert evidence.classification == "limited"
    assert evidence.samples_used == 100
    assert evidence.samples_available == 10042
    # The placement is readable, because evidence measured on CPU does not
    # describe a run that would offload to the GPU.
    assert evidence.placement == {"--n-gpu-layers": "0"}
    assert any("Limited run" in line for line in evidence.limitations)
    assert any("placement" in line for line in evidence.limitations)
    assert any("100 of 10042" in line for line in evidence.limitations)


def test_no_date_is_invented_when_the_harness_did_not_record_one() -> None:
    record = synthetic_full_record()
    record["result"].pop("date", None)
    assert describe_record(record).evaluated_at is None

    record["result"]["date"] = 1790943344.0
    stamped = describe_record(record).evaluated_at
    assert stamped == datetime.fromtimestamp(1790943344.0, tz=UTC)

    for unusable in ("2026-10-02", None, float("inf")):
        record["result"]["date"] = unusable
        assert describe_record(record).evaluated_at is None


def test_placement_reads_only_the_flags_that_decide_where_layers_run() -> None:
    record = synthetic_full_record()
    record["identity"]["runtime"]["backend_flags"] = ["--n-gpu-layers", "-1", "--device", "CUDA0"]
    assert placement_of(record) == {"--n-gpu-layers": "-1", "--device": "CUDA0"}

    # A trailing flag with no value is not a placement claim.
    record["identity"]["runtime"]["backend_flags"] = ["--threads", "4", "--n-gpu-layers"]
    assert placement_of(record) == {}


@pytest.mark.parametrize("flag,initial,final", [
    ("--n-gpu-layers", "-1", "0"),
    ("--device", "CUDA0", "none"),
])
def test_placement_uses_the_last_value_like_llama_cpp(
    flag: str, initial: str, final: str
) -> None:
    record = synthetic_full_record()
    flags = [flag, initial, "--threads", "4", flag, final]
    record["identity"]["runtime"]["backend_flags"] = flags
    record["identity_sha256"] = digest(record["identity"])
    assert describe_record(record).placement == {flag: final}
    assert record["identity"]["runtime"]["backend_flags"] == flags

    # A malformed final occurrence must not inherit the earlier placement.
    record["identity"]["runtime"]["backend_flags"] = flags[:-1]
    assert placement_of(record) == {}


def test_the_advisor_reads_evidence_through_the_injected_store(tmp_path: Path) -> None:
    store = QualityEvidenceStore(tmp_path)
    advisor = AdvisorService(services=_services(), quality_store=store)
    assert advisor.list_quality_ids() == []
    assert advisor.quality_evidence() == []

    record = synthetic_full_record()
    advisor.save_quality_record(record)
    assert advisor.list_quality_ids() == [record["identity_sha256"]]
    assert advisor.load_quality_record(record["identity_sha256"]) == record
    assert advisor.lookup_quality(record["identity"]) == record["result"]
    assert [item.suite for item in advisor.quality_evidence()] == [
        record["identity"]["suite"]["name"]
    ]
