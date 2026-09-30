"""Corrupt evidence must produce store errors without modifying its bytes."""

from collections.abc import Callable
from pathlib import Path

import pytest

from jaull.benchmarks.errors import BenchmarkStoreError
from jaull.benchmarks.storage import BenchmarkStore
from jaull.cases.errors import CaseStoreError
from jaull.cases.storage import CaseStore
from jaull.cases.validation import identity_for_experiment
from jaull.domain.cases import ExperimentalCaseManifest
from jaull.experiments.errors import ExperimentStoreError
from jaull.experiments.storage import ExperimentStore
from tests._case_fixtures import benchmark_record, experiment_record


@pytest.mark.parametrize(
    "store_type,error_type,record_id",
    [
        (BenchmarkStore, BenchmarkStoreError, "bench-corrupt"),
        (ExperimentStore, ExperimentStoreError, "exp-corrupt"),
        (CaseStore, CaseStoreError, "case-corrupt"),
    ],
)
def test_invalid_utf8_is_reported_without_rewriting_evidence(
    tmp_path: Path,
    store_type: type[BenchmarkStore | ExperimentStore | CaseStore],
    error_type: type[Exception],
    record_id: str,
) -> None:
    store = store_type(root=tmp_path)
    path = store.path_for(record_id)
    original = b'{"corrupt": "\xff"}'
    path.write_bytes(original)

    with pytest.raises(error_type, match="not UTF-8") as error:
        store.load(record_id)

    assert isinstance(error.value.__cause__, UnicodeDecodeError)
    assert path.read_bytes() == original


@pytest.fixture(params=["benchmark", "experiment", "case", "runtime_log"])
def save_evidence(request: pytest.FixtureRequest) -> tuple[Callable[[Path], Path], type[Exception]]:
    if request.param == "benchmark":
        return lambda root: BenchmarkStore(root).save(benchmark_record()), BenchmarkStoreError
    if request.param == "experiment":
        return lambda root: ExperimentStore(root).save(experiment_record()), ExperimentStoreError
    if request.param == "case":
        manifest = ExperimentalCaseManifest(
            identity=identity_for_experiment(experiment_record()),
            experiment_record_id="exp-test",
        )
        return lambda root: CaseStore(root).save(manifest), CaseStoreError
    return (
        lambda root: ExperimentStore(root).save_runtime_log("exp-test", stdout="log", stderr=""),
        ExperimentStoreError,
    )


def test_store_root_that_is_a_file_produces_store_error(
    tmp_path: Path,
    save_evidence: tuple[Callable[[Path], Path], type[Exception]],
) -> None:
    root = tmp_path / "not-a-directory"
    root.write_bytes(b"existing data")
    save, error_type = save_evidence

    with pytest.raises(error_type, match="Could not save") as error:
        save(root)

    assert isinstance(error.value.__cause__, OSError)
    assert root.read_bytes() == b"existing data"


def test_cleanup_failure_does_not_mask_original_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_evidence: tuple[Callable[[Path], Path], type[Exception]],
) -> None:
    original_error = PermissionError("write denied")

    def deny_open(*args: object, **kwargs: object) -> None:
        raise original_error

    def deny_cleanup(*args: object, **kwargs: object) -> None:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(Path, "open", deny_open)
    monkeypatch.setattr(Path, "unlink", deny_cleanup)
    save, error_type = save_evidence

    with pytest.raises(error_type, match="write denied") as error:
        save(tmp_path)

    assert error.value.__cause__ is original_error


def test_unreadable_existing_runtime_log_is_not_overwritten(tmp_path: Path) -> None:
    store = ExperimentStore(root=tmp_path)
    path = store.path_for("exp-corrupt").with_suffix(".runtime-log")
    original = b"\xff original log"
    path.write_bytes(original)

    with pytest.raises(ExperimentStoreError, match="Could not read runtime log"):
        store.save_runtime_log("exp-corrupt", stdout="new output", stderr="")

    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["experiment", "legacy_experiment", "benchmark", "case"])
def test_record_identity_must_match_requested_id(tmp_path: Path, kind: str) -> None:
    if kind in {"experiment", "legacy_experiment"}:
        store = ExperimentStore(tmp_path)
        record = experiment_record()
        source = store.save(record)
        error_type = ExperimentStoreError
        if kind == "legacy_experiment":
            source.write_text(record.model_dump_json(), encoding="utf-8")
    elif kind == "benchmark":
        store = BenchmarkStore(tmp_path)
        source = store.save(benchmark_record())
        error_type = BenchmarkStoreError
    else:
        store = CaseStore(tmp_path)
        source = store.save(ExperimentalCaseManifest(
            identity=identity_for_experiment(experiment_record()),
            experiment_record_id="exp-test",
        ))
        error_type = CaseStoreError
    path = store.path_for("wrong-id")
    original = source.read_bytes()
    path.write_bytes(original)

    with pytest.raises(error_type, match="identity"):
        store.load("wrong-id")

    assert path.read_bytes() == original
    assert source.read_bytes() == original
