"""The offline tg128 predictor: its arithmetic and that scoring never reuses calibration."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import timedelta

import pytest
from scripts.speed_predictor import (
    Point,
    _write_new,
    calibration_cohort,
    cmd_score,
    comparable_points,
    fit_residual,
    leave_one_out,
    predict,
    relative_error,
    score_rows,
)

BW = 336e9  # RTX 2060

# The three llama_bench_v2 plans measured on the RTX 2060 (median tg128).
CALIBRATION = [
    Point("a" * 64, "qwen2.5-0.5b-instruct-q5_k_m.gguf", 522186592, 246.1, ("b1",)),
    Point("b" * 64, "LFM2.5-1.2B-Instruct-Q5_K_M.gguf", 843354944, 230.5, ("b2",)),
    Point("c" * 64, "qwen2.5-1.5b-instruct-q5_k_m.gguf", 1285494304, 150.7, ("b3",)),
]


def test_the_residual_reproduces_a_single_calibration_point_exactly() -> None:
    point = CALIBRATION[0]
    residual = fit_residual([point], BW)
    assert predict(point.size_bytes, BW, residual)["additive"] == pytest.approx(246.1)


def test_reference_and_llmfit_follow_bandwidth_over_bytes() -> None:
    predicted = predict(522186592, BW, residual=0.0)
    assert predicted["reference"] == pytest.approx(643.4, abs=0.1)
    assert predicted["llmfit"] == pytest.approx(predicted["reference"] * 0.55)
    assert predicted["additive"] == pytest.approx(predicted["reference"])


def test_leave_one_out_never_fits_the_point_it_predicts() -> None:
    for point, predicted in leave_one_out(CALIBRATION, BW):
        others = [p for p in CALIBRATION if p is not point]
        assert predicted == pytest.approx(
            predict(point.size_bytes, BW, fit_residual(others, BW))["additive"])


def test_score_uses_the_frozen_prediction_and_reports_missing_targets() -> None:
    frozen = {"targets": [
        {"filename": "x.gguf", "sha256": "x" * 64, "size_bytes": 1,
         "predicted": {"additive": 110.0}},
        {"filename": "y.gguf", "sha256": "y" * 64, "size_bytes": 1,
         "predicted": {"additive": 50.0}},
    ]}
    rows = score_rows(frozen, [Point("x" * 64, "x.gguf", 1, 100.0, ("b1", "b2", "b3"))])
    assert rows[0]["error"] == {"additive": pytest.approx(0.10)}
    assert rows[0]["calls"] == 3
    assert rows[1]["measured"] is None and rows[1]["error"] is None
    assert relative_error(90.0, 100.0) == pytest.approx(-0.10)


def _record():
    from tests.test_fastest_v2_applicability import _cuda_case

    return _cuda_case()[1]


def _select(records, **kwargs):
    record = records[0]
    return comparable_points(records, record.hardware.gpus[0].name,
                             record.observation.protocol.build_number, **kwargs)


def test_record_checks_are_shared_with_the_product():
    record = _record()
    assert len(_select([record])) == 1
    command = list(record.observation.command)
    del command[command.index("-fa"):command.index("-fa") + 2]
    unaudited = record.model_copy(update={
        "observation": record.observation.model_copy(update={"command": tuple(command)}),
    })
    assert _select([unaudited]) == []


@pytest.mark.parametrize("change", ["kernel", "cpu", "driver", "build_commit", "device"])
def test_other_environments_never_enter_the_same_cohort(change):
    from jaull.recommendation.local_evidence import speed_cohort

    original = _record()
    if change == "build_commit":
        other = original.model_copy(update={"observation": original.observation.model_copy(update={
            "protocol": original.observation.protocol.model_copy(update={"build_commit": "a" * 9}),
            "raw_stdout": "build: aaaaaaaaa (1)\n",
        }), "llama_bench_capability": None})
    elif change == "device":
        from tests.test_fastest_v2_applicability import _cuda_case

        other = _cuda_case(device="CUDA1")[1]
    else:
        hardware = original.hardware
        update = {
            "kernel": {"os": "Linux another-kernel"},
            "cpu": {"cpu": hardware.cpu.model_copy(update={"model": "Other CPU"})},
            "driver": {"gpus": [g.model_copy(update={"driver_version": "another-driver"})
                                for g in hardware.gpus]},
        }[change]
        other = original.model_copy(update={"hardware": hardware.model_copy(update=update)})
    other = other.model_copy(update={"identity": other.identity.model_copy(update={
        "benchmark_id": "other-run",
    })})
    assert len(_select([other])) == 1
    with pytest.raises(ValueError, match="Multiple speed cohorts"):
        _select([original, other])
    assert _select([original, other], cohort=speed_cohort(original)) == _select([original])


def test_mislabeled_prefill_is_not_decode_and_invalid_runs_do_not_count():
    from jaull.domain.benchmarks import BenchmarkMeasurementKind

    record = _record()
    wrong = record.model_copy(update={"observation": record.observation.model_copy(update={
        "measurements": [m.model_copy(update={"kind": BenchmarkMeasurementKind.PREFILL})
                         if m.tokens == 128 else m for m in record.observation.measurements],
    })})
    assert _select([wrong]) == []
    failed_exit = record.model_copy(update={"observation": record.observation.model_copy(update={
        "exit_code": 1,
    })})
    assert _select([failed_exit]) == []
    many = record.model_copy(update={"hardware": record.hardware.model_copy(update={
        "gpus": record.hardware.gpus * 2,
    })})
    assert _select([many]) == []


def test_duplicate_records_are_deduplicated_and_conflicting_ids_are_rejected():
    record = _record()
    assert _select([record, record]) == _select([record])
    conflict = record.model_copy(update={"notes": ["different content"]})
    for records in ([record, conflict], [conflict, record]):
        with pytest.raises(ValueError, match="Conflicting benchmark ID"):
            _select(records)


def test_legacy_freeze_recovers_its_original_cohort_without_rewriting():
    from jaull.recommendation.local_evidence import speed_cohort

    record = _record()
    point = _select([record])[0]
    frozen = {"gpu": record.hardware.gpus[0].name, "llama_cpp_build": 1,
              "frozen_at": (record.identity.created_at + timedelta(seconds=1)).isoformat(),
              "calibration": [dataclasses.asdict(point)]}
    before = repr(frozen)
    assert calibration_cohort(frozen, [record]) == speed_cohort(record)
    assert repr(frozen) == before
    with pytest.raises(ValueError, match="missing"):
        calibration_cohort(frozen, [])
    frozen["calibration"][0]["median_tps"] *= 2
    with pytest.raises(ValueError, match="differs"):
        calibration_cohort(frozen, [record])


def test_calibration_cannot_be_taken_after_freezing():
    record = _record()
    frozen = {"gpu": record.hardware.gpus[0].name, "llama_cpp_build": 1,
              "frozen_at": record.identity.created_at.isoformat(),
              "calibration": [dataclasses.asdict(_select([record])[0])]}
    with pytest.raises(ValueError, match="predate"):
        calibration_cohort(frozen, [record])


@pytest.mark.parametrize("size,bw,c", [(0, BW, 0), (1, 0, 0),
                                      (1, float("nan"), 0), (1, BW, float("inf")),
                                      (1, BW, -1.0)])
def test_invalid_predictions_fail_instead_of_returning_negative_or_nonfinite_tps(size, bw, c):
    with pytest.raises(ValueError):
        predict(size, bw, c)


@pytest.mark.parametrize("tps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_calibration_points_fail(tps):
    with pytest.raises(ValueError):
        Point("a", "a.gguf", 1, tps, ("run",))


def test_two_calls_are_not_a_complete_validation_and_wrong_cohort_cannot_score():
    point = dataclasses.replace(CALIBRATION[0], benchmark_ids=("r1", "r2"), cohort="A")
    frozen = {"speed_cohort": "A", "targets": [
        {"sha256": point.sha256, "filename": point.filename, "size_bytes": point.size_bytes,
         "predicted": {"additive": 100.0}},
    ]}
    row = score_rows(frozen, [point])[0]
    assert row["status"] == "insufficient_calls" and row["error"] is None
    with pytest.raises(ValueError, match="cohort"):
        score_rows(frozen, [dataclasses.replace(point, cohort="B")])


def test_prediction_and_report_outputs_cannot_overwrite_existing_evidence(tmp_path):
    path = tmp_path / "frozen.json"
    _write_new(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _write_new(path, {"overwritten": True})
    assert path.read_bytes() == original
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        _write_new(invalid, {"not_a_number": float("nan")})
    assert not invalid.exists()


def test_source_commit_comes_from_this_checkout_not_the_callers_directory(tmp_path, monkeypatch):
    from scripts.speed_predictor import _git

    expected = _git("rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)
    assert _git("rev-parse", "HEAD") == expected


def test_no_git_marker_means_unknown_provenance_not_an_ancestor_commit(tmp_path, monkeypatch):
    import scripts.speed_predictor as module

    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / "speed_predictor.py"))
    assert module._git("rev-parse", "HEAD") is None


def test_validation_cannot_score_a_calibration_artifact_or_an_empty_target_set():
    point = CALIBRATION[0]
    with pytest.raises(ValueError, match="No frozen targets"):
        score_rows({"targets": []}, [])
    with pytest.raises(ValueError, match="calibration artifact"):
        score_rows({"calibration": [dataclasses.asdict(point)], "targets": [
            {"sha256": point.sha256},
        ]}, [point])
    with pytest.raises(ValueError, match="distinct artifacts"):
        score_rows({"targets": [{"sha256": point.sha256}] * 2}, [point])


def test_measure_aborts_before_network_or_gpu_if_the_kernel_changed(tmp_path, monkeypatch):
    from scripts.speed_predictor import cmd_measure

    from jaull.advisor.service import AdvisorService

    record = _record()
    frozen = {"gpu": record.hardware.gpus[0].name, "llama_cpp_build": 1,
              "frozen_at": (record.identity.created_at + timedelta(seconds=1)).isoformat(),
              "calibration": [dataclasses.asdict(_select([record])[0])]}

    class OtherKernelAdvisor:
        def scan_hardware(self):
            return record.hardware.model_copy(update={"os": "Linux newer-kernel"})

        def list_benchmark_ids(self):
            return [record.identity.benchmark_id]

        def load_benchmark_record(self, key):
            assert key == record.identity.benchmark_id
            return record

        def inspect_model(self, *args, **kwargs):
            pytest.fail("No network/model inspection is allowed after the environment mismatch")

        def benchmark_plan_speed(self, *args, **kwargs):
            pytest.fail("No GPU work is allowed after the environment mismatch")

    monkeypatch.setattr(AdvisorService, "default", lambda: OtherKernelAdvisor())
    path = tmp_path / "frozen.json"
    _write_new(path, frozen)
    with pytest.raises(ValueError, match="Current environment differs"):
        cmd_measure(argparse.Namespace(frozen=path))


def test_measure_rejects_a_different_sha_even_when_the_filename_matches(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.speed_predictor import cmd_measure

    from jaull.advisor.service import AdvisorService
    from tests._workflow_fixtures import answers

    record = _record()
    frozen = {"gpu": record.hardware.gpus[0].name, "llama_cpp_build": 1,
              "frozen_at": (record.identity.created_at + timedelta(seconds=1)).isoformat(),
              "calibration": [dataclasses.asdict(_select([record])[0])],
              "targets": [{"repo_id": "org/model", "filename": "new.gguf", "sha256": "b" * 64}]}

    class WrongArtifactAdvisor:
        def scan_hardware(self):
            return record.hardware

        def list_benchmark_ids(self):
            return [record.identity.benchmark_id]

        def load_benchmark_record(self, key):
            return record

        def inspect_model(self, repo):
            assert repo == "org/model"
            return SimpleNamespace(classification=SimpleNamespace(gguf_variants=[
                SimpleNamespace(files=[SimpleNamespace(path="new.gguf")], sha256="c" * 64),
            ]))

        def benchmark_plan_speed(self, *args, **kwargs):
            pytest.fail("The different artifact must never reach GPU execution")

    monkeypatch.setattr(AdvisorService, "default", lambda: WrongArtifactAdvisor())
    path, capture = tmp_path / "frozen.json", tmp_path / "capture.json"
    _write_new(path, frozen)
    _write_new(capture, {"input": {"answers": answers().model_dump(mode="json")}})
    with pytest.raises(SystemExit, match="Cannot pin"):
        cmd_measure(argparse.Namespace(frozen=path, capture=capture, calls=3, pause=0))


@pytest.mark.parametrize("changed_kernel", [False, True])
def test_score_requires_same_cohort_even_with_same_gpu_build_and_postfreeze_timestamp(
    tmp_path, monkeypatch, changed_kernel,
):
    import json

    from jaull.advisor.service import AdvisorService

    record = _record()
    point = _select([record])[0]
    frozen = {"gpu": record.hardware.gpus[0].name, "llama_cpp_build": 1,
              "frozen_at": (record.identity.created_at + timedelta(seconds=1)).isoformat(),
              "calibration": [dataclasses.asdict(point)],
              "targets": [{"filename": "new.gguf", "sha256": "b" * 64,
                           "size_bytes": point.size_bytes, "predicted": {"additive": 100.0}}]}
    new_artifact = record.artifact.model_copy(update={"sha256": "b" * 64})
    target = record.model_copy(update={
        "artifact": new_artifact, "request": record.request.model_copy(update={
            "artifact": new_artifact,
        }),
        "hardware": (record.hardware.model_copy(update={"os": "Linux newer-kernel"})
                     if changed_kernel else record.hardware),
        "identity": record.identity.model_copy(update={"benchmark_id": "target-run",
            "created_at": record.identity.created_at + timedelta(seconds=2)}),
    })
    records = [record, *(target.model_copy(update={"identity": target.identity.model_copy(update={
        "benchmark_id": f"target-{index}",
        "created_at": record.identity.created_at + timedelta(seconds=index + 2),
    })}) for index in range(3))]

    class StoreAdvisor:
        def list_benchmark_ids(self):
            return [r.identity.benchmark_id for r in records]

        def load_benchmark_record(self, key):
            return next(r for r in records if r.identity.benchmark_id == key)

    monkeypatch.setattr(AdvisorService, "default", lambda: StoreAdvisor())
    path, output = tmp_path / "frozen.json", tmp_path / "scores.json"
    _write_new(path, frozen)
    before = path.read_bytes()
    if changed_kernel:
        with pytest.raises(SystemExit, match="Validation incomplete"):
            cmd_score(argparse.Namespace(frozen=path, out=output))
    else:
        cmd_score(argparse.Namespace(frozen=path, out=output))
    result = json.loads(output.read_text())
    assert result["complete"] is not changed_kernel
    assert result["rows"][0]["status"] == ("missing" if changed_kernel else "comparable")
    assert (result["rows"][0]["error"] is None) is changed_kernel
    assert path.read_bytes() == before
