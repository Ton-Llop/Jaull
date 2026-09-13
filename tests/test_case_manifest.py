"""The case manifest and the verdict on it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from jaull.cases.validation import CaseValidationService, identity_for_experiment
from jaull.domain.artifacts import (
    ArtifactIdentity,
    ModelArtifact,
    artifact_identity_matches,
)
from jaull.domain.benchmarks import BenchmarkEnvironment, BenchmarkRecord
from jaull.domain.cases import (
    CaseConsistencyStatus,
    EvidenceFileReference,
    EvidenceFileRole,
    ExperimentalCaseManifest,
)
from jaull.domain.experiments import (
    ExperimentBackendTrace,
    ExperimentEnvironment,
    ExperimentRecord,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeName
from jaull.experiments.errors import ExperimentRecordNotFoundError
from tests._case_fixtures import artifact, benchmark_record, experiment_record, hardware


def _manifest(
    experiment: ExperimentRecord,
    benchmarks: list[BenchmarkRecord] | None = None,
    *,
    evidence: tuple[EvidenceFileReference, ...] = (),
) -> ExperimentalCaseManifest:
    return ExperimentalCaseManifest(
        identity=identity_for_experiment(experiment, label="qwen-2060"),
        experiment_record_id=experiment.identity.experiment_id,
        benchmark_record_ids=tuple(
            record.identity.benchmark_id for record in (benchmarks or [])
        ),
        evidence_files=evidence,
    )


def _service(
    experiment: ExperimentRecord | None,
    benchmarks: list[BenchmarkRecord] | None = None,
    *,
    evidence_root: Path | None = None,
) -> CaseValidationService:
    by_id = {record.identity.benchmark_id: record for record in (benchmarks or [])}

    def load_experiment(experiment_id: str) -> ExperimentRecord:
        if experiment is None or experiment_id != experiment.identity.experiment_id:
            raise ExperimentRecordNotFoundError(
                f"Experiment record not found: {experiment_id}."
            )
        return experiment

    def load_benchmark(benchmark_id: str) -> BenchmarkRecord:
        if benchmark_id not in by_id:
            raise ExperimentRecordNotFoundError(
                f"Benchmark record not found: {benchmark_id}."
            )
        return by_id[benchmark_id]

    return CaseValidationService(
        load_experiment=load_experiment,
        load_benchmark=load_benchmark,
        evidence_root=evidence_root or Path(),
    )


def test_a_case_groups_one_experiment_and_several_benchmarks() -> None:
    experiment = experiment_record()
    benchmarks = [benchmark_record(), benchmark_record(), benchmark_record()]
    manifest = _manifest(experiment, benchmarks)

    result = _service(experiment, benchmarks).validate(manifest)

    assert manifest.experiment_record_id == experiment.identity.experiment_id
    assert len(manifest.benchmark_record_ids) == 3
    assert result.status is CaseConsistencyStatus.PARTIAL
    checks = {check.name: check.status for check in result.checks}
    assert checks["records_exist"] is CaseConsistencyStatus.VALID
    assert checks["machine_identity"] is CaseConsistencyStatus.VALID
    assert checks["artifact_identity"] is CaseConsistencyStatus.VALID


def test_manifest_round_trips_through_json_unchanged() -> None:
    manifest = _manifest(experiment_record(), [benchmark_record()])

    restored = ExperimentalCaseManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest


def test_benchmark_ids_are_ordered_deterministically() -> None:
    experiment = experiment_record()
    ids = ("bench-c", "bench-a", "bench-b")

    manifest = ExperimentalCaseManifest(
        identity=identity_for_experiment(experiment),
        experiment_record_id=experiment.identity.experiment_id,
        benchmark_record_ids=ids,
    )
    reversed_manifest = ExperimentalCaseManifest(
        identity=manifest.identity,
        experiment_record_id=experiment.identity.experiment_id,
        benchmark_record_ids=tuple(reversed(ids)),
    )

    assert manifest.benchmark_record_ids == ("bench-a", "bench-b", "bench-c")
    assert manifest.benchmark_record_ids == reversed_manifest.benchmark_record_ids


def test_a_repeated_benchmark_id_is_rejected() -> None:
    experiment = experiment_record()

    with pytest.raises(ValidationError, match="must not repeat"):
        ExperimentalCaseManifest(
            identity=identity_for_experiment(experiment),
            experiment_record_id=experiment.identity.experiment_id,
            benchmark_record_ids=("bench-a", "bench-a"),
        )


def test_validation_does_not_mutate_the_records_it_reads() -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()
    before = (experiment.model_dump_json(), benchmark.model_dump_json())

    _service(experiment, [benchmark]).validate(_manifest(experiment, [benchmark]))

    assert (experiment.model_dump_json(), benchmark.model_dump_json()) == before


def test_a_different_artifact_makes_the_case_inconsistent() -> None:
    experiment = experiment_record()
    other = benchmark_record(model_artifact=artifact(quantization="Q5_K_M"))
    manifest = _manifest(experiment, [other])

    result = _service(experiment, [other]).validate(manifest)

    assert result.status is CaseConsistencyStatus.INCONSISTENT
    assert any("different artifact" in reason for reason in result.reasons)


def test_a_different_machine_makes_the_case_inconsistent() -> None:
    experiment = experiment_record()
    elsewhere = benchmark_record(profile=hardware(gpu_name="NVIDIA GeForce RTX 4060"))
    manifest = _manifest(experiment, [elsewhere])

    result = _service(experiment, [elsewhere]).validate(manifest)

    assert result.status is CaseConsistencyStatus.INCONSISTENT
    assert any("another machine" in reason for reason in result.reasons)


def test_the_same_gpu_with_different_free_memory_is_a_warning_not_an_error() -> None:
    """Free memory moves between runs. That is a condition, not a contradiction."""
    experiment = experiment_record()
    later = benchmark_record(
        profile=hardware(ram_available_bytes=3 * 1024**3, vram_available_bytes=2 * 1024**3)
    )
    manifest = _manifest(experiment, [later])

    result = _service(experiment, [later]).validate(manifest)

    assert result.status is CaseConsistencyStatus.PARTIAL
    checks = {check.name: check.status for check in result.checks}
    assert checks["machine_identity"] is CaseConsistencyStatus.VALID
    assert checks["execution_conditions"] is CaseConsistencyStatus.VALID
    assert any("available RAM differed" in warning for warning in result.warnings)
    assert any("available VRAM" in warning for warning in result.warnings)
    assert not any("available RAM" in reason for reason in result.reasons)


def test_a_missing_digest_is_partial_rather_than_false_certainty() -> None:
    experiment = experiment_record(model_artifact=artifact(sha256=None))
    benchmark = benchmark_record(model_artifact=artifact(sha256=None))
    manifest = _manifest(experiment, [benchmark])

    result = _service(experiment, [benchmark]).validate(manifest)

    hash_check = next(check for check in result.checks if check.name == "artifact_hash")
    assert hash_check.status is CaseConsistencyStatus.PARTIAL
    assert hash_check.detail == "unavailable"
    assert any("not cryptographically verified" in reason for reason in result.reasons)


def test_conflicting_digests_are_inconsistent() -> None:
    experiment = experiment_record()
    benchmark = benchmark_record(model_artifact=artifact(sha256="b" * 64))
    manifest = _manifest(experiment, [benchmark])

    result = _service(experiment, [benchmark]).validate(manifest)

    assert result.status is CaseConsistencyStatus.INCONSISTENT
    assert any("different artifact digests" in reason for reason in result.reasons)


def test_a_missing_record_is_reported_not_raised() -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()
    manifest = _manifest(experiment, [benchmark])

    # The benchmark is named by the manifest but absent from the store.
    result = _service(experiment, []).validate(manifest)

    assert result.status is CaseConsistencyStatus.INCONSISTENT
    assert any(
        benchmark.identity.benchmark_id in reason and "could not be loaded" in reason
        for reason in result.reasons
    )


def test_a_missing_experiment_stops_short_without_raising() -> None:
    experiment = experiment_record()
    manifest = _manifest(experiment)

    result = _service(None).validate(manifest)

    assert result.status is CaseConsistencyStatus.INCONSISTENT
    assert any("experiment record" in reason for reason in result.reasons)
    assert {check.name for check in result.checks} == {
        "records_exist",
        "evidence_files",
    }


def test_validation_never_touches_hardware_hugging_face_or_a_runtime() -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("validation must stay offline")

    service = CaseValidationService(
        load_experiment=lambda _: experiment,
        load_benchmark=lambda _: benchmark,
    )
    # Nothing here is patched: the service takes no client, no detector and no
    # locator, so there is nothing it could call. The absence is the assertion.
    assert not hasattr(service, "hf_client")
    assert not hasattr(service, "detect_hardware")
    assert not hasattr(service, "runtime_locator")
    del explode

    assert service.validate(_manifest(experiment, [benchmark])).checks


def test_hfa_blocks_and_llama_cpp_units_never_meet() -> None:
    """The manifest links records; it does not reconcile their layer counts."""
    experiment = experiment_record()
    benchmark = benchmark_record()

    result = _service(experiment, [benchmark]).validate(_manifest(experiment, [benchmark]))

    rendered = result.model_dump_json()
    assert "gpu_transformer_blocks" not in rendered
    assert "gpu_layers" not in rendered


def test_missing_benchmark_context_is_explicitly_partial() -> None:
    """Historical llama-bench records cannot prove their workload scenario."""
    experiment = experiment_record(context_length=8192)
    benchmark = benchmark_record()

    result = _service(experiment, [benchmark]).validate(_manifest(experiment, [benchmark]))

    context = next(check for check in result.checks if check.name == "context")
    assert context.status is CaseConsistencyStatus.PARTIAL
    assert "8192" in (context.detail or "")
    assert "no declared context" in (context.detail or "")
    assert not any("ctx-size" in reason for reason in result.reasons)
    assert not any("ctx-size" in warning for warning in result.warnings)


def test_matching_declared_benchmark_context_is_valid_but_not_runner_enforced() -> None:
    experiment = experiment_record(context_length=8192)
    benchmark = benchmark_record(context_length=8192)

    result = _service(experiment, [benchmark]).validate(_manifest(experiment, [benchmark]))

    context = next(check for check in result.checks if check.name == "context")
    assert context.status is CaseConsistencyStatus.VALID
    assert "8192" in (context.detail or "")
    assert "not runner-enforced" in (context.detail or "")


def test_missing_provenance_is_named_rather_than_glossed_over() -> None:
    experiment = experiment_record()
    manifest = _manifest(experiment)

    result = _service(experiment).validate(manifest)

    assert experiment.environment.git_commit is None
    provenance = next(check for check in result.checks if check.name == "provenance")
    assert provenance.status is CaseConsistencyStatus.PARTIAL
    assert any("git commit" in reason for reason in result.reasons)
    assert any("command it executed" in reason for reason in result.reasons)
    assert any("backend" in reason for reason in result.reasons)


def test_complete_provenance_and_equivalent_cli_bench_build_are_valid() -> None:
    experiment = experiment_record().model_copy(
        update={
            "environment": ExperimentEnvironment.capture(git_commit="abc123"),
            "backend_trace": ExperimentBackendTrace(
                observed_backend=ComputeBackend.CUDA,
                observed_source="llama.cpp runtime output",
                executed_command=("llama-cli", "--model", "model.gguf"),
            ),
        }
    )
    benchmark = benchmark_record(
        version_text="build: 689e227db (10357)",
        context_length=4096,
    ).model_copy(
        update={"environment": BenchmarkEnvironment.capture(git_commit="abc123")}
    )

    result = _service(experiment, [benchmark]).validate(_manifest(experiment, [benchmark]))

    checks = {check.name: check.status for check in result.checks}
    assert checks["runtime_build"] is CaseConsistencyStatus.VALID
    assert checks["context"] is CaseConsistencyStatus.VALID
    assert checks["provenance"] is CaseConsistencyStatus.VALID


def test_a_different_runtime_family_is_inconsistent() -> None:
    experiment = experiment_record()
    from tests._case_fixtures import runtime as make_runtime

    other = benchmark_record(recommendation=make_runtime(name=RuntimeName.TRANSFORMERS))
    manifest = _manifest(experiment, [other])

    result = _service(experiment, [other]).validate(manifest)

    assert result.status is CaseConsistencyStatus.INCONSISTENT
    assert any("used runtime" in reason for reason in result.reasons)


def test_an_unavailable_runtime_build_is_partial() -> None:
    experiment = experiment_record(version_text=None)
    benchmark = benchmark_record(version_text=None)
    manifest = _manifest(experiment, [benchmark])

    result = _service(experiment, [benchmark]).validate(manifest)

    build = next(check for check in result.checks if check.name == "runtime_build")
    assert build.status is CaseConsistencyStatus.PARTIAL
    assert any("runtime build" in reason for reason in result.reasons)


# ----------------------------------------------------------------------
# Evidence file references
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/report.json",
        "C:/windows/report.json",
        "validation\\report.json",
        "../escape.json",
        "validation/../../escape.json",
        "   ",
    ],
)
def test_evidence_paths_must_stay_relative_and_inside_their_root(path: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceFileReference(path=path)


def test_a_missing_evidence_file_is_partial_not_a_crash(tmp_path: Path) -> None:
    experiment = experiment_record()
    manifest = _manifest(
        experiment,
        evidence=(EvidenceFileReference(path="validation/report.json"),),
    )

    result = _service(experiment, evidence_root=tmp_path).validate(manifest)

    evidence = next(check for check in result.checks if check.name == "evidence_files")
    assert evidence.status is CaseConsistencyStatus.PARTIAL
    assert any("is missing" in reason for reason in result.reasons)


def test_an_evidence_digest_that_does_not_match_is_reported(tmp_path: Path) -> None:
    (tmp_path / "validation").mkdir()
    report = tmp_path / "validation" / "report.json"
    report.write_text("{}", encoding="utf-8")
    experiment = experiment_record()
    manifest = _manifest(
        experiment,
        evidence=(
            EvidenceFileReference(
                path="validation/report.json",
                role=EvidenceFileRole.REPORT,
                sha256="0" * 64,
            ),
        ),
    )

    result = _service(experiment, evidence_root=tmp_path).validate(manifest)

    assert any("does not match its recorded digest" in r for r in result.reasons)


def test_a_matching_evidence_digest_passes(tmp_path: Path) -> None:
    (tmp_path / "validation").mkdir()
    report = tmp_path / "validation" / "report.json"
    report.write_bytes(b"{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    experiment = experiment_record()
    manifest = _manifest(
        experiment,
        evidence=(
            EvidenceFileReference(
                path="validation/report.json",
                role=EvidenceFileRole.REPORT,
                sha256=digest,
            ),
        ),
    )

    result = _service(experiment, evidence_root=tmp_path).validate(manifest)

    evidence = next(check for check in result.checks if check.name == "evidence_files")
    assert evidence.status is CaseConsistencyStatus.VALID


# ----------------------------------------------------------------------
# The extracted artifact-identity predicate
# ----------------------------------------------------------------------


def test_artifact_identity_ignores_machine_local_state() -> None:
    here = artifact()
    there = here.model_copy(
        update={"local_path": Path("/elsewhere.gguf"), "is_downloaded": False}
    )

    assert artifact_identity_matches(here, there)
    assert ArtifactIdentity.of(here) == ArtifactIdentity.of(there)


def test_artifact_identity_treats_absent_corroboration_as_no_conflict() -> None:
    known = artifact()
    unknown = known.model_copy(update={"sha256": None, "size_bytes": None})

    assert artifact_identity_matches(known, unknown)


def test_artifact_identity_rejects_two_known_and_different_digests() -> None:
    known = artifact()
    other = known.model_copy(update={"sha256": "b" * 64})

    assert not artifact_identity_matches(known, other)


def test_artifact_identity_rejects_a_different_quantization() -> None:
    assert not artifact_identity_matches(
        artifact(quantization="Q4_K_M"), artifact(quantization="Q5_K_M")
    )


def test_artifact_identity_of_accepts_a_bare_reference() -> None:
    bare = ModelArtifact(
        repo_id="owner/repo", revision="main", filename="m.gguf", format="gguf"
    )

    assert ArtifactIdentity.of(bare).quantization is None
