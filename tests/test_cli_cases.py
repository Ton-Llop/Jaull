"""The `jaull experiments case` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from jaull.cases.storage import CaseStore
from jaull.cli.app import app
from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.cases import ExperimentalCaseManifest
from jaull.domain.experiments import ExperimentRecord
from jaull.experiments.errors import ExperimentRecordNotFoundError
from tests._case_fixtures import benchmark_record, experiment_record


class _FakeAdvisor:
    """Only what the case commands reach for."""

    def __init__(
        self,
        tmp_path: Path,
        experiment: ExperimentRecord,
        benchmarks: list[BenchmarkRecord],
    ) -> None:
        self._experiment = experiment
        self._benchmarks = {
            record.identity.benchmark_id: record for record in benchmarks
        }
        self._store = CaseStore(root=tmp_path / "cases")

    # -- record access -------------------------------------------------
    def load_experiment_record(self, experiment_id: str) -> ExperimentRecord:
        if experiment_id != self._experiment.identity.experiment_id:
            raise ExperimentRecordNotFoundError(
                f"Experiment record not found: {experiment_id}."
            )
        return self._experiment

    def load_benchmark_record(self, benchmark_id: str) -> BenchmarkRecord:
        if benchmark_id not in self._benchmarks:
            raise ExperimentRecordNotFoundError(
                f"Benchmark record not found: {benchmark_id}."
            )
        return self._benchmarks[benchmark_id]

    # -- case access ---------------------------------------------------
    def build_case_manifest(self, **kwargs: Any) -> ExperimentalCaseManifest:
        from jaull.cases.validation import identity_for_experiment

        record = self.load_experiment_record(kwargs["experiment_id"])
        return ExperimentalCaseManifest(
            identity=identity_for_experiment(record, label=kwargs.get("label")),
            experiment_record_id=kwargs["experiment_id"],
            benchmark_record_ids=tuple(kwargs.get("benchmark_ids", ())),
            evidence_files=tuple(kwargs.get("evidence_files", ())),
            notes=tuple(kwargs.get("notes", ())),
        )

    def save_case_manifest(self, manifest: ExperimentalCaseManifest) -> Path:
        return self._store.save(manifest)

    def load_case_manifest(self, case_id: str) -> ExperimentalCaseManifest:
        return self._store.load(case_id)

    def list_case_ids(self) -> list[str]:
        return self._store.list_ids()

    def validate_case(self, case_id: str, *, evidence_root: Path | None = None) -> Any:
        from jaull.cases.validation import CaseValidationService

        service = CaseValidationService(
            load_experiment=self.load_experiment_record,
            load_benchmark=self.load_benchmark_record,
            evidence_root=evidence_root or Path(),
        )
        return service.validate(self._store.load(case_id))


def _install(monkeypatch: Any, advisor: _FakeAdvisor) -> None:
    monkeypatch.setattr(
        "jaull.cli.cases.AdvisorService.default", lambda: advisor, raising=False
    )


def test_create_emits_json_with_the_case_and_its_verdict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()
    _install(monkeypatch, _FakeAdvisor(tmp_path, experiment, [benchmark]))

    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "case",
            "create",
            "--experiment",
            experiment.identity.experiment_id,
            "--benchmark",
            benchmark.identity.benchmark_id,
            "--label",
            "qwen-2060",
            "--json",
        ],
        catch_exceptions=False,
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["case"]["experiment_record_id"] == experiment.identity.experiment_id
    assert payload["case"]["identity"]["label"] == "qwen-2060"
    assert payload["validation"]["status"] == "partial"


def test_create_refuses_a_benchmark_id_that_does_not_exist(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiment = experiment_record()
    advisor = _FakeAdvisor(tmp_path, experiment, [])
    _install(monkeypatch, advisor)

    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "case",
            "create",
            "--experiment",
            experiment.identity.experiment_id,
            "--benchmark",
            "bench-typo",
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 3
    assert advisor.list_case_ids() == []


def test_create_refuses_an_unknown_evidence_role(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiment = experiment_record()
    _install(monkeypatch, _FakeAdvisor(tmp_path, experiment, []))

    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "case",
            "create",
            "--experiment",
            experiment.identity.experiment_id,
            "--evidence",
            "validation/report.json:nonsense",
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2


def test_evidence_is_parsed_as_path_and_role(tmp_path: Path, monkeypatch: Any) -> None:
    experiment = experiment_record()
    _install(monkeypatch, _FakeAdvisor(tmp_path, experiment, []))

    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "case",
            "create",
            "--experiment",
            experiment.identity.experiment_id,
            "--evidence",
            "validation/qwen/report.json:report",
            "--evidence",
            "validation/qwen/llama-ngl0.log:runtime_log",
            "--json",
        ],
        catch_exceptions=False,
    )

    files = json.loads(result.stdout)["case"]["evidence_files"]
    assert result.exit_code == 0
    assert [(item["path"], item["role"]) for item in files] == [
        ("validation/qwen/report.json", "report"),
        ("validation/qwen/llama-ngl0.log", "runtime_log"),
    ]


def test_show_and_validate_read_back_a_saved_case(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()
    advisor = _FakeAdvisor(tmp_path, experiment, [benchmark])
    _install(monkeypatch, advisor)
    manifest = advisor.build_case_manifest(
        experiment_id=experiment.identity.experiment_id,
        benchmark_ids=[benchmark.identity.benchmark_id],
    )
    advisor.save_case_manifest(manifest)
    case_id = manifest.identity.case_id

    runner = CliRunner()
    shown = runner.invoke(
        app, ["experiments", "case", "show", case_id, "--json"], catch_exceptions=False
    )
    validated = runner.invoke(
        app,
        ["experiments", "case", "validate", case_id, "--json"],
        catch_exceptions=False,
    )
    listed = runner.invoke(
        app, ["experiments", "case", "list", "--json"], catch_exceptions=False
    )

    assert json.loads(shown.stdout)["case"]["identity"]["case_id"] == case_id
    assert json.loads(validated.stdout)["case_id"] == case_id
    assert json.loads(listed.stdout)["case_ids"] == [case_id]


def test_validating_an_absent_case_exits_three(tmp_path: Path, monkeypatch: Any) -> None:
    _install(monkeypatch, _FakeAdvisor(tmp_path, experiment_record(), []))

    result = CliRunner().invoke(
        app,
        ["experiments", "case", "validate", "case-missing", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 3


def test_the_human_report_names_the_status_and_the_gaps(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiment = experiment_record()
    benchmark = benchmark_record()
    _install(monkeypatch, _FakeAdvisor(tmp_path, experiment, [benchmark]))

    result = CliRunner().invoke(
        app,
        [
            "experiments",
            "case",
            "create",
            "--experiment",
            experiment.identity.experiment_id,
            "--benchmark",
            benchmark.identity.benchmark_id,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "artifact_identity: valid" in result.stdout
    assert "provenance: partial" in result.stdout
    assert "git commit" in result.stdout
    assert "Status:" in result.stdout
    # Historical llama-bench records cannot prove the workload context.
    assert "no declared context" in result.stdout
