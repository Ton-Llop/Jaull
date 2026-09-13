"""Experimental case commands: group evidence, then check that it holds together."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.cases.errors import CaseStoreError
from jaull.domain.cases import (
    CaseConsistencyStatus,
    CaseValidationResult,
    EvidenceFileReference,
    EvidenceFileRole,
    ExperimentalCaseManifest,
)
from jaull.experiments.errors import ExperimentStoreError
from jaull.presentation.console import make_console


@dataclass(frozen=True)
class CreateCaseOptions:
    experiment_id: str
    benchmark_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    label: str | None = None
    note: tuple[str, ...] = ()
    as_json: bool = False


@dataclass(frozen=True)
class CaseOptions:
    as_json: bool = False


@dataclass(frozen=True)
class ExportCaseOptions:
    destination: Path
    evidence_root: Path = Path()
    as_json: bool = False


def run_create_case(
    options: CreateCaseOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()

    try:
        evidence = tuple(_parse_evidence(item) for item in options.evidence)
    except ValueError as exc:
        _print_error(console, exc, options.as_json)
        return 2

    try:
        manifest = resolved.build_case_manifest(
            experiment_id=options.experiment_id,
            benchmark_ids=options.benchmark_ids,
            evidence_files=evidence,
            label=options.label,
            notes=options.note,
        )
    except (ExperimentStoreError, ValueError) as exc:
        _print_error(console, exc, options.as_json)
        return 3

    # Validate before saving so a mistyped benchmark id is refused rather than
    # persisted. Every other shortfall -- a missing digest, absent provenance --
    # is a real property of the evidence and is saved with the case.
    result = _validate(resolved, manifest)
    missing = [reason for reason in result.reasons if "could not be loaded" in reason]
    if missing:
        _print_error(console, ValueError("; ".join(missing)), options.as_json)
        return 3

    try:
        resolved.save_case_manifest(manifest)
    except CaseStoreError as exc:
        _print_error(console, exc, options.as_json)
        return 3

    if options.as_json:
        _write_json(
            {
                "case": manifest.model_dump(mode="json"),
                "validation": result.model_dump(mode="json"),
            }
        )
        return 0

    _render_case(manifest, result, console)
    return 0


def run_show_case(
    case_id: str,
    options: CaseOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        manifest = resolved.load_case_manifest(case_id)
    except CaseStoreError as exc:
        _print_error(console, exc, options.as_json)
        return 3

    result = _validate(resolved, manifest)
    if options.as_json:
        _write_json(
            {
                "case": manifest.model_dump(mode="json"),
                "validation": result.model_dump(mode="json"),
            }
        )
        return 0

    _render_case(manifest, result, console)
    return 0


def run_validate_case(
    case_id: str,
    options: CaseOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        result = resolved.validate_case(case_id, evidence_root=Path())
    except CaseStoreError as exc:
        _print_error(console, exc, options.as_json)
        return 3

    if options.as_json:
        _write_json(result.model_dump(mode="json"))
        return 0

    _render_validation(result, console)
    return 0


def run_list_cases(
    options: CaseOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        case_ids = resolved.list_case_ids()
    except CaseStoreError as exc:
        _print_error(console, exc, options.as_json)
        return 3

    if options.as_json:
        _write_json({"schema_version": 1, "case_ids": case_ids})
        return 0

    if not case_ids:
        console.print("No cases recorded yet.")
        return 0
    for case_id in case_ids:
        console.print(case_id)
    return 0


def run_export_case(
    case_id: str,
    options: ExportCaseOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        path = resolved.export_case_bundle(
            case_id,
            destination=options.destination,
            evidence_root=options.evidence_root,
        )
    except (CaseStoreError, ValueError) as exc:
        _print_error(console, exc, options.as_json)
        return 3

    if options.as_json:
        _write_json({"schema_version": 1, "case_id": case_id, "bundle_path": str(path)})
    else:
        console.print(f"Exported case bundle: {path}")
    return 0


def run_validate_case_bundle(
    root: Path,
    options: CaseOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        result = resolved.validate_case_bundle(root)
    except (CaseStoreError, ValueError) as exc:
        _print_error(console, exc, options.as_json)
        return 3

    if options.as_json:
        _write_json(result.model_dump(mode="json"))
    else:
        _render_validation(result, console)
    return 0


def _validate(
    advisor: AdvisorService,
    manifest: ExperimentalCaseManifest,
) -> CaseValidationResult:
    from jaull.cases.validation import CaseValidationService

    service = CaseValidationService(
        load_experiment=advisor.load_experiment_record,
        load_benchmark=advisor.load_benchmark_record,
        evidence_root=Path(),
    )
    return service.validate(manifest)


def _parse_evidence(item: str) -> EvidenceFileReference:
    """Parse ``path:role``; a bare path takes the ``other`` role."""

    path, separator, role = item.rpartition(":")
    if not separator:
        return EvidenceFileReference(path=item)
    try:
        parsed_role = EvidenceFileRole(role)
    except ValueError as exc:
        allowed = ", ".join(sorted(member.value for member in EvidenceFileRole))
        raise ValueError(
            f"Unknown evidence role {role!r} in {item!r}. Expected one of: {allowed}."
        ) from exc
    return EvidenceFileReference(path=path, role=parsed_role)


def _render_case(
    manifest: ExperimentalCaseManifest,
    result: CaseValidationResult,
    console: Console,
) -> None:
    label = f"  ({manifest.identity.label})" if manifest.identity.label else ""
    console.print(f"[bold]Case[/bold] {manifest.identity.case_id}{label}")
    console.print(f"Experiment: {manifest.experiment_record_id}")
    console.print(f"Benchmarks: {len(manifest.benchmark_record_ids)}")
    for benchmark_id in manifest.benchmark_record_ids:
        console.print(f"- {benchmark_id}")
    console.print(f"Runtime: {manifest.identity.runtime.value}")
    console.print(f"Artifact: {manifest.identity.artifact.repo_id}")
    if manifest.evidence_files:
        console.print("\n[bold]Evidence[/bold]")
        for reference in manifest.evidence_files:
            console.print(f"- {reference.path} ({reference.role.value})")
    for note in manifest.notes:
        console.print(f"Note: {note}")
    console.print("")
    _render_validation(result, console)


def _render_validation(result: CaseValidationResult, console: Console) -> None:
    for check in result.checks:
        detail = f": {check.detail}" if check.detail else ""
        console.print(f"{check.name}: {check.status.value}{detail}")
    for reason in result.reasons:
        console.print(f"- {reason}")
    for warning in result.warnings:
        console.print(f"Warning: {warning}")
    console.print(f"[bold]Status:[/bold] {_status_markup(result.status)}")


def _status_markup(status: CaseConsistencyStatus) -> str:
    colour = {
        CaseConsistencyStatus.VALID: "green",
        CaseConsistencyStatus.PARTIAL: "yellow",
        CaseConsistencyStatus.INCONSISTENT: "red",
    }[status]
    return f"[{colour}]{status.value}[/{colour}]"


def _write_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def _print_error(console: Console, exc: Exception, as_json: bool) -> None:
    message = str(exc)
    if as_json:
        sys.stderr.write(json.dumps({"schema_version": 1, "error": message}) + "\n")
    else:
        console.print(f"[red]{message}[/red]")


__all__ = [
    "CaseOptions",
    "CreateCaseOptions",
    "ExportCaseOptions",
    "run_create_case",
    "run_export_case",
    "run_list_cases",
    "run_show_case",
    "run_validate_case",
    "run_validate_case_bundle",
]
