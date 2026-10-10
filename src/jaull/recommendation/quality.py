"""Read-only quality projection for diagnostics and shadow; no active ranking."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.recommendation import (
    ApplicableQuality,
    AssessmentLevel,
    ExternalEvaluationEvidence,
    LocalQualityReference,
    QualityAssessment,
    QualityTargetProfile,
)
from jaull.domain.requirements import UseCase, UserRequirements
from jaull.evaluation.quality_comparison import comparison_key
from jaull.evaluation.quality_records import (
    QualityEvidence,
    describe_record,
    digest,
    identity_mode,
    repeat_key,
)

#: The one audited suite that can serve each profile. A profile missing here has
#: no applicable evidence at all, however complete its records look.
PROFILE_SUITES = {"ifeval-instructions-en-v1": "jaull-ifeval-chat-v1"}
#: Declared before any result is seen, as the plan requires.
PRIMARY_METRICS = {"ifeval-instructions-en-v1": "prompt_level_strict_acc"}
# Pins from the Step 4 audit, not from model scores. Drift needs another audit.
IFEVAL_SUITE_SHA256 = "2822c4af6d1274027884bcb825610a48a060ce58d9714051cb48fd51bde1a9c8"
IFEVAL_DATASET = {
    "repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "sha256": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
    "total_samples": 541, "sample_ids": list(range(541)),
}
IFEVAL_EVALUATOR_COMMIT = "ad8737ae7fad24cf64e50fc7fc31397bff586b9e"
IFEVAL_SOURCES_SHA256 = "97d0fb7fe8ea896a9e54b5042b7e0feb21b8cbb1d75b4b92d03f778d068fc850"
IFEVAL_SERVER_SHA256 = "cdb0749a2cffc6f2fe710a9616263f90e6a016e8ac07a9caa5be160fc2c5d98e"


def _audited_ifeval(identity: dict[str, Any]) -> bool:
    evaluator = identity["evaluator"]
    return (
        identity_mode(identity) == "chat_generation"
        and identity["suite"]["sha256"] == IFEVAL_SUITE_SHA256
        and identity["dataset"] == IFEVAL_DATASET
        and evaluator["suite_sha256"] == IFEVAL_SUITE_SHA256
        and evaluator["evaluator_commit"] == IFEVAL_EVALUATOR_COMMIT
        and digest(evaluator["source_sha256"]) == IFEVAL_SOURCES_SHA256
        and evaluator["context"] == 4096
        and evaluator["langdetect_seed"] == 0
        and identity["runtime"]["server_sha256"] == IFEVAL_SERVER_SHA256
        and identity["runtime"]["fingerprint"] == "b10357-689e227db"
    )


@dataclass(frozen=True)
class QualityRecordIndex:
    """Records validated and digested once, looked up per plan by artifact SHA256.

    Validation and the whole-record digest do not depend on the plan, and the
    store keeps every repeated run, so doing them per plan grows with plans x
    records. Only the lookup is per plan.
    """

    by_artifact: Mapping[
        str, tuple[tuple[QualityEvidence, str, tuple[str, bool, str]], ...]
    ] = field(
        default_factory=dict,
    )
    invalid: int = 0


def index_quality_records(records: Sequence[dict[str, Any]]) -> QualityRecordIndex:
    grouped: dict[str, list[tuple[QualityEvidence, str, tuple[str, bool, str]]]] = {}
    invalid = 0
    for record in records:
        # Context may come from callers other than the validated store.
        try:
            evidence = describe_record(record)
            record_sha256 = digest(record)
            keys = (
                comparison_key(record), _audited_ifeval(record["identity"]),
                repeat_key(record),
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            invalid += 1
            continue
        grouped.setdefault(evidence.artifact_sha256, []).append(
            (evidence, record_sha256, keys),
        )
    return QualityRecordIndex(
        by_artifact={sha: tuple(items) for sha, items in grouped.items()}, invalid=invalid,
    )


@dataclass(frozen=True)
class StoredQuality:
    """What the store holds for one artifact under one profile, decided once.

    Shadow applicability and evaluation selection both read this, so a screen
    can never call a result applicable that the shadow would not apply.
    """

    applicable: QualityEvidence | None = None
    record_sha256: str | None = None
    cohort: str | None = None
    #: Audited complete runs of the profile's suite, and how many results they give.
    complete: int = 0
    distinct: int = 0

    @property
    def conflict(self) -> bool:
        return self.distinct > 1


def stored_quality(
    index: QualityRecordIndex, artifact_sha256: str | None,
    profile: QualityTargetProfile | None,
) -> StoredQuality:
    """One applicable result, a conflict between repeats, or nothing applicable."""
    suite = PROFILE_SUITES.get(profile) if profile is not None else None
    if artifact_sha256 is None or profile is None or suite is None:
        return StoredQuality()
    complete = sorted(
        (record_sha256, evidence, cohort, repeat)
        for evidence, record_sha256, (cohort, audited, repeat)
        in index.by_artifact.get(artifact_sha256, ())
        if evidence.suite == suite and evidence.reusable and audited
        and any(m.name == PRIMARY_METRICS[profile] for m in evidence.metrics)
    )
    # Repeats with the same answers are one result; the lowest digest stands for them.
    distinct = len({repeat for *_, repeat in complete})
    if distinct != 1:
        return StoredQuality(complete=len(complete), distinct=distinct)
    record_sha256, evidence, cohort, _ = complete[0]
    return StoredQuality(evidence, record_sha256, cohort, len(complete), distinct)


def requested_profile(requirements: UserRequirements) -> QualityTargetProfile | None:
    """The quality target a request asks for, from the request alone.

    Never from available scores or model tags. These are targets, not enabled
    evaluator suites or authorization to rank; evaluation selection reuses this
    so it can only offer a suite the assessment would then accept.
    """
    if {language.strip().casefold() for language in requirements.languages} != {"en"}:
        return None
    if requirements.use_case is UseCase.GENERAL_CHAT:
        return "ifeval-instructions-en-v1"
    if requirements.use_case is UseCase.CODING:
        return "humaneval-python-en-v1"
    return None


def assess_quality(
    plan: ExecutionPlan, requirements: UserRequirements, metadata_prior: AssessmentLevel,
    records: Sequence[dict[str, Any]] | QualityRecordIndex, *,
    published: Sequence[ExternalEvaluationEvidence] = (),
) -> QualityAssessment:
    """Select a target profile and project already-attributed historical references.

    Only the audited complete IFEval profile can apply to English chat in shadow.
    A suite label or a record's own total cannot prove that profile. Published
    attribution remains the engine's responsibility, with provenance separate.
    """
    index = records if isinstance(records, QualityRecordIndex) else index_quality_records(records)
    profile = requested_profile(requirements)
    suite = PROFILE_SUITES.get(profile) if profile is not None else None
    profile_blocker = (
        "No validated quality profile covers the requested task and languages."
        if profile is None else
        f"Profile {profile} is applicable only through one complete {suite} run "
        "of this exact artifact, matching the audited pins for suite, dataset, evaluator "
        "and runtime."
        if suite is not None else
        f"Profile {profile} requires an audited generation contract; none exists yet."
    )
    matches = (
        index.by_artifact.get(plan.artifact.sha256, ())
        if plan.artifact.sha256 is not None else ()
    )
    references = [
        LocalQualityReference(
            identity_sha256=evidence.identity_sha256,
            record_sha256=record_sha256,
            artifact_sha256=evidence.artifact_sha256,
            suite=evidence.suite,
            classification=evidence.classification,
            samples_used=evidence.samples_used,
            samples_available=evidence.samples_available,
            metrics={metric.name: metric.value for metric in evidence.metrics},
            blockers=(*evidence.limitations,
                      "Current runtime, placement and effective protocol are not verified.",
                      profile_blocker),
        )
        for evidence, record_sha256, _ in matches
    ]
    stored = stored_quality(index, plan.artifact.sha256, profile)
    applicable = None
    if stored.applicable is not None and profile is not None:
        evidence, record_sha256, cohort = stored.applicable, stored.record_sha256, stored.cohort
        assert record_sha256 is not None and cohort is not None
        metric = next(m for m in evidence.metrics if m.name == PRIMARY_METRICS[profile])
        applicable = ApplicableQuality(
            suite=evidence.suite, metric=metric.name, value=metric.value,
            correct=metric.correct, samples=metric.samples, cohort=cohort,
            record_sha256=record_sha256,
        )
        # The applied record is exactly what the profile blocker asks for.
        references = [
            item.model_copy(update={"blockers": tuple(
                blocker for blocker in item.blockers if blocker != profile_blocker
            )}) if item.record_sha256 == record_sha256 else item
            for item in references
        ]
    assessment = QualityAssessment(
        use_case=requirements.use_case, languages=tuple(requirements.languages),
        requested_profile=profile, metadata_prior=metadata_prior,
        applicability=(
            "controlled_artifact" if applicable is not None
            else "reference_only" if references or published else "absent"
        ),
        applicable=applicable,
        local_references=tuple(sorted(references, key=lambda item: (
            item.identity_sha256, item.record_sha256,
        ))),
    )
    limitations = (
        [assessment.limitations[0], applicable.scope] if applicable is not None
        else [*assessment.limitations, profile_blocker]
    )
    if stored.conflict:
        # Repeats that disagree: picking one would be choosing the result.
        limitations.append(
            f"{stored.complete} complete {suite} runs of this artifact give {stored.distinct} "
            "different results; no rule selects one, so none is applied."
        )
    elif stored.complete > 1:
        limitations.append(
            f"{stored.complete} complete {suite} runs of this artifact gave identical "
            "answers; they count as one result."
        )
    if published:
        limitations.append(
            "Published model references have not passed task, representation and protocol "
            "applicability gates; they do not measure this artifact or replace local records."
        )
    if index.invalid:
        limitations.append(f"Ignored {index.invalid} invalid local quality record(s).")
    return assessment.model_copy(update={"limitations": tuple(limitations)})
