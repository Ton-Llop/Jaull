"""Selection guard shared by manual and automatically prepared quality runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import ArtifactVariant, ArtifactVariantFormat, ExecutionPlan

if TYPE_CHECKING:
    from jaull.runtime.quality_eval_runner import QualityProfile


@dataclass(frozen=True)
class QualityReadiness:
    """Each row is (state, name, detail); state is "ok", "warn" or "bad"."""

    rows: tuple[tuple[str, str, str], ...]
    image: str | None = None

    @property
    def blocked(self) -> bool:
        return any(state == "bad" for state, _, _ in self.rows)

    @property
    def dataset_missing(self) -> bool:
        return any(name == "Dataset" and state == "warn" for state, name, _ in self.rows)


def check_quality_readiness(
    prepare: Callable[..., None], profile: QualityProfile, values: Mapping[str, str], *,
    is_cancelled: Callable[[], bool] | None = None,
) -> QualityReadiness:
    """Resolve and verify the evaluator locally; never download, pull or build.

    ``prepare`` is the advisor's shared infrastructure preflight, run only when
    every input is present, so a missing dataset is reported instead of fetched.
    """
    from jaull.runtime.quality_eval_runner import (
        BUILD_COMMAND,
        DATASET_BYTES,
        PROFILE_SUITE,
        QualityEvaluationCancelled,
        QualityRunRequest,
        find_evaluator_image,
        pilot_checkout_complete,
    )

    suite = PROFILE_SUITE[profile]
    rows: list[tuple[str, str, str]] = []
    image = find_evaluator_image(profile, values.get("image") or None)
    rows.append(("ok", "Evaluator image", image) if image else (
        "bad", "Evaluator image",
        f"No local image supports {suite}. Build it once: {BUILD_COMMAND}",
    ))
    server = Path(values["server"]).expanduser() if values.get("server") else None
    rows.append(("ok", "llama-server", str(server)) if server and server.is_file() else (
        "bad", "llama-server", "Not found; set its path in Settings.",
    ))
    root = Path(values["root"]) if values.get("root") else None
    rows.append(("ok", "Pilot checkout", str(root))
                if root and pilot_checkout_complete(root) else (
        "bad", "Pilot checkout", "Not found; set the Jaull source checkout in Settings.",
    ))
    dataset = Path(values["dataset"]).expanduser() if values.get("dataset") else None
    present = bool(dataset and dataset.is_file())
    size = DATASET_BYTES[suite]
    rows.append(("ok", "Dataset", f"Pinned {suite} file present") if present else (
        "warn", "Dataset", f"Missing; {size / 1024**2:.1f} MB download at start, with permission.",
    ))
    if not any(state == "bad" for state, _, _ in rows):
        if present:
            assert image is not None and server is not None and root is not None
            assert dataset is not None
            try:
                prepare(QualityRunRequest(
                    artifact_json=None, dataset_file=dataset, llama_server=server, image=image,
                    pilot_root=root, output=Path(values.get("output") or "."), profile=profile,
                ), is_cancelled=is_cancelled)
                rows.append(("ok", "Verification",
                             "Pinned runtime binary, image contract and dataset match"))
            except QualityEvaluationCancelled:
                raise
            except Exception as exc:  # The preflight's own reason is the useful message.
                rows.append(("bad", "Verification", str(exc)))
        else:
            rows.append(("warn", "Verification", "Runs at start, after the dataset download."))
    return QualityReadiness(tuple(rows), image)


def quality_evaluation_block_reason(plan: ExecutionPlan) -> str | None:
    """Known pilot blockers; server/image and fixed-context fit still need preflight.

    Do not use llama-cli readiness: this pilot uses a separately pinned server,
    and its CUDA/full-offload protocol is independent of the selected launch.
    """
    if plan.artifact.format is not ArtifactVariantFormat.GGUF:
        return "Select a GGUF path to evaluate this model."
    if plan.artifact.filename is None:
        return "The selected path does not identify a single GGUF file."
    if plan.artifact.file_count not in (None, 1):
        return "Multipart GGUF is not supported by the quality pilot."
    if plan.hardware is not None and len(plan.hardware.gpus) != 1:
        return "The fixed quality pilot requires one CUDA GPU."
    return None


def check_selected_artifact(artifact: ModelArtifact, selected: ArtifactVariant) -> None:
    """A manifest cannot silently substitute another selected variant/revision."""
    if selected.format is not ArtifactVariantFormat.GGUF:
        raise ValueError("Quality evaluation requires a GGUF artifact.")
    if selected.filename is None:
        raise ValueError("The selected path does not identify a single GGUF file.")
    if selected.file_count is not None and selected.file_count != 1:
        raise ValueError("Multipart GGUF is not supported by the quality pilot.")
    if (
        artifact.format != "gguf"
        or artifact.repo_id != selected.repo_id
        or artifact.filename != selected.filename
        or artifact.quantization != selected.quantization
        or (selected.sha256 is not None and artifact.sha256 != selected.sha256)
        or (selected.size_bytes is not None and artifact.size_bytes != selected.size_bytes)
        or (selected.revision not in (None, "main") and artifact.revision != selected.revision)
    ):
        raise ValueError("The local manifest does not match the selected artifact.")
