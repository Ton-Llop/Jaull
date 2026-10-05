"""Selection guard shared by manual and automatically prepared quality runs."""

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import ArtifactVariant, ArtifactVariantFormat, ExecutionPlan


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
