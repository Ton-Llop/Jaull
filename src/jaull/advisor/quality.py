"""Selection guard shared by manual and automatically prepared quality runs."""

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import ArtifactVariant, ArtifactVariantFormat


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
