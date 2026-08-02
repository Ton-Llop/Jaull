from __future__ import annotations

from rich.console import Console
from rich.table import Table

from local_ai_check.domain.model import ModelAnalysis
from local_ai_check.presentation.console import format_bytes


def render_model(analysis: ModelAnalysis, console: Console) -> None:
    repo = analysis.repo
    classification = analysis.classification
    config = analysis.config

    header = Table(title="Model repository", show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value")

    header.add_row("Repository", repo.repo_id)
    if repo.author:
        header.add_row("Author", repo.author)
    header.add_row("Type", classification.primary_type.value)
    if len(classification.detected_types) > 1:
        types_str = ", ".join(sorted(t.value for t in classification.detected_types))
        header.add_row("Detected types", types_str)
    if repo.pipeline_tag:
        header.add_row("Task", repo.pipeline_tag)
    if repo.library_name:
        header.add_row("Library", repo.library_name)

    if config is not None:
        if config.architectures:
            header.add_row("Architecture", ", ".join(config.architectures))
        if config.model_type:
            header.add_row("Model type", config.model_type)
        if config.torch_dtype:
            header.add_row("Original dtype", config.torch_dtype)
        if config.max_position_embeddings:
            header.add_row("Context length", str(config.max_position_embeddings))

    if repo.license:
        header.add_row("License", repo.license)
    header.add_row("Gated", "Yes" if repo.gated else "No")
    if repo.private:
        header.add_row("Private", "Yes")
    if repo.downloads is not None:
        header.add_row("Downloads", f"{repo.downloads:,}")
    if repo.likes is not None:
        header.add_row("Likes", str(repo.likes))
    if repo.last_modified is not None:
        header.add_row("Last modified", repo.last_modified.strftime("%Y-%m-%d %H:%M UTC"))

    if classification.formats:
        formats = ", ".join(sorted(f.value for f in classification.formats))
        header.add_row("Formats", formats)

    header.add_row("Repository size", format_bytes(analysis.total_size_bytes))
    if classification.gguf_variants:
        largest = max(v.total_bytes for v in classification.gguf_variants)
        header.add_row("Largest variant", format_bytes(largest))
    header.add_row("Files", str(len(analysis.files)))

    console.print(header)

    if classification.gguf_variants:
        console.print()
        variants = Table(title="GGUF variants", show_lines=False)
        variants.add_column("Quantization", style="cyan")
        variants.add_column("Files", justify="right")
        variants.add_column("Size", justify="right")
        for variant in classification.gguf_variants:
            variants.add_row(
                variant.quantization,
                str(len(variant.files)),
                format_bytes(variant.total_bytes),
            )
        console.print(variants)

    if analysis.relevant_files:
        console.print()
        console.print("[bold]Relevant files[/bold]")
        for name in analysis.relevant_files:
            console.print(f"  {name}")

    if analysis.warnings:
        console.print()
        for warning in analysis.warnings:
            console.print(f"[yellow]! {warning}[/yellow]")
