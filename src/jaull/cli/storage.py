"""``storage`` subcommands: what the TUI's Free disk space screen does, for other front ends."""

from __future__ import annotations

import json
import sys

from jaull.advisor.service import AdvisorService
from jaull.presentation.console import format_bytes, make_console


def run_storage_list(advisor: AdvisorService | None = None, *, as_json: bool = False) -> int:
    resolved = advisor or AdvisorService.default()
    models = resolved.local_models()
    folders, run_bytes = resolved.quality_run_folders()
    if as_json:
        payload = {
            "schema_version": 1,
            "models": [
                {
                    "repo_id": item.repo_id,
                    "filename": item.filename,
                    # What `serve start -q` needs to load this exact file again.
                    "quantization": resolved.gguf_quantization(item.filename),
                    "size_bytes": item.size_bytes,
                    "verified": item.sha256 is not None,
                    # A quality result was measured on this file; the result stays if it is removed.
                    "measured": measured,
                }
                for item, measured in models
            ],
            "evaluation_logs": {"folders": folders, "size_bytes": run_bytes},
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    console = make_console()
    if not models:
        console.print("No downloaded model files.")
    for item, measured in models:
        note = " (quality measured)" if measured else ""
        size = format_bytes(item.size_bytes)
        line = f"{size:>10}  {item.repo_id}/{item.filename}{note}"
        console.print(line, highlight=False, markup=False)
    logs = format_bytes(run_bytes)
    console.print(f"Evaluation logs: {folders} folders, {logs}.", highlight=False)
    return 0


def run_storage_remove(
    repo_id: str | None,
    filename: str | None,
    *,
    evaluation_logs: bool = False,
    advisor: AdvisorService | None = None,
    as_json: bool = False,
) -> int:
    """Remove one downloaded model file, or the evaluation run folders. Saved results stay."""
    resolved = advisor or AdvisorService.default()
    if evaluation_logs:
        freed = resolved.delete_quality_run_folders()
    elif repo_id and filename:
        freed = resolved.delete_local_model(repo_id, filename)
    else:
        message = "Give a repo id and a file name, or --evaluation-logs."
        if as_json:
            sys.stdout.write(json.dumps({"schema_version": 1, "error": message}) + "\n")
        else:
            make_console().print(message, style="red")
        return 2
    if as_json:
        sys.stdout.write(json.dumps({"schema_version": 1, "freed_bytes": freed}) + "\n")
    else:
        make_console().print(f"Freed {format_bytes(freed)}.", highlight=False)
    return 0


__all__ = ["run_storage_list", "run_storage_remove"]
