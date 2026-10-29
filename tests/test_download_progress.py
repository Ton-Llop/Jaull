"""A download reports bytes; the TUI turns them into one line that replaces itself."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.cells import cell_len

from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.tui.artifact_preparation import prepare_recommendation_artifact
from jaull.workflow.download_progress import (
    DownloadProgressReporter,
    format_download_progress,
    is_progress_line,
    record_step,
    render_bar,
)

GIB = 1024**3


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        repo_id="owner/repo", revision="abc", filename="model-q4_k_m.gguf", format="gguf",
        quantization="Q4_K_M", size_bytes=1000,
    )


class _Downloader:
    """What `hf_hub_download` does with `tqdm_class`: one byte bar, updated as chunks arrive."""

    def __init__(self, chunks: list[int], total: int = 1000) -> None:
        self.chunks = chunks
        self.total = total
        self.received_tqdm_class = False

    def __call__(self, **kwargs: Any) -> str:
        bar_class = kwargs.get("tqdm_class")
        self.received_tqdm_class = bar_class is not None
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"x" * self.total)
        if bar_class is not None:
            bar = bar_class(total=self.total, unit="B", unit_scale=True, desc="model")
            for chunk in self.chunks:
                bar.update(chunk)
            bar.close()
        return str(target)


def _service(tmp_path: Path, downloader: Any) -> ArtifactService:
    return ArtifactService(
        resolver=SimpleNamespace(),  # type: ignore[arg-type]
        storage=ArtifactStorage(root=tmp_path), downloader=downloader,
    )


def test_a_download_reports_the_bytes_received_and_the_total(tmp_path: Path) -> None:
    downloader = _Downloader([400, 350, 250])
    seen: list[tuple[int, int]] = []

    artifact = _service(tmp_path, downloader).download(
        _artifact(), lambda done, total: seen.append((done, total)),
    )

    assert seen == [(400, 1000), (750, 1000), (1000, 1000)]
    assert artifact.sha256 == hashlib.sha256(b"x" * 1000).hexdigest()


def test_a_download_without_a_callback_does_not_change_how_the_downloader_is_called(
    tmp_path: Path,
) -> None:
    """Existing downloaders (and the real one when nobody listens) get no progress class."""
    downloader = _Downloader([1000])

    _service(tmp_path, downloader).download(_artifact())

    assert downloader.received_tqdm_class is False


def test_the_progress_line_shows_a_bar_the_sizes_the_speed_and_the_time_left() -> None:
    line = format_download_progress(6 * GIB, 15 * GIB, elapsed_seconds=60.0)

    assert is_progress_line(line)
    assert "40 %" in line
    assert "6.0 / 15.0 GiB" in line
    assert "MiB/s" in line  # 6 GiB in 60 s = 102 MiB/s
    assert "ETA 1:30" in line  # 9 GiB left at 102 MiB/s


def test_the_bar_keeps_its_width_and_is_only_full_when_the_download_is() -> None:
    empty, half, almost, full = render_bar(0.0), render_bar(0.4), render_bar(0.999), render_bar(1.0)

    assert empty == "░" * 20 and full == "█" * 20
    assert half == "█" * 8 + "░" * 12
    assert almost == "█" * 19 + "░"  # 99.9 % is not shown as finished
    assert {cell_len(bar) for bar in (empty, half, almost, full)} == {20}


def test_a_download_of_unknown_size_still_reports_what_has_arrived() -> None:
    line = format_download_progress(700 * 1024**2, 0, elapsed_seconds=7.0)

    assert is_progress_line(line)
    assert "700.0 MiB" in line and "%" not in line and "ETA" not in line


def test_the_reporter_limits_how_often_it_reports_and_always_reports_the_end() -> None:
    now = [0.0]
    lines: list[str] = []
    reporter = DownloadProgressReporter(lines.append, clock=lambda: now[0], min_interval=0.25)

    for step in range(100):  # 1 % per call, 10 ms apart
        now[0] = step * 0.01
        reporter(step + 1, 100)

    assert len(lines) < 10
    assert "100 %" in lines[-1]  # the end is never swallowed by the limit


def test_a_progress_line_replaces_the_previous_one_in_the_step_log() -> None:
    log: list[str] = []
    record_step(log, "Resolve artifact for owner/repo")
    record_step(log, "Downloading artifact")
    record_step(log, format_download_progress(1 * GIB, 10 * GIB, elapsed_seconds=5.0))
    record_step(log, format_download_progress(2 * GIB, 10 * GIB, elapsed_seconds=10.0))
    record_step(log, "Verifying artifact")
    record_step(log, "Model ready")

    assert len(log) == 5  # one progress line, not two
    assert "20 %" in log[2]
    assert log[3:] == ["Verifying artifact", "Model ready"]


def test_preparing_an_artifact_reports_the_download_progress() -> None:
    steps: list[str] = []
    artifact = _artifact()

    class _Advisor:
        def resolve_artifact(self, **kwargs: Any) -> ModelArtifact:
            return artifact

        def download_artifact(self, item: ModelArtifact, on_progress: Any = None) -> ModelArtifact:
            for done in (250, 500, 1000):
                on_progress(done, 1000)
            return item

        def verify_artifact(self, item: ModelArtifact) -> ModelArtifact:
            return item

    recommendation = SimpleNamespace(
        repo_id="owner/repo", evaluated=SimpleNamespace(selected_configuration=None),
    )
    prepare_recommendation_artifact(
        advisor=_Advisor(), recommendation=recommendation,  # type: ignore[arg-type]
        report_step=steps.append,
    )

    progress = [step for step in steps if is_progress_line(step)]
    assert progress and "100 %" in progress[-1]
    assert steps[0].startswith("Resolve artifact") and steps[-1] == "Model ready"
