"""Explicit CLI bridge to the audited, source-checkout quality pilot.

The wheel consumes records but does not contain the evaluator/container build.
Keep that boundary explicit rather than importing an optional pilot at startup.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from jaull.domain.artifacts import ModelArtifact
from jaull.evaluation.quality_records import load_record
from jaull.exceptions import JaullError
from jaull.paths import user_data_dir


class QualityProfile(StrEnum):
    SMOKE = "smoke"
    HELLASWAG100 = "hellaswag100"
    # Full IFEval split through the GGUF's own chat template; see pilot/quality_eval/ifeval.py.
    IFEVAL = "ifeval"


#: Pilot suite, produced record grade and evaluation context of each profile.
PROFILE_SUITE = {
    QualityProfile.SMOKE: "hellaswag", QualityProfile.HELLASWAG100: "hellaswag",
    QualityProfile.IFEVAL: "ifeval",
}
PROFILE_GRADE = {
    QualityProfile.SMOKE: "plumbing", QualityProfile.HELLASWAG100: "limited",
    QualityProfile.IFEVAL: "full",
}
PROFILE_CONTEXT = {
    QualityProfile.SMOKE: 2048, QualityProfile.HELLASWAG100: 2048, QualityProfile.IFEVAL: 4096,
}


#: Image labels the pilot's Dockerfile declares (pilot/quality_eval/Dockerfile).
_CONTRACT_LABEL = ("io.jaull.quality.artifact-contract", "exact-local-gguf-v1")
_SUITES_LABEL = "io.jaull.quality.suites"
#: The suite capability an image must declare; HellaSwag predates the label.
_SUITE_CAPABILITY = {"hellaswag": None, "ifeval": "ifeval-chat-v1"}
EVALUATOR_REPOSITORY = "jaull-quality-eval"
BUILD_COMMAND = (
    "docker build --platform linux/amd64 --tag jaull-quality-eval:ifeval-chat-v1 "
    "pilot/quality_eval"
)


def find_evaluator_image(profile: QualityProfile, preferred: str | None = None) -> str | None:
    """A local evaluator image whose labels serve this profile's suite, or None.

    Local only: lists and inspects existing images, never pulls or builds. The
    remembered image wins when it qualifies; otherwise the newest that does.
    """
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["docker", "image", *args], text=True, timeout=15, stderr=subprocess.DEVNULL,
        )

    try:
        listed = run("ls", "--format", "{{.Repository}}:{{.Tag}}", EVALUATOR_REPOSITORY).split()
    except (OSError, subprocess.SubprocessError):
        return None
    names = [name for name in listed if not name.endswith(":<none>")]
    if preferred and preferred not in names:
        names.insert(0, preferred)
    names.sort(key=lambda name: name != preferred)
    capability = _SUITE_CAPABILITY[PROFILE_SUITE[profile]]
    for name in names:
        try:
            labels = json.loads(run("inspect", "--format", "{{json .Config.Labels}}", name))
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        if not isinstance(labels, dict) or labels.get(_CONTRACT_LABEL[0]) != _CONTRACT_LABEL[1]:
            continue
        if capability is None or capability in str(labels.get(_SUITES_LABEL, "")).split(","):
            return name
    return None


#: Bytes of each suite's pinned dataset file, for telling a person what downloads.
DATASET_BYTES = {"hellaswag": 6315951, "ifeval": 207111}


def pilot_checkout_complete(root: Path) -> bool:
    """Whether this directory holds every pilot file the bridge would execute."""
    root = root.expanduser()
    return all((root / name).is_file() for name in _PILOT_FILES)


def default_dataset(profile: QualityProfile) -> Path:
    """Where the pinned dataset of this profile's suite is prepared by default."""
    if PROFILE_SUITE[profile] == "ifeval":
        return user_data_dir("quality-datasets") / "ifeval-v1" / "ifeval_input_data.jsonl"
    return user_data_dir("quality-datasets") / "hellaswag-v1" / "validation.parquet"


class QualityEvaluationError(JaullError):
    """The pilot failed; its output must not become stored quality evidence."""


class QualityEvaluationCancelled(QualityEvaluationError):
    """The operator cancelled; owned processes have finished cleanup."""


@dataclass(frozen=True)
class QualityRunRequest:
    artifact_json: Path | None
    dataset_file: Path
    llama_server: Path
    image: str
    output: Path
    pilot_root: Path
    profile: QualityProfile = QualityProfile.SMOKE
    allow_dataset_download: bool = False


_SETUP_KEYS = ("dataset", "server", "root", "image")
_PILOT_FILES = (
    "scripts/quality_eval_smoke.py", "pilot/quality_eval/evaluate.py",
    "pilot/quality_eval/records.py", "pilot/quality_eval/suite.yaml",
    "pilot/quality_eval/setup.py", "pilot/quality_eval/ifeval.py",
    "pilot/quality_eval/suite_ifeval.yaml",
)


def quality_setup_defaults() -> dict[str, str]:
    """Local suggestions only: no network, Docker inspection or CWD execution."""
    root = Path(__file__).resolve().parents[3]
    defaults = {
        "dataset": str(default_dataset(QualityProfile.SMOKE)),
        "image": "jaull-quality-eval:gguf-v1",
    }
    if all((root / name).is_file() for name in _PILOT_FILES):
        defaults["root"] = str(root)
    return defaults


def _pilot_root(request: QualityRunRequest) -> Path:
    if os.name != "posix":
        raise QualityEvaluationError("This pinned host/container pilot requires Linux or WSL.")
    root = request.pilot_root.expanduser().resolve()
    if any(not (root / name).is_file() for name in _PILOT_FILES):
        raise QualityEvaluationError(
            "Quality execution requires the trusted Jaull pilot checkout, including setup.py."
        )
    if not isinstance(request.profile, QualityProfile):
        raise QualityEvaluationError("Unsupported fixed evaluation profile.")
    return root


def load_quality_setup() -> dict[str, str]:
    path = user_data_dir("quality-setup.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or set(payload) != {"schema_version", *_SETUP_KEYS}
        or any(not isinstance(payload[key], str) or not payload[key].strip()
               for key in _SETUP_KEYS)
    ):
        raise QualityEvaluationError("Invalid quality setup; enter the evaluator paths again.")
    return {key: payload[key] for key in _SETUP_KEYS}


def save_quality_setup(request: QualityRunRequest) -> None:
    """Remember infrastructure only, never artifact, output, consent or results.

    The remembered dataset is the HellaSwag one; an IFEval run keeps it rather than
    overwrite it with a file the HellaSwag profiles would then reject.
    """
    dataset = request.dataset_file.expanduser().resolve()
    if PROFILE_SUITE[request.profile] == "ifeval":
        try:
            remembered = load_quality_setup().get("dataset")
        except (OSError, ValueError, QualityEvaluationError):
            remembered = None
        dataset = Path(remembered) if remembered else default_dataset(QualityProfile.SMOKE)
    payload = {
        "schema_version": 1, "dataset": str(dataset),
        "server": str(request.llama_server.expanduser().resolve()),
        "root": str(request.pilot_root.expanduser().resolve()), "image": request.image,
    }
    path = user_data_dir("quality-setup.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _check_cancel(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise QualityEvaluationCancelled("Evaluation cancelled; no result imported.")


def _invoke(
    command: list[str], root: Path, log: Path,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    _check_cancel(is_cancelled)
    environment = os.environ.copy()
    # The evaluator remains an explicit source-checkout dependency, even when
    # Jaull itself was installed from a wheel. Never execute through a shell.
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    with (
        log.open("x", encoding="utf-8") as handle,
        subprocess.Popen(
            command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            if is_cancelled is None:
                code = process.wait()
            else:
                while True:
                    _check_cancel(is_cancelled)
                    try:
                        code = process.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        except QualityEvaluationCancelled:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
        except KeyboardInterrupt:
            # Let the pilot's finally stop its owned container/server and
            # finish artifact-after verification; do not kill its owner.
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
    if code:
        raise QualityEvaluationError(f"Quality pilot exited with {code}; inspect {log}.")


def prepare_quality_setup(
    request: QualityRunRequest, *, is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Shared preflight, before any model download; optional pinned dataset preparation."""
    _check_cancel(is_cancelled)
    root = _pilot_root(request)
    if not request.llama_server.expanduser().is_file():
        raise QualityEvaluationError(f"Runtime input missing: {request.llama_server}")
    if not request.dataset_file.expanduser().is_file() and not request.allow_dataset_download:
        raise QualityEvaluationError(
            "Evaluation input missing: pinned dataset. Enable dataset download permission first."
        )
    if on_progress is not None:
        on_progress("Checking runtime/image and preparing the pinned dataset")
    output = user_data_dir("quality-setup-runs") / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    report = output / "setup.json"
    command = [
        sys.executable, "-m", "pilot.quality_eval.setup",
        "--dataset-file", str(request.dataset_file.expanduser().resolve()),
        "--llama-server", str(request.llama_server.expanduser().resolve()),
        "--image", request.image, "--output", str(report),
        "--suite", PROFILE_SUITE[request.profile],
    ]
    if request.allow_dataset_download:
        command.append("--allow-dataset-download")
    try:
        _invoke(command, root, output / "setup.log", is_cancelled)
    except QualityEvaluationCancelled:
        raise
    except QualityEvaluationError as exc:
        if report.is_file():
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise QualityEvaluationError(
                    f"Invalid infrastructure preflight report; inspect {output}."
                ) from error
            if isinstance(payload, dict) and payload.get("status") == "blocked":
                raise QualityEvaluationError(
                    f"{payload.get('error', 'Setup blocked')} Logs: {output}"
                ) from exc
        raise
    _check_cancel(is_cancelled)
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualityEvaluationError(
            f"Missing or invalid infrastructure preflight report; inspect {output}."
        ) from exc
    if (
        not isinstance(payload, dict) or set(payload) != {"status", "image_id"}
        or payload["status"] != "ready" or not isinstance(payload["image_id"], str)
        or re.fullmatch("sha256:[0-9a-f]{64}", payload["image_id"]) is None
    ):
        raise QualityEvaluationError(f"Invalid infrastructure preflight result; inspect {output}.")


def run_quality_evaluation(
    request: QualityRunRequest, *, is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one bounded profile, then reconstruct its HTTP-validated snapshot.

    No model download, build, cache reuse or model substitution is performed here.
    A missing pinned dataset is prepared only with explicit request permission.
    The existing pilot owns verification, fit/readiness and process cleanup.
    """
    _check_cancel(is_cancelled)
    root = _pilot_root(request)
    if request.artifact_json is None:
        raise QualityEvaluationError("Prepare the selected GGUF or provide an artifact manifest.")
    artifact = ModelArtifact.model_validate_json(
        request.artifact_json.expanduser().read_text(encoding="utf-8")
    )
    if artifact.local_path is None or not artifact.sha256:
        raise QualityEvaluationError("An exact local artifact path and SHA256 are required.")
    local_path = artifact.local_path.expanduser().resolve()
    artifact = artifact.model_copy(update={"local_path": local_path})
    dataset = request.dataset_file.expanduser().resolve()
    server = request.llama_server.expanduser().resolve()
    if not local_path.is_file():
        raise QualityEvaluationError(
            "Artifact must already exist locally."
        )
    output = request.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    prepare_quality_setup(request, is_cancelled=is_cancelled)
    output.mkdir(parents=True, exist_ok=False)
    artifact_json = output / "artifact.json"
    artifact_json.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    bundle, snapshot = output / "bundle", output / "record.json"
    # Keep the synchronous CLI call unchanged; TUI workers opt into polling.
    invoke_args = () if is_cancelled is None else (is_cancelled,)
    _invoke(
        [sys.executable, "-m", "scripts.quality_eval_smoke",
         "--artifact-json", str(artifact_json), "--dataset-file", str(dataset),
         "--llama-server", str(server), "--image", request.image,
         "--profile", request.profile.value, "--output", str(bundle)],
        root, output / "run.log", *invoke_args,
    )
    _invoke(
        [sys.executable, "-m", "pilot.quality_eval.records",
         "--bundle", str(bundle), "--output", str(snapshot)],
        root, output / "snapshot.log", *invoke_args,
    )
    record = load_record(snapshot)
    if record["identity"]["artifact_sha256"] != artifact.sha256:
        raise QualityEvaluationError("Produced record belongs to a different artifact.")
    if record["classification"] != PROFILE_GRADE[request.profile]:
        raise QualityEvaluationError("Produced record belongs to a different evaluation profile.")
    _check_cancel(is_cancelled)
    return record
