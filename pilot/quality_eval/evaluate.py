"""Fixed smoke/limited samples. No response cache, ranking or speed evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
from pathlib import Path
from typing import Any

TASK = "jaull-quality-smoke-v1"
SAMPLE_IDS = [0, 1, 2]
LIMITED_TASK = "jaull-hellaswag-100-v1"
DATASET_SIZE = 10042
PROFILES = {
    "smoke": {"task": TASK, "sample_ids": SAMPLE_IDS, "classification": "plumbing",
              "status": "plumbing_passed"},
    "hellaswag100": {
        "task": LIMITED_TASK,
        "sample_ids": sorted(random.Random(20261002).sample(range(DATASET_SIZE), 100)),
        "classification": "limited", "status": "limited_passed",
    },
}


def profile_for_task(task: str) -> dict[str, Any]:
    for profile in PROFILES.values():
        if profile["task"] == task:
            return profile
    raise ValueError("Unknown fixed evaluation profile")


CONTEXT = 2048
EVALUATOR_COMMIT = "ad8737ae7fad24cf64e50fc7fc31397bff586b9e"
BACKEND_SHA256 = "71161c2b04e55699b0397e3f745da74a8c39e010f88371418a65bdcad3b8bc63"
SERVER_SHA256 = "cdb0749a2cffc6f2fe710a9616263f90e6a016e8ac07a9caa5be160fc2c5d98e"
ARTIFACT_SHA256 = "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"
ARTIFACT_PINS = {
    ARTIFACT_SHA256: {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "revision": "52e7645ba7c309695bec7ac98f4f005b139cf465",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": 668788096,
    },
    "b46661073c18e5b56a41fa320975f866a00def1ff08feef4718e013258896f8c": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
        "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf",
        "format": "gguf",
        "quantization": "Q5_K_M",
        "size_bytes": 1285494304,
    },
}
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
DATASET_URL = (
    f"https://huggingface.co/datasets/Rowan/hellaswag/resolve/{DATASET_REVISION}/"
    "data/validation-00000-of-00001.parquet"
)
DATASET_SHA256 = "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"


def validate_artifact_identity(artifact: dict[str, Any]) -> None:
    pin = ARTIFACT_PINS.get(artifact.get("sha256"))
    if pin is None or any(artifact.get(field) != value for field, value in pin.items()):
        raise ValueError("Only the two audited exact GGUF identities are supported")


def verified_dataset_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATASET_SHA256:
        raise ValueError("Pinned HellaSwag validation file failed SHA256 verification")
    return data


def write_json(path: Path, value: Any) -> None:
    # Exclusive creation protects previous attempts, including failed ones.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        handle.write("\n")


def validate_task(config: dict[str, Any]) -> None:
    process_docs = config.get("process_docs")
    expected = {
        "task": TASK,
        "dataset_path": "parquet",
        "dataset_name": None,
        "dataset_kwargs": {"data_files": {"validation": DATASET_URL}},
        "output_type": "multiple_choice",
        "validation_split": "validation",
        "training_split": None,
        "test_split": None,
        "num_fewshot": 0,
        "process_docs": "lm_eval.tasks.hellaswag.utils.process_docs",
        "doc_to_text": "{{query}}",
        "doc_to_target": "{{label}}",
        "doc_to_choice": "choices",
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
        ],
        "metadata": {
            "version": 1.0,
            "dataset_repo": "Rowan/hellaswag",
            "dataset_revision": DATASET_REVISION,
        },
    }
    definition = config | {
        "process_docs": (
            f"{getattr(process_docs, '__module__', '')}.{getattr(process_docs, '__name__', '')}"
        ),
    }
    if not callable(process_docs) or definition != expected:
        raise ValueError("Only the fixed zero-shot multiple-choice smoke suite is supported")


def validate_scoring_request(payload: dict[str, Any]) -> None:
    if set(payload) != {"prompt", "temperature", "max_tokens", "logprobs", "logit_bias", "id_slot"}:
        raise ValueError("Unsupported or missing GGUF inference setting")
    ids = payload["prompt"]
    bias = payload["logit_bias"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(token) is not int or token < 0 for token in ids)
        or len(ids) + 1 > CONTEXT
        or any(type(payload[key]) is not int for key in ("max_tokens", "logprobs", "id_slot"))
        or type(payload["temperature"]) not in (int, float)
        or payload["temperature"] != 0
        or payload["max_tokens"] != 1
        or payload["logprobs"] != 2
        or payload["id_slot"] != 0
        or not isinstance(bias, list)
        or len(bias) != 1
        or not isinstance(bias[0], list)
        or len(bias[0]) != 2
        or type(bias[0][0]) is not int
        or bias[0][0] < 0
        or bias[0][1] != 100
    ):
        raise ValueError("Scoring request differs from the observed one-slot protocol")


def validate_scoring_response(payload: dict[str, Any], response: dict[str, Any]) -> None:
    entry = response["choices"][0]["logprobs"]["content"][0]
    top = entry.get("top_logprobs", [])
    if (
        response.get("system_fingerprint") != "b10357-689e227db"
        or len(response["choices"][0]["logprobs"]["content"]) != 1
        or response.get("usage", {}).get("completion_tokens") != 1
        or response.get("usage", {}).get("prompt_tokens") != len(payload["prompt"])
        or response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") != 0
        or entry.get("id") != payload["logit_bias"][0][0]
        or type(entry.get("logprob")) not in (int, float)
        or not math.isfinite(entry["logprob"])
        or entry["logprob"] > 0
        or not top
        or any(not math.isfinite(token["logprob"]) for token in top)
    ):
        raise ValueError("Missing/mismatched token, runtime, uncached prompt or finite logprobs")


def run(output: Path, base_url: str, profile_name: str = "smoke") -> None:
    profile = PROFILES[profile_name]
    task, sample_ids = profile["task"], profile["sample_ids"]
    import requests
    from lm_eval import simple_evaluate
    from lm_eval.models import gguf
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.utils import handle_non_serializable

    source = Path(gguf.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != BACKEND_SHA256:
        raise ValueError("Installed GGUF backend differs from the audited source")
    suite = Path(__file__).with_name("suite.yaml")
    config = load_yaml(suite, resolve_func=True)
    validate_task(config)
    # Same pinned task definition; the profile gives the larger subset its own identity.
    config["task"] = task
    artifact = json.loads((output / "verified-artifact.json").read_text())
    validate_artifact_identity(artifact)
    with Path("/artifact/model.gguf").open("rb") as model:
        digest = hashlib.file_digest(model, "sha256").hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError("Read-only container artifact differs from the audited GGUF")
    write_json(output / "container-artifact.json", {"sha256": digest, "mount": "read-only"})
    props = requests.get(base_url + "/props", timeout=15)
    props.raise_for_status()
    props = props.json()
    write_json(output / "server-props.json", props)
    if (
        props.get("total_slots") != 1
        or props["default_generation_settings"].get("n_ctx") != CONTEXT
        or props["default_generation_settings"]["params"].get("post_sampling_probs") is not False
        or props.get("model_path")
        != artifact["local_path"]
    ):
        raise ValueError("Server slot/context/logprob settings differ from the pilot")
    write_json(
        output / "evaluator-config.json",
        {
            "evaluator_commit": EVALUATOR_COMMIT,
            "backend_sha256": BACKEND_SHA256,
            "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
            "sample_ids": sample_ids,
            "num_fewshot": 0,
            "context": CONTEXT,
            "apply_chat_template": False,
            "system_instruction": None,
            "generation": None,
            "use_cache": None,
            "cache_requests": False,
            "seeds": [0, 1234, 1234, 1234],
            "server_seed": 0,
            "python": platform.python_version(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    )
    dataset = verified_dataset_bytes(Path("/dataset/validation.parquet"))
    dataset_path = output / "dataset-validation.parquet"
    with dataset_path.open("xb") as handle:
        handle.write(dataset)
    write_json(
        output / "dataset.json",
        {
            "repo": "Rowan/hellaswag",
            "revision": DATASET_REVISION,
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "sample_ids": sample_ids,
        },
    )
    # Load the verified file with the native parquet loader, retaining split checks.
    config["dataset_kwargs"] = {"data_files": {"validation": str(dataset_path)}}

    with (output / "http.jsonl").open("x", encoding="utf-8") as exchanges:

        class CheckedGGUF(gguf.GGUFLM):
            def _post_with_retries(self, url, payload, **kwargs):
                record = {"url": url, "request": payload}
                try:
                    if url == self.completions_url:
                        validate_scoring_request(payload)
                    elif url != self.tokenize_url or set(payload) != {"content", "add_special"}:
                        raise ValueError("Unsupported GGUF endpoint or tokenizer setting")
                    response = super()._post_with_retries(url, payload, **kwargs)
                    record["response"] = response
                    if url == self.completions_url:
                        validate_scoring_response(payload, response)
                    return response
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    # Raw failed evidence may contain non-finite values; never reuse it.
                    exchanges.write(json.dumps(record) + "\n")
                    exchanges.flush()

            def loglikelihood(self, requests, disable_tqdm=False):
                # Validate every selected continuation before any scoring inference.
                for request in requests:
                    context, continuation = request.args
                    spaces = len(context) - len(context.rstrip())
                    if spaces:
                        continuation = context[-spaces:] + continuation
                        context = context[:-spaces]
                    whole = self._tokenize(context + continuation, add_special=True)
                    prefix = self._tokenize(context, add_special=True)
                    if not prefix or len(whole) > CONTEXT or whole[: len(prefix)] != prefix:
                        raise ValueError("Context overflow or ambiguous tokenization boundary")
                return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

        lm = CheckedGGUF(base_url=base_url, parallel=1, temperature=0, timeout=15)
        result = simple_evaluate(
            model=lm,
            tasks=[config],
            task_manager=TaskManager(include_path=str(suite.parent), include_defaults=False),
            num_fewshot=0,
            samples={task: sample_ids},
            log_samples=True,
            use_cache=None,
            cache_requests=False,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            bootstrap_iters=0,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
    if (
        result is None
        or result["n-samples"][task] != {"original": DATASET_SIZE, "effective": len(sample_ids)}
        or sorted(sample["doc_id"] for sample in result["samples"][task]) != sample_ids
    ):
        raise ValueError("Harness did not complete exactly the selected samples")
    # Preserve full harness results, including samples. Not a reusable quality record.
    with (output / "lm-eval-results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=handle_non_serializable)
        handle.write("\n")
    write_json(
        output / "smoke-status.json", {"status": profile["status"], "quality_evidence": False}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, choices=["http://host.docker.internal:18083"])
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    try:
        run(args.output, args.base_url, args.profile)
    except Exception as exc:
        write_json(
            args.output / "evaluator-error.json", {"type": type(exc).__name__, "error": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
