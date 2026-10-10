"""IFEval through the GGUF's own chat template. Generation, so nothing here is HellaSwag.

Contract (docs/recommendation-vfinal-plan.md, step 4 audit):
- `local-chat-completions` sends unrendered messages; llama-server applies the
  jinja template embedded in the GGUF, and `/apply-template` records the text.
- Reasoning is off and verified on every response; any trace invalidates the run.
- A response the backend cannot parse fails the run instead of scoring as "".
- `langdetect` is seeded, so this suite is Jaull's and not comparable upstream.
- Prompt plus the full 1280-token budget must fit the context, checked per reply.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from pilot.quality_eval.evaluate import (
    EVALUATOR_COMMIT,
    PROFILES,
    validate_artifact_identity,
    write_json,
)

CONTEXT = 4096
MAX_GEN_TOKS = 1280
SEED = 1234
MODEL_NAME = "jaull-gguf"
RUNTIME_FINGERPRINT = "b10357-689e227db"
DATASET_SIZE = 541
DATASET_REPO = "google/IFEval"
DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/"
    "ifeval_input_data.jsonl"
)
DATASET_SHA256 = "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"
DATASET_BYTES = 207111
# The exact sources the contract was audited against, by installed module.
# Source: .codex-night/ifeval-audit-20261008/src/MANIFEST.json, fetched at the pin.
SOURCE_SHA256 = {
    "lm_eval.models.openai_completions":
        "0afb2771209f9261a2ec0146cbd01928f22747e994a25225c24fd046c5c138ff",
    "lm_eval.models.api_models":
        "8da25fde90c9a14ee7c603177a08814a3ac57aed7f18f6a00e040771335212cf",
    "lm_eval.tasks.ifeval.utils":
        "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632",
    "lm_eval.tasks.ifeval.instructions":
        "511cc41a53787d818c292d8335c8e98aa833296a0789ac470db56bcd6496345e",
    "lm_eval.tasks.ifeval.instructions_registry":
        "9db1c062cdb70a91420d3789a588235a1064c2b9a1a66e09e5fb9df9e7584a6c",
    "lm_eval.tasks.ifeval.instructions_util":
        "e8c4d9187bac1482d93941fb46469609c3ae78d896195bfcb300a0707d492567",
}
REQUEST_KEYS = {"messages", "model", "max_tokens", "temperature", "stop", "seed"}
THINK_MARKERS = ("<think>", "</think>")


def verify_installed_sources() -> None:
    """Fail on any drift from the audited code, before a server or model is involved."""
    import nltk

    # Importing the IFEval checkers downloads punkt_tab when it is missing. It is
    # baked into the image; refuse to reach that code path rather than fetch it.
    nltk.data.find("tokenizers/punkt_tab")
    for module_name, expected in SOURCE_SHA256.items():
        module = importlib.import_module(module_name)
        actual = hashlib.sha256(Path(str(module.__file__)).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Installed {module_name} differs from the audited source")


def verified_dataset_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) != DATASET_BYTES or hashlib.sha256(data).hexdigest() != DATASET_SHA256:
        raise ValueError("Pinned IFEval input file failed SHA256 verification")
    return data


def validate_task(config: dict[str, Any]) -> None:
    """The upstream v4.0 definition with only the dataset pinned to local bytes."""
    process = config.get("process_results")
    aggregate = (config.get("metric_list") or [{}, {}])[1].get("aggregation")
    named = config | {
        "process_results": _qualified(process),
        "metric_list": [
            item | ({"aggregation": _qualified(item["aggregation"])}
                    if callable(item.get("aggregation")) else {})
            for item in config.get("metric_list", [])
        ],
    }
    expected = {
        "task": "jaull-ifeval-chat-v1",
        "dataset_path": "json",
        "dataset_name": None,
        "dataset_kwargs": {"data_files": {"train": DATASET_URL}},
        "output_type": "generate_until",
        "test_split": "train",
        "num_fewshot": 0,
        "doc_to_text": "prompt",
        "doc_to_target": 0,
        "generation_kwargs": {
            "until": [], "do_sample": False, "temperature": 0.0, "max_gen_toks": MAX_GEN_TOKS,
        },
        "process_results": "lm_eval.tasks.ifeval.utils.process_results",
        "metric_list": [
            {"metric": "prompt_level_strict_acc", "aggregation": "mean",
             "higher_is_better": True},
            {"metric": "inst_level_strict_acc",
             "aggregation": "lm_eval.tasks.ifeval.utils.agg_inst_level_acc",
             "higher_is_better": True},
            {"metric": "prompt_level_loose_acc", "aggregation": "mean",
             "higher_is_better": True},
            {"metric": "inst_level_loose_acc",
             "aggregation": "lm_eval.tasks.ifeval.utils.agg_inst_level_acc",
             "higher_is_better": True},
        ],
        "metadata": {
            "version": 1.0,
            "upstream_task_version": 4.0,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
        },
    }
    if not callable(process) or not callable(aggregate) or named != expected:
        raise ValueError("Only the fixed zero-shot IFEval chat suite is supported")


def _qualified(function: Any) -> str:
    return f"{getattr(function, '__module__', '')}.{getattr(function, '__name__', '')}"


def validate_chat_request(payload: dict[str, Any]) -> None:
    """Exactly the audited payload: one user turn, greedy, full budget, no stops."""
    messages = payload.get("messages")
    if (
        set(payload) != REQUEST_KEYS
        or not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0]["role"] != "user"
        or not isinstance(messages[0]["content"], str)
        or not messages[0]["content"].strip()
        or payload["model"] != MODEL_NAME
        or type(payload["max_tokens"]) is not int
        or payload["max_tokens"] != MAX_GEN_TOKS
        or type(payload["temperature"]) not in (int, float)
        or payload["temperature"] != 0
        or payload["stop"] != []
        or type(payload["seed"]) is not int
        or payload["seed"] != SEED
    ):
        raise ValueError("Chat request differs from the audited IFEval protocol")


def validate_chat_response(response: dict[str, Any]) -> str:
    """Return finish_reason after checking runtime, reasoning, cache and context."""
    choices = response.get("choices")
    usage = response.get("usage") or {}
    if response.get("system_fingerprint") != RUNTIME_FINGERPRINT:
        raise ValueError("Response came from a runtime other than the audited build")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Chat response must carry exactly one choice")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str):
        raise ValueError("Chat response has no assistant text to score")
    if message.get("reasoning_content") or any(mark in content for mark in THINK_MARKERS):
        raise ValueError("Reasoning appeared although the protocol runs it off")
    finish = choices[0].get("finish_reason")
    if finish not in {"stop", "length"}:
        raise ValueError(f"Unexpected finish_reason {finish!r}")
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if (
        type(prompt_tokens) is not int or type(completion_tokens) is not int
        or prompt_tokens <= 0 or completion_tokens < 0
    ):
        raise ValueError("Chat response lacks valid token usage")
    if completion_tokens > MAX_GEN_TOKS or prompt_tokens + MAX_GEN_TOKS > CONTEXT:
        raise ValueError("Rendered prompt plus the generation budget exceeds the context")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens") != 0:
        raise ValueError("Prompt cache was used; the protocol requires it off")
    return finish


def validate_server_props(props: dict[str, Any], artifact: dict[str, Any]) -> str:
    """Hash the effective runtime template; the host checks embedded-template presence."""
    template = props.get("chat_template")
    if (
        props.get("total_slots") != 1
        or props.get("default_generation_settings", {}).get("n_ctx") != CONTEXT
        or props.get("model_path") != artifact["local_path"]
        or not isinstance(template, str)
        or not template.strip()
    ):
        raise ValueError("Server slots, context, model or chat template differ from the contract")
    return hashlib.sha256(template.encode()).hexdigest()


def summarise_generations(exchanges: list[dict[str, Any]]) -> dict[str, int]:
    """Truncated replies stay in the denominator; this only counts them."""
    finishes = [item["finish_reason"] for item in exchanges if "finish_reason" in item]
    return {"responses": len(finishes), "truncated": finishes.count("length")}


def run(output: Path, base_url: str, profile_name: str) -> None:
    profile = PROFILES[profile_name]
    task, sample_ids = profile["task"], profile["sample_ids"]
    import requests

    verify_installed_sources()
    from langdetect import DetectorFactory
    from lm_eval import simple_evaluate
    from lm_eval.models import openai_completions
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.utils import handle_non_serializable

    DetectorFactory.seed = 0
    suite = Path(__file__).with_name("suite_ifeval.yaml")
    config = load_yaml(suite, resolve_func=True)
    validate_task(config)
    config["task"] = task
    artifact = json.loads((output / "verified-artifact.json").read_text())
    validate_artifact_identity(artifact)
    with Path("/artifact/model.gguf").open("rb") as model:
        digest = hashlib.file_digest(model, "sha256").hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError("Read-only container artifact differs from the requested GGUF")
    write_json(output / "container-artifact.json", {"sha256": digest, "mount": "read-only"})
    props = requests.get(base_url + "/props", timeout=15)
    props.raise_for_status()
    props = props.json()
    write_json(output / "server-props.json", props)
    template_sha256 = validate_server_props(props, artifact)
    dataset = verified_dataset_bytes(Path("/dataset/ifeval.jsonl"))
    dataset_path = output / "dataset-ifeval.jsonl"
    with dataset_path.open("xb") as handle:
        handle.write(dataset)
    write_json(output / "dataset.json", {
        "repo": DATASET_REPO, "revision": DATASET_REVISION, "url": DATASET_URL,
        "sha256": DATASET_SHA256, "sample_ids": sample_ids,
    })
    write_json(output / "evaluator-config.json", {
        "evaluator_commit": EVALUATOR_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "sample_ids": sample_ids,
        "num_fewshot": 0,
        "context": CONTEXT,
        "apply_chat_template": True,
        "chat_template_source": "gguf",
        "chat_template_sha256": template_sha256,
        "reasoning": "off",
        "system_instruction": None,
        "generation": {"until": [], "temperature": 0, "max_gen_toks": MAX_GEN_TOKS,
                       "seed": SEED},
        "langdetect_seed": 0,
        "use_cache": None,
        "cache_requests": False,
        "python": platform.python_version(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    })
    config["dataset_kwargs"] = {"data_files": {"train": str(dataset_path)}}
    chat_url = base_url + "/v1/chat/completions"
    exchanges: list[dict[str, Any]] = []

    with (output / "http.jsonl").open("x", encoding="utf-8") as log:

        def record(entry: dict[str, Any]) -> None:
            exchanges.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()

        class CheckedChat(openai_completions.LocalChatCompletion):
            def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
                # No retries: a failed request is a failed run, never a silent repeat.
                payload = self._create_payload(
                    self.create_message(messages), generate=generate,
                    gen_kwargs=copy.deepcopy(gen_kwargs), seed=self._seed,
                    eos=self.eos_string, **kwargs,
                )
                entry: dict[str, Any] = {"url": chat_url, "request": payload}
                try:
                    validate_chat_request(payload)
                    rendered = requests.post(
                        base_url + "/apply-template", json={"messages": payload["messages"]},
                        timeout=15,
                    )
                    rendered.raise_for_status()
                    entry["rendered_prompt"] = rendered.json()["prompt"]
                    response = requests.post(chat_url, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    entry["response"] = response.json()
                    entry["finish_reason"] = validate_chat_response(entry["response"])
                    return entry["response"]
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    record(entry)

            def parse_generations(self, outputs, **kwargs):
                # Upstream turns an unparseable reply into "" and scores it.
                outputs = outputs if isinstance(outputs, list) else [outputs]
                return [validated["choices"][0]["message"]["content"] for validated in outputs]

        lm = CheckedChat(
            base_url=chat_url, model=MODEL_NAME, max_gen_toks=MAX_GEN_TOKS, seed=SEED,
            num_concurrent=1, max_retries=1, timeout=900, max_length=CONTEXT,
        )
        result = simple_evaluate(
            model=lm,
            tasks=[config],
            task_manager=TaskManager(include_path=str(suite.parent), include_defaults=False),
            num_fewshot=0,
            samples={task: sample_ids},
            log_samples=True,
            use_cache=None,
            cache_requests=False,
            apply_chat_template=True,
            system_instruction=None,
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
    if summarise_generations(exchanges)["responses"] != len(sample_ids):
        raise ValueError("Every selected prompt must have exactly one validated reply")
    with (output / "lm-eval-results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=handle_non_serializable)
        handle.write("\n")
    write_json(output / "generation-summary.json", summarise_generations(exchanges))
    write_json(output / "smoke-status.json",
               {"status": profile["status"], "quality_evidence": False})
