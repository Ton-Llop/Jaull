from __future__ import annotations

import json
import logging

from jaull.analyzers.base import AnalyzerResult
from jaull.analyzers.generic import collect_relevant_files
from jaull.domain.model import (
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.exceptions import ConfigurationNotFoundError, JaullError
from jaull.huggingface.client import HfClientProtocol

logger = logging.getLogger(__name__)


class TransformersAnalyzer:
    def analyze(
        self,
        repo: ModelRepositoryInfo,
        files: list[ModelFile],
        classification: RepositoryClassification,
        client: HfClientProtocol,
    ) -> AnalyzerResult:
        del classification  # not needed at this depth
        warnings: list[str] = []
        config: ModelConfig | None = None

        try:
            local = client.download_small_file(repo.repo_id, "config.json")
            data = json.loads(local.read_text(encoding="utf-8"))
            config = _model_config_from_dict(data)
        except ConfigurationNotFoundError:
            warnings.append("config.json is missing; transformer metadata is unavailable.")
        except JaullError as exc:
            warnings.append(f"Could not fetch config.json: {exc}")
        except (OSError, ValueError) as exc:
            logger.debug("config.json parse failed", exc_info=exc)
            warnings.append("config.json is present but could not be parsed.")

        return AnalyzerResult(
            config=config,
            relevant_files=collect_relevant_files(files),
            warnings=warnings,
        )


_RAW_FLAG_KEYS = (
    # Mixture of experts hints
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    # Deepseek-style Multi-head Latent Attention
    "q_lora_rank",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    # Custom architectures loaded via trust_remote_code
    "auto_map",
    # Multimodal composite configs
    "text_config",
    "vision_config",
    "audio_config",
)


def _model_config_from_dict(data: dict[str, object]) -> ModelConfig:
    raw_architectures = data.get("architectures")
    architectures = (
        [str(x) for x in raw_architectures] if isinstance(raw_architectures, list) else []
    )

    def _int_or_none(key: str) -> int | None:
        value = data.get(key)
        return int(value) if isinstance(value, int) else None

    def _str_or_none(key: str) -> str | None:
        value = data.get(key)
        return str(value) if isinstance(value, str) else None

    def _bool_or_none(key: str) -> bool | None:
        value = data.get(key)
        return bool(value) if isinstance(value, bool) else None

    def _dict_or_none(key: str) -> dict[str, object] | None:
        value = data.get(key)
        return dict(value) if isinstance(value, dict) else None

    raw_flags: dict[str, object] = {
        key: data[key] for key in _RAW_FLAG_KEYS if key in data
    }

    return ModelConfig(
        architectures=architectures,
        model_type=_str_or_none("model_type"),
        torch_dtype=_str_or_none("torch_dtype"),
        max_position_embeddings=_int_or_none("max_position_embeddings"),
        hidden_size=_int_or_none("hidden_size"),
        num_hidden_layers=_int_or_none("num_hidden_layers"),
        num_attention_heads=_int_or_none("num_attention_heads"),
        num_key_value_heads=_int_or_none("num_key_value_heads"),
        head_dim=_int_or_none("head_dim"),
        intermediate_size=_int_or_none("intermediate_size"),
        sliding_window=_int_or_none("sliding_window"),
        rope_scaling=_dict_or_none("rope_scaling"),
        tie_word_embeddings=_bool_or_none("tie_word_embeddings"),
        vocab_size=_int_or_none("vocab_size"),
        quantization_config=_dict_or_none("quantization_config"),
        raw_flags=raw_flags,
    )
