from __future__ import annotations

from jaull.discovery.models import EvaluatedCandidate, ModelCandidate
from jaull.discovery.series import (
    group_by_series,
    parameter_label,
    series_key,
)


def _e(repo_id: str) -> EvaluatedCandidate:
    return EvaluatedCandidate(candidate=ModelCandidate(repo_id=repo_id))


def test_same_family_different_sizes_share_a_series_key() -> None:
    small = series_key(_e("Qwen/Qwen2.5-0.5B-Instruct"))
    big = series_key(_e("Qwen/Qwen2.5-7B-Instruct"))
    assert small == big


def test_different_variants_do_not_merge() -> None:
    base = series_key(_e("meta-llama/Meta-Llama-3.1-8B"))
    instruct = series_key(_e("meta-llama/Meta-Llama-3.1-8B-Instruct"))
    coder = series_key(_e("meta-llama/Meta-Llama-3.1-8B-Coder"))
    assert base != instruct
    assert instruct != coder
    assert base != coder


def test_unknown_family_gets_solo_key() -> None:
    key = series_key(_e("random-user/wat-model"))
    assert key.startswith("solo:")


def test_group_by_series_buckets_by_size() -> None:
    candidates = [
        _e("Qwen/Qwen2.5-0.5B-Instruct"),
        _e("Qwen/Qwen2.5-1.5B-Instruct"),
        _e("Qwen/Qwen2.5-3B-Instruct"),
        _e("Qwen/Qwen2.5-7B-Instruct"),
    ]
    buckets = group_by_series(candidates)
    assert len(buckets) == 1
    assert len(next(iter(buckets.values()))) == 4


def test_parameter_label_formats_billion_counts() -> None:
    assert parameter_label(500_000_000) == "500M"
    assert parameter_label(1_500_000_000) == "1.5B"
    assert parameter_label(7_000_000_000) == "7B"
    assert parameter_label(None) == "unknown size"
