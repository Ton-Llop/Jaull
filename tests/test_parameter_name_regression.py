import pytest

from jaull.domain.candidates import ModelCandidate
from jaull.domain.families import resolve_parameter_count


@pytest.mark.parametrize(("name", "expected"), [
    ("org/Llama-3.2-8X3B-MOE-Dark-Champion-18.4B-GGUF", 18_400_000_000),
    ("org/Model-8x7B-GGUF", None),
    ("org/Model-30B-A3B-GGUF", 30_000_000_000),
    ("org/Model-A3B-30B-GGUF", 30_000_000_000),
    ("org/Model-135M", 135_000_000),
    ("org/Model-7_B", 7_000_000_000),
    ("owner-70B/Model-9B", 9_000_000_000),
])
def test_name_inference_does_not_confuse_expert_active_and_total_sizes(name, expected):
    result = resolve_parameter_count(ModelCandidate(repo_id=name), None)
    assert result.count == expected
