"""Guard: the test suite must never reach Hugging Face or the network.

CI runs without credentials and often without egress, and a test that silently
depends on a live API is a flaky test that also makes the suite slow. This
module enforces the rule structurally rather than by convention.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Modules that legitimately reference the SDK because they test our *mapping*
# of its error types, using stub objects rather than real calls.
_ALLOWED_HF_IMPORTS = {"test_discovery_search.py"}

_REAL_CALL_PATTERNS = (
    re.compile(r"\bHfApi\s*\("),
    re.compile(r"\bhf_hub_download\s*\("),
    re.compile(r"\brequests\.(get|post)\s*\("),
    re.compile(r"\bhttpx\.(get|post)\s*\("),
    re.compile(r"\burlopen\s*\("),
)


def _test_modules() -> list[Path]:
    return sorted(TESTS_DIR.glob("*.py"))


def test_no_test_module_instantiates_a_real_hub_client() -> None:
    offenders: list[str] = []
    for path in _test_modules():
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in _REAL_CALL_PATTERNS:
            if pattern.search(source):
                offenders.append(f"{path.name}: {pattern.pattern}")
    assert offenders == [], offenders


def test_hugging_face_sdk_imports_are_limited_to_error_mapping_tests() -> None:
    offenders: list[str] = []
    for path in _test_modules():
        if path.name in _ALLOWED_HF_IMPORTS or path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import huggingface_hub", "from huggingface_hub")):
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == [], offenders


def test_guarded_modules_still_exist() -> None:
    """Fail loudly if the allowlist drifts away from reality."""
    names = {path.name for path in _test_modules()}
    assert names >= _ALLOWED_HF_IMPORTS
