"""Replay an exported inspection shortlist using only its recorded inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jaull.discovery.candidate_filter import shortlist
from jaull.domain.candidates import ModelCandidate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.requirements import UserRequirements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    audit = report.get("search_strategy", {}).get("inspection_audit")
    if not audit or audit.get("version") != 1:
        parser.error("Report has no supported inspection audit; generate a new report.")
    rows = audit["candidates"]
    replay: dict[str, Any] = {}
    selected = shortlist(
        [ModelCandidate.model_validate(row["input"]) for row in rows.values()],
        UserRequirements.model_validate(audit["requirements"]),
        audit["limit"],
        hardware=HardwareProfile.model_validate(audit["hardware"]),
        targets_other_task=lambda candidate: rows[candidate.repo_id]["other_task"],
        audit=replay,
    )
    expected = [step["repo_id"] for step in audit["steps"]]
    actual = [candidate.repo_id for candidate in selected]
    # JSON normalises tuples in the version keys before comparing decision traces.
    trace_matches = json.dumps(replay, sort_keys=True) == json.dumps(audit, sort_keys=True)
    print(json.dumps({
        "selection_matches": actual == expected,
        "trace_matches": trace_matches,
        "expected": expected,
        "replayed": actual,
    }, indent=2))
    return 0 if actual == expected and trace_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
