"""Public derivatives retain explicit provenance and truthful bundle status."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from jaull.cases.bundle import CaseBundleService
from jaull.cases.errors import CaseBundleError


def test_public_4060_derivatives() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads((root / "validation/public-anonymization.json").read_text())
    assert ledger["publication_type"] == "anonymized_derivative"
    assert ledger["changed_files"]
    for entry in ledger["changed_files"]:
        data = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["public_sha256"]
        assert entry["original_sha256"] != entry["public_sha256"]
    for path in (root / "validation").glob("rtx4060-*/**/*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
        text = data.decode(encoding, errors="replace")
        assert not re.search(r"ROGERLLOP|(?:\\+|/)Users(?:\\+|/)PC\b", text, re.I), path

    def unused_loader(_: str):
        raise AssertionError("Offline bundles must not depend on local stores")

    service = CaseBundleService(unused_loader, unused_loader)
    for bundle in ledger["bundles"]:
        path = root / bundle["path"]
        if bundle["original_file_integrity"] == "invalid":
            with pytest.raises(CaseBundleError):
                service.load(path)
        else:
            assert service.validate(path).status.value == "valid"
