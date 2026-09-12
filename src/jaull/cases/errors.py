"""Failures of the case store.

Deliberately narrow: these are *parsing and persistence* failures. A manifest
that references a record which does not exist is not an error here — it is an
``INCONSISTENT`` validation result with a concrete reason. Keeping the two apart
is what lets a broken reference be reported instead of crashing a listing.
"""

from __future__ import annotations

from jaull.exceptions import JaullError


class CaseStoreError(JaullError):
    """Base class for case store failures."""


class CaseManifestNotFoundError(CaseStoreError):
    """The requested case manifest does not exist."""


class InvalidCaseIdError(CaseStoreError):
    """The case id cannot be mapped safely to a local path."""


__all__ = [
    "CaseManifestNotFoundError",
    "CaseStoreError",
    "InvalidCaseIdError",
]
