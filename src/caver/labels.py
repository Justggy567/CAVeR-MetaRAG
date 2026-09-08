"""Three-way evidence labels and their explicit binary evaluation mapping."""

from __future__ import annotations

import re
from enum import Enum


class Verdict(str, Enum):
    YES = "YES"
    NO = "NO"
    NOT_SURE = "NOT_SURE"


class EvidenceRelation(str, Enum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class BinaryTarget(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class VerdictParseError(ValueError):
    """Raised when a verifier response has no unambiguous leading verdict."""


_LEADING_LABEL = re.compile(
    r"^\s*(?:ANSWER\s*:\s*)?(NOT[\s_-]*SURE|YES|NO)(?=\s|[.:\-,;]|$)",
    flags=re.IGNORECASE,
)
_ANY_LABEL = re.compile(
    r"(?<![A-Z0-9_])(NOT[\s_-]*SURE|YES|NO)(?![A-Z0-9_])",
    flags=re.IGNORECASE,
)


def _normalize_label(label: str) -> Verdict:
    normalized = re.sub(r"[\s_-]+", "_", label.strip().upper())
    if normalized == "NOT_SURE":
        return Verdict.NOT_SURE
    return Verdict(normalized)


def parse_verdict(text: str) -> Verdict:
    """Avoid substring-based YES/NO errors when parsing validator responses.

The required format is a leading label, optionally prefixed with "Answer:".
A compatibility fallback accepts labels in other positions only when exactly one distinct verdict appears. Ambiguous or empty responses fail explicitly rather than silently defaulting to NOT_SURE.
    """

    if not isinstance(text, str) or not text.strip():
        raise VerdictParseError("Verifier response is empty.")

    match = _LEADING_LABEL.search(text)
    if match:
        return _normalize_label(match.group(1))

    labels = {_normalize_label(item) for item in _ANY_LABEL.findall(text)}
    if len(labels) == 1:
        return next(iter(labels))
    if not labels:
        raise VerdictParseError(f"No verdict label found in response: {text!r}")
    raise VerdictParseError(f"Ambiguous verifier response: {text!r}")


def verdict_to_relation(verdict: Verdict) -> EvidenceRelation:
    return {
        Verdict.YES: EvidenceRelation.ENTAILED,
        Verdict.NO: EvidenceRelation.CONTRADICTED,
        Verdict.NOT_SURE: EvidenceRelation.NOT_ENOUGH_INFO,
    }[verdict]


def relation_to_binary(relation: EvidenceRelation) -> BinaryTarget:
    if relation is EvidenceRelation.ENTAILED:
        return BinaryTarget.SUPPORTED
    return BinaryTarget.UNSUPPORTED


def supported_bool_to_binary(supported: bool) -> BinaryTarget:
    if not isinstance(supported, bool):
        raise TypeError("Gold support must be a bool, not a truthy/falsy value.")
    return BinaryTarget.SUPPORTED if supported else BinaryTarget.UNSUPPORTED
