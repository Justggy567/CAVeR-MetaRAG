"""Canonical CAVeR-MetaRAG implementation.

All modified entry points for the RQ1, RQ2, and RQ3 experiments must import method logic from this package.
"""

from .labels import BinaryTarget, EvidenceRelation, Verdict, parse_verdict
from .routing import RoutingPolicy, compute_signals, should_escalate
from .scoring import aggregate_response_scores, score_verification
from .schema import ProbeSet, VerificationBundle

__all__ = [
    "BinaryTarget",
    "EvidenceRelation",
    "ProbeSet",
    "RoutingPolicy",
    "Verdict",
    "VerificationBundle",
    "aggregate_response_scores",
    "compute_signals",
    "parse_verdict",
    "score_verification",
    "should_escalate",
]
