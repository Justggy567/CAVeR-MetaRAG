"""One authoritative routing implementation for RQ1, RQ2, and RQ3."""

from __future__ import annotations

from enum import Enum

from .labels import Verdict
from .schema import TriggerSignals, VerificationBundle


class RoutingPolicy(str, Enum):
    STRUCTURE_AWARE = "structure_aware"
    STRICT = "strict"
    HIGH_COVERAGE = "high_coverage"
    PARADOX_ONLY = "paradox_only"
    UNCERTAINTY_ONLY = "uncertainty_only"
    ANOMALY_COUNT = "anomaly_count"
    NO_ESCALATION = "no_escalation"
    FULL_ESCALATION = "full_escalation"
    STRUCTURE_AWARE_NO_UNCERTAINTY = "structure_aware_no_uncertainty"
    STRUCTURE_AWARE_NO_PARADOX = "structure_aware_no_paradox"
    STRUCTURE_AWARE_NO_TWO_ANTONYM = "structure_aware_no_two_antonym"
    STRUCTURE_AWARE_NO_CROSS_SIDE = "structure_aware_no_cross_side"


def reverse_verdict(verdict: Verdict) -> Verdict:
    if verdict is Verdict.YES:
        return Verdict.NO
    if verdict is Verdict.NO:
        return Verdict.YES
    raise ValueError("NOT_SURE has no deterministic reverse verdict.")


def compute_signals(bundle: VerificationBundle) -> TriggerSignals:
    positive = (bundle.original, *bundle.synonyms)
    positive_certain = {item for item in positive if item is not Verdict.NOT_SURE}
    negative_certain = {item for item in bundle.antonyms if item is not Verdict.NOT_SURE}
    explicit_paradox = bool(positive_certain.intersection(negative_certain))

    if bundle.original is Verdict.NOT_SURE:
        synonym_anomalies = 0
        antonym_anomalies = 0
    else:
        synonym_anomalies = sum(item is not bundle.original for item in bundle.synonyms)
        expected_antonym = reverse_verdict(bundle.original)
        antonym_anomalies = sum(item is not expected_antonym for item in bundle.antonyms)

    return TriggerSignals(
        original_uncertainty=bundle.original is Verdict.NOT_SURE,
        explicit_paradox=explicit_paradox,
        synonym_anomaly_count=synonym_anomalies,
        antonym_anomaly_count=antonym_anomalies,
        two_antonym_anomalies=antonym_anomalies >= 2,
        cross_side_anomaly=synonym_anomalies >= 1 and antonym_anomalies >= 1,
    )


def should_escalate(
    signals: TriggerSignals,
    policy: RoutingPolicy = RoutingPolicy.STRUCTURE_AWARE,
    anomaly_count_threshold: int = 2,
) -> bool:
    if policy is RoutingPolicy.NO_ESCALATION:
        return False
    if policy is RoutingPolicy.FULL_ESCALATION:
        return True
    if policy is RoutingPolicy.PARADOX_ONLY:
        return signals.explicit_paradox
    if policy is RoutingPolicy.UNCERTAINTY_ONLY:
        return signals.original_uncertainty
    if policy is RoutingPolicy.ANOMALY_COUNT:
        return (
            signals.original_uncertainty
            or signals.synonym_anomaly_count + signals.antonym_anomaly_count >= anomaly_count_threshold
        )
    if policy in {RoutingPolicy.STRICT, RoutingPolicy.HIGH_COVERAGE}:
        return (
            signals.original_uncertainty
            or signals.explicit_paradox
            or signals.synonym_anomaly_count > 0
            or signals.antonym_anomaly_count > 0
        )
    structure_conditions = {
        "original_uncertainty": signals.original_uncertainty,
        "explicit_paradox": signals.explicit_paradox,
        "two_antonym_anomalies": signals.two_antonym_anomalies,
        "cross_side_anomaly": signals.cross_side_anomaly,
    }
    exclusions = {
        RoutingPolicy.STRUCTURE_AWARE: None,
        RoutingPolicy.STRUCTURE_AWARE_NO_UNCERTAINTY: "original_uncertainty",
        RoutingPolicy.STRUCTURE_AWARE_NO_PARADOX: "explicit_paradox",
        RoutingPolicy.STRUCTURE_AWARE_NO_TWO_ANTONYM: "two_antonym_anomalies",
        RoutingPolicy.STRUCTURE_AWARE_NO_CROSS_SIDE: "cross_side_anomaly",
    }
    if policy in exclusions:
        excluded = exclusions[policy]
        return any(value for name, value in structure_conditions.items() if name != excluded)
    raise ValueError(f"Unsupported routing policy: {policy}")


def routing_reasons(
    signals: TriggerSignals,
    policy: RoutingPolicy,
    *,
    anomaly_count_threshold: int = 2,
) -> tuple[str, ...]:
    """Return the real reason for upgrading the specified policy, avoiding mistakenly triggering across policies."""

    if not should_escalate(signals, policy, anomaly_count_threshold=anomaly_count_threshold):
        return ()
    if policy is RoutingPolicy.FULL_ESCALATION:
        return ("full_escalation",)
    if policy is RoutingPolicy.PARADOX_ONLY:
        return ("explicit_paradox",)
    if policy is RoutingPolicy.UNCERTAINTY_ONLY:
        return ("original_uncertainty",)
    if policy is RoutingPolicy.ANOMALY_COUNT:
        reasons = []
        if signals.original_uncertainty:
            reasons.append("original_uncertainty")
        if signals.synonym_anomaly_count + signals.antonym_anomaly_count >= anomaly_count_threshold:
            reasons.append("anomaly_count_threshold")
        return tuple(reasons)
    if policy in {RoutingPolicy.STRICT, RoutingPolicy.HIGH_COVERAGE}:
        reasons = list(signals.active_names())
        if signals.synonym_anomaly_count:
            reasons.append("any_synonym_anomaly")
        if signals.antonym_anomaly_count:
            reasons.append("any_antonym_anomaly")
        return tuple(dict.fromkeys(reasons))
    exclusions = {
        RoutingPolicy.STRUCTURE_AWARE: None,
        RoutingPolicy.STRUCTURE_AWARE_NO_UNCERTAINTY: "original_uncertainty",
        RoutingPolicy.STRUCTURE_AWARE_NO_PARADOX: "explicit_paradox",
        RoutingPolicy.STRUCTURE_AWARE_NO_TWO_ANTONYM: "two_antonym_anomalies",
        RoutingPolicy.STRUCTURE_AWARE_NO_CROSS_SIDE: "cross_side_anomaly",
    }
    excluded = exclusions.get(policy)
    if policy in exclusions:
        return tuple(name for name in signals.active_names() if name != excluded)
    raise ValueError(f"Unsupported routing policy: {policy}")
