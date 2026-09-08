"""Offline routing controls that reuse the same frozen verifier ledger."""

from __future__ import annotations

import random
from typing import Iterable, Mapping

from .routing import RoutingPolicy, should_escalate
from .schema import TriggerSignals


def route_from_signals(
    signals_by_id: Mapping[str, TriggerSignals],
    policy: RoutingPolicy,
    *,
    anomaly_count_threshold: int = 2,
) -> set[str]:
    return {
        item_id
        for item_id, signals in signals_by_id.items()
        if should_escalate(signals, policy, anomaly_count_threshold=anomaly_count_threshold)
    }


def random_matched_count(item_ids: Iterable[str], target_count: int, *, seed: int) -> set[str]:
    items = sorted({str(item) for item in item_ids})
    if target_count < 0 or target_count > len(items):
        raise ValueError("target_count must be between zero and the number of unique items.")
    generator = random.Random(seed)
    return set(generator.sample(items, target_count))


def random_matched_budget(
    cost_by_id: Mapping[str, int],
    target_budget: int,
    *,
    seed: int,
) -> set[str]:
    if target_budget < 0:
        raise ValueError("target_budget cannot be negative.")
    if any(cost < 0 for cost in cost_by_id.values()):
        raise ValueError("Item costs cannot be negative.")
    items = sorted(cost_by_id)
    random.Random(seed).shuffle(items)
    selected: set[str] = set()
    used = 0
    for item_id in items:
        cost = int(cost_by_id[item_id])
        if used + cost <= target_budget:
            selected.add(item_id)
            used += cost
    return selected


def stage1_feature_dict(signals: TriggerSignals) -> dict[str, int]:
    """Deployable features for a later held-out learned router."""

    return {
        "original_uncertainty": int(signals.original_uncertainty),
        "explicit_paradox": int(signals.explicit_paradox),
        "synonym_anomaly_count": signals.synonym_anomaly_count,
        "antonym_anomaly_count": signals.antonym_anomaly_count,
        "two_antonym_anomalies": int(signals.two_antonym_anomalies),
        "cross_side_anomaly": int(signals.cross_side_anomaly),
    }
