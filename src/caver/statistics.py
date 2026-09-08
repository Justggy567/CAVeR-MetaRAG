"""Question-clustered paired uncertainty estimates without extra dependencies."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedPrediction:
    question_id: str
    gold_hallucinated: bool
    first_hallucinated: bool
    second_hallucinated: bool


def _accuracy(rows: Sequence[PairedPrediction], field: str) -> float:
    if not rows:
        raise ValueError("Cannot score an empty sample.")
    correct = sum(getattr(row, field) == row.gold_hallucinated for row in rows)
    return correct / len(rows)


def _f1(rows: Sequence[PairedPrediction], field: str) -> float:
    tp = fp = fn = 0
    for row in rows:
        predicted = bool(getattr(row, field))
        if row.gold_hallucinated and predicted:
            tp += 1
        elif not row.gold_hallucinated and predicted:
            fp += 1
        elif row.gold_hallucinated and not predicted:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty collection.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_paired_difference(
    records: Iterable[PairedPrediction],
    *,
    metric: str,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    rows = tuple(records)
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions.")
    metric_function: Callable[[Sequence[PairedPrediction], str], float]
    if metric == "accuracy":
        metric_function = _accuracy
    elif metric == "f1":
        metric_function = _f1
    else:
        raise ValueError("metric must be accuracy or f1.")
    clusters: dict[str, list[PairedPrediction]] = defaultdict(list)
    for row in rows:
        clusters[row.question_id].append(row)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        raise ValueError("No clusters supplied.")
    generator = random.Random(seed)
    differences: list[float] = []
    for _ in range(repetitions):
        sampled: list[PairedPrediction] = []
        for cluster_id in generator.choices(cluster_ids, k=len(cluster_ids)):
            sampled.extend(clusters[cluster_id])
        differences.append(
            metric_function(sampled, "second_hallucinated")
            - metric_function(sampled, "first_hallucinated")
        )
    observed = metric_function(rows, "second_hallucinated") - metric_function(rows, "first_hallucinated")
    return {
        "difference": observed,
        "ci_low": _quantile(differences, 0.025),
        "ci_high": _quantile(differences, 0.975),
        "clusters": len(cluster_ids),
        "responses": len(rows),
        "repetitions": repetitions,
        "seed": seed,
    }


def mcnemar_exact(records: Iterable[PairedPrediction]) -> dict[str, float | int]:
    first_only = 0
    second_only = 0
    for row in records:
        first_correct = row.first_hallucinated == row.gold_hallucinated
        second_correct = row.second_hallucinated == row.gold_hallucinated
        if first_correct and not second_correct:
            first_only += 1
        elif second_correct and not first_correct:
            second_only += 1
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant": discordant,
        "p_value_two_sided_exact": p_value,
    }
