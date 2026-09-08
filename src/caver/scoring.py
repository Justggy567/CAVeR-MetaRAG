"""Frozen scoring and response aggregation used by every experiment."""

from __future__ import annotations

from collections.abc import Iterable

from .labels import Verdict
from .schema import VerificationBundle


POSITIVE_PENALTY = {
    Verdict.YES: 0.0,
    Verdict.NOT_SURE: 0.5,
    Verdict.NO: 1.0,
}
NEGATIVE_PENALTY = {
    Verdict.NO: 0.0,
    Verdict.NOT_SURE: 0.5,
    Verdict.YES: 1.0,
}
DEFAULT_THRESHOLD = 0.5


def score_verification(bundle: VerificationBundle) -> float:
    penalties = [
        POSITIVE_PENALTY[bundle.original],
        *(POSITIVE_PENALTY[item] for item in bundle.synonyms),
        *(NEGATIVE_PENALTY[item] for item in bundle.antonyms),
    ]
    return sum(penalties) / len(penalties)


def aggregate_response_scores(scores: Iterable[float]) -> float:
    values = tuple(float(value) for value in scores)
    if not values:
        raise ValueError("Cannot aggregate an empty factoid score collection.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Factoid scores must be in [0, 1].")
    return max(values)


def predicts_hallucination(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    if not 0.0 <= score <= 1.0:
        raise ValueError("Score must be in [0, 1].")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")
    return score >= threshold
