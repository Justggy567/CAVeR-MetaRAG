

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .labels import BinaryTarget
from .schema import ResponseResult


def defined_rate(numerator: int, denominator: int) -> float | None:
    """Returns the auditable proportion; a zero denominator indicates that it is not estimable."""

    return numerator / denominator if denominator else None


def transition_metrics_from_counts(
    *,
    rescued: int,
    harmed: int,
    unchanged_correct: int,
    unchanged_wrong: int,
    routed_responses: int,
    routed_stage1_wrong: int,
) -> dict[str, object]:


    counts = {
        "rescued": rescued,
        "harmed": harmed,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "routed_responses": routed_responses,
        "routed_stage1_wrong": routed_stage1_wrong,
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("Transition counts must be non-negative.")
    total = rescued + harmed + unchanged_correct + unchanged_wrong
    if routed_responses > total:
        raise AssertionError("Routed responses cannot exceed total responses.")
    if routed_stage1_wrong > routed_responses:
        raise AssertionError("Routed Stage-1 errors cannot exceed routed responses.")
    if rescued > routed_stage1_wrong:
        raise AssertionError("A rescued response must be a routed Stage-1 error.")
    if harmed > routed_responses - routed_stage1_wrong:
        raise AssertionError("A harmed response must be a routed Stage-1-correct response.")

    rescue_rate_response = defined_rate(rescued, routed_responses)
    harm_rate_response = defined_rate(harmed, routed_responses)
    error_conditional_rescue_rate = defined_rate(rescued, routed_stage1_wrong)
    rescue_rate_all_responses = defined_rate(rescued, total)
    harm_rate_all_responses = defined_rate(harmed, total)
    undefined_rates = [
        name
        for name, value in (
            ("rescue_rate_response", rescue_rate_response),
            ("harm_rate_response", harm_rate_response),
            ("error_conditional_rescue_rate", error_conditional_rescue_rate),
            ("rescue_rate_all_responses", rescue_rate_all_responses),
            ("harm_rate_all_responses", harm_rate_all_responses),
        )
        if value is None
    ]
    return {
        **counts,
        "n": total,
        "net_correct_gain": rescued - harmed,
        "rescue_rate_response": rescue_rate_response,
        "harm_rate_response": harm_rate_response,
        "error_conditional_rescue_rate": error_conditional_rescue_rate,
        "rescue_rate_all_responses": rescue_rate_all_responses,
        "harm_rate_all_responses": harm_rate_all_responses,
        "rescue_rate_among_routed": rescue_rate_response,
        "harm_rate_among_routed": harm_rate_response,
        "error_conditional_rescue": error_conditional_rescue_rate,
        "overall_rescue_rate": rescue_rate_all_responses,
        "overall_harm_rate": harm_rate_all_responses,
        "rate_denominators": {
            "rescue_rate_response": routed_responses,
            "harm_rate_response": routed_responses,
            "error_conditional_rescue_rate": routed_stage1_wrong,
            "rescue_rate_all_responses": total,
            "harm_rate_all_responses": total,
        },
        "undefined_rates": undefined_rates,
    }


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, *, gold_hallucinated: bool, predicted_hallucinated: bool) -> None:
        if gold_hallucinated and predicted_hallucinated:
            self.tp += 1
        elif not gold_hallucinated and predicted_hallucinated:
            self.fp += 1
        elif not gold_hallucinated and not predicted_hallucinated:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def metrics(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (self.tp + self.tn) / self.total if self.total else 0.0
        return {
            **asdict(self),
            "n": self.total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


@dataclass
class TransitionCounts:
    rescued: int = 0
    harmed: int = 0
    unchanged_correct: int = 0
    unchanged_wrong: int = 0
    routed_responses: int = 0
    routed_stage1_wrong: int = 0

    def metrics(self) -> dict[str, object]:
        return transition_metrics_from_counts(**asdict(self))


def _is_gold_hallucinated(target: BinaryTarget) -> bool:
    return target is BinaryTarget.UNSUPPORTED


def summarize_results(results: Iterable[ResponseResult]) -> dict[str, object]:
    stage1_confusion = ConfusionMatrix()
    final_confusion = ConfusionMatrix()
    transitions = TransitionCounts()
    total_factoids = 0
    routed_factoids = 0

    materialized = tuple(results)
    for result in materialized:
        gold = _is_gold_hallucinated(result.gold_target)
        stage1_confusion.update(gold_hallucinated=gold, predicted_hallucinated=result.stage1_hallucinated)
        final_confusion.update(gold_hallucinated=gold, predicted_hallucinated=result.final_hallucinated)
        stage1_correct = result.stage1_hallucinated == gold
        final_correct = result.final_hallucinated == gold
        routed = any(item.routed for item in result.factoids)
        if not routed and result.stage1_hallucinated != result.final_hallucinated:
            raise AssertionError("A non-routed response cannot change its final prediction.")
        total_factoids += len(result.factoids)
        routed_factoids += sum(item.routed for item in result.factoids)
        if routed:
            transitions.routed_responses += 1
            if not stage1_correct:
                transitions.routed_stage1_wrong += 1
        if not stage1_correct and final_correct:
            transitions.rescued += 1
        elif stage1_correct and not final_correct:
            transitions.harmed += 1
        elif stage1_correct:
            transitions.unchanged_correct += 1
        else:
            transitions.unchanged_wrong += 1

    stage1 = stage1_confusion.metrics()
    final = final_confusion.metrics()
    transition = transitions.metrics()
    observed_gain = int(final["tp"]) + int(final["tn"]) - int(stage1["tp"]) - int(stage1["tn"])
    if observed_gain != transition["net_correct_gain"]:
        raise AssertionError("Rescue/harm invariant failed.")
    return {
        "stage1": stage1,
        "final": final,
        "transitions": transition,
        "routing": {
            "response_escalation_rate": defined_rate(
                transitions.routed_responses, len(materialized)
            ),
            "factoid_escalation_rate": defined_rate(routed_factoids, total_factoids),
            "routed_responses": transitions.routed_responses,
            "total_responses": len(materialized),
            "routed_factoids": routed_factoids,
            "total_factoids": total_factoids,
        },
    }


def summarize_by_metadata(
    results: Iterable[ResponseResult],
    *,
    key: str,
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[ResponseResult]] = {}
    for result in results:
        value = result.metadata.get(key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(result)
    return {name: summarize_results(items) for name, items in sorted(groups.items())}


def summarize_serialized(
    responses: Iterable[dict[str, object]],
    factoids: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Reconstruct the complete summary from the append-only ledger to enable unified aggregation after resuming from an interruption."""

    response_rows = tuple(responses)
    factoid_rows = tuple(factoids)
    factoids_by_response: dict[str, list[dict[str, object]]] = {}
    for factoid in factoid_rows:
        factoids_by_response.setdefault(str(factoid["response_id"]), []).append(factoid)
    stage1_confusion = ConfusionMatrix()
    final_confusion = ConfusionMatrix()
    transitions = TransitionCounts()
    routed_factoids = 0
    for response in response_rows:
        response_id = str(response["response_id"])
        gold = str(response["gold_target"]) == BinaryTarget.UNSUPPORTED.value
        stage1_prediction = bool(response["stage1_hallucinated"])
        final_prediction = bool(response["final_hallucinated"])
        stage1_confusion.update(gold_hallucinated=gold, predicted_hallucinated=stage1_prediction)
        final_confusion.update(gold_hallucinated=gold, predicted_hallucinated=final_prediction)
        response_factoids = factoids_by_response.get(response_id, [])
        routed = any(bool(item["routed"]) for item in response_factoids)
        if not routed and stage1_prediction != final_prediction:
            raise AssertionError("A non-routed serialized response changed prediction.")
        routed_factoids += sum(bool(item["routed"]) for item in response_factoids)
        stage1_correct = stage1_prediction == gold
        final_correct = final_prediction == gold
        if routed:
            transitions.routed_responses += 1
            if not stage1_correct:
                transitions.routed_stage1_wrong += 1
        if not stage1_correct and final_correct:
            transitions.rescued += 1
        elif stage1_correct and not final_correct:
            transitions.harmed += 1
        elif stage1_correct:
            transitions.unchanged_correct += 1
        else:
            transitions.unchanged_wrong += 1
    stage1 = stage1_confusion.metrics()
    final = final_confusion.metrics()
    transition = transitions.metrics()
    if int(final["tp"]) + int(final["tn"]) - int(stage1["tp"]) - int(stage1["tn"]) != transition["net_correct_gain"]:
        raise AssertionError("Serialized rescue/harm invariant failed.")
    total_factoids = len(factoid_rows)
    return {
        "stage1": stage1,
        "final": final,
        "transitions": transition,
        "routing": {
            "response_escalation_rate": defined_rate(
                transitions.routed_responses, len(response_rows)
            ),
            "factoid_escalation_rate": defined_rate(routed_factoids, total_factoids),
            "routed_responses": transitions.routed_responses,
            "total_responses": len(response_rows),
            "routed_factoids": routed_factoids,
            "total_factoids": total_factoids,
        },
    }
