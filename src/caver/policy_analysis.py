"""Replay routing strategies offline using the same complete Stage-1/Stage-2 ledger.

This module must be used for formal RQ2/RQ3 strategy comparisons to avoid introducing probe effects or model output drift caused by re-invoking the model for each strategy. The input matrix run must include Stage 2 data for every factoid.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baselines import random_matched_budget, random_matched_count, stage1_feature_dict
from .labels import BinaryTarget
from .ledger import atomic_write_json
from .metrics import defined_rate, transition_metrics_from_counts
from .routing import RoutingPolicy, should_escalate
from .schema import TriggerSignals
from .scoring import aggregate_response_scores, predicts_hallucination
from .statistics import PairedPrediction, clustered_paired_difference, mcnemar_exact


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _signals(row: Mapping[str, Any]) -> TriggerSignals:
    value = row["signals"]
    return TriggerSignals(
        original_uncertainty=bool(value["original_uncertainty"]),
        explicit_paradox=bool(value["explicit_paradox"]),
        synonym_anomaly_count=int(value["synonym_anomaly_count"]),
        antonym_anomaly_count=int(value["antonym_anomaly_count"]),
        two_antonym_anomalies=bool(value["two_antonym_anomalies"]),
        cross_side_anomaly=bool(value["cross_side_anomaly"]),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _confusion(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        gold = bool(row["gold_hallucinated"])
        predicted = bool(row["final_hallucinated"])
        if gold and predicted:
            tp += 1
        elif not gold and predicted:
            fp += 1
        elif not gold and not predicted:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap clustered by source question."""

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_id"])].append(row)
    cluster_ids = sorted(clusters)
    generator = random.Random(seed)
    metric_names = ("accuracy", "precision", "recall", "f1")
    resource_names = (
        "cloud_calls",
        "local_calls",
        "cloud_input_tokens",
        "cloud_output_tokens",
        "cloud_tokens",
        "local_tokens",
        "total_tokens",
        "latency_seconds",
        "cloud_latency_seconds",
        "local_wall_latency_seconds",
        "local_ollama_duration_seconds",
        "local_ollama_load_seconds",
        "local_prompt_eval_seconds",
        "local_generation_seconds",
        "routing_latency_seconds",
    )
    samples: dict[str, list[float]] = {name: [] for name in (*metric_names, *resource_names)}
    for _ in range(repetitions):
        sampled: list[dict[str, Any]] = []
        for cluster_id in generator.choices(cluster_ids, k=len(cluster_ids)):
            sampled.extend(clusters[cluster_id])
        confusion = _confusion(sampled)
        for name in metric_names:
            samples[name].append(float(confusion[name]))
        for name in resource_names:
            samples[name].append(statistics.fmean(float(row[name]) for row in sampled))
    point = _confusion(rows)
    output: dict[str, dict[str, float]] = {}
    for name in metric_names:
        output[name] = {
            "estimate": float(point[name]),
            "ci_low": _quantile(samples[name], 0.025),
            "ci_high": _quantile(samples[name], 0.975),
        }
    for name in resource_names:
        estimate = statistics.fmean(float(row[name]) for row in rows) if rows else 0.0
        output[name] = {
            "estimate": estimate,
            "ci_low": _quantile(samples[name], 0.025),
            "ci_high": _quantile(samples[name], 0.975),
        }
    return output


def _latency_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["latency_seconds"]) for row in rows]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "median": _quantile(values, 0.5),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "p90": _quantile(values, 0.9),
        "p95": _quantile(values, 0.95),
        "max": max(values, default=0.0),
    }


def _pricing_charge(
    call: Mapping[str, Any],
    pricing: Mapping[str, Any],
) -> float | None:
    model = str(call.get("model_id") or call.get("requested_model_id") or "")
    item = pricing.get(model)
    if not isinstance(item, Mapping):
        return None
    output_price = item.get("output_per_million")
    if output_price is None:
        return None
    usage = call.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", call.get("input_tokens", 0)))
    output_tokens = int(usage.get("output_tokens", call.get("output_tokens", 0)))
    cached_price = item.get("cached_input_per_million")
    uncached_price = item.get("uncached_input_per_million")
    if cached_price is not None and uncached_price is not None:
        details = call.get("usage_details") or {}
        cached_tokens = int(details.get("prompt_cache_hit_tokens") or 0)
        miss_value = details.get("prompt_cache_miss_tokens")
        uncached_tokens = int(miss_value) if miss_value is not None else input_tokens - cached_tokens
        return (
            cached_tokens * float(cached_price)
            + uncached_tokens * float(uncached_price)
            + output_tokens * float(output_price)
        ) / 1_000_000.0
    input_price = item.get("input_per_million")
    if input_price is None:
        return None
    return (input_tokens * float(input_price) + output_tokens * float(output_price)) / 1_000_000.0


def _call_resources(
    calls: Iterable[Mapping[str, Any]],
    *,
    pricing: Mapping[str, Any],
) -> dict[str, float | int | None]:
    cloud_input = cloud_output = local_tokens = total_tokens = calls_count = 0
    cloud_calls = local_calls = 0
    latency = 0.0
    cloud_latency = local_wall_latency = 0.0
    local_duration = 0.0
    local_load = local_prompt_eval = local_generation = 0.0
    charge_total = 0.0
    charge_complete = True
    phase_latency: dict[str, float] = defaultdict(float)
    for call in calls:
        usage = call.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", call.get("input_tokens", 0)))
        output_tokens = int(usage.get("output_tokens", call.get("output_tokens", 0)))
        backend = str(call.get("backend", ""))
        call_count = int(usage.get("calls", call.get("calls", 1)))
        call_latency = float(call.get("latency_seconds", 0.0))
        calls_count += call_count
        latency += call_latency
        phase_latency[str(call.get("phase", "unknown"))] += call_latency
        total_tokens += input_tokens + output_tokens
        if backend == "openai_compatible":
            cloud_calls += call_count
            cloud_latency += call_latency
            cloud_input += input_tokens
            cloud_output += output_tokens
            charge = _pricing_charge(call, pricing)
            if charge is None:
                charge_complete = False
            else:
                charge_total += charge
        elif backend == "ollama":
            local_calls += call_count
            local_wall_latency += call_latency
            local_tokens += input_tokens + output_tokens
            metadata = call.get("provider_metadata") or {}
            local_duration += float(metadata.get("total_duration_ns", 0) or 0) / 1_000_000_000.0
            local_load += float(metadata.get("load_duration_ns", 0) or 0) / 1_000_000_000.0
            local_prompt_eval += float(
                metadata.get("prompt_eval_duration_ns", 0) or 0
            ) / 1_000_000_000.0
            local_generation += float(
                metadata.get("eval_duration_ns", 0) or 0
            ) / 1_000_000_000.0
    return {
        "cloud_calls": cloud_calls,
        "local_calls": local_calls,
        "cloud_input_tokens": cloud_input,
        "cloud_output_tokens": cloud_output,
        "cloud_tokens": cloud_input + cloud_output,
        "local_tokens": local_tokens,
        "total_tokens": total_tokens,
        "calls": calls_count,
        "latency_seconds": latency,
        "cloud_latency_seconds": cloud_latency,
        "local_wall_latency_seconds": local_wall_latency,
        "local_ollama_duration_seconds": local_duration,
        "local_ollama_load_seconds": local_load,
        "local_prompt_eval_seconds": local_prompt_eval,
        "local_generation_seconds": local_generation,
        "routing_latency_seconds": 0.0,
        "phase_latency_seconds": dict(sorted(phase_latency.items())),
        "cloud_charge": charge_total if charge_complete else None,
    }


def _transition(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    rescued = harmed = unchanged_correct = unchanged_wrong = routed_responses = routed_wrong = 0
    for row in rows:
        gold = bool(row["gold_hallucinated"])
        stage1_correct = bool(row["stage1_hallucinated"]) == gold
        final_correct = bool(row["final_hallucinated"]) == gold
        routed = bool(row["routed_factoids"])
        if not routed and bool(row["stage1_hallucinated"]) != bool(row["final_hallucinated"]):
            raise AssertionError("A non-routed response cannot change its final prediction.")
        if routed:
            routed_responses += 1
            if not stage1_correct:
                routed_wrong += 1
        if not stage1_correct and final_correct:
            rescued += 1
        elif stage1_correct and not final_correct:
            harmed += 1
        elif stage1_correct:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1
    return transition_metrics_from_counts(
        rescued=rescued,
        harmed=harmed,
        unchanged_correct=unchanged_correct,
        unchanged_wrong=unchanged_wrong,
        routed_responses=routed_responses,
        routed_stage1_wrong=routed_wrong,
    )


def _external_single_stage_summary(
    run_directory: Path,
    *,
    preparation_calls_path: Path | None,
    pricing: Mapping[str, Any],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Load the RQ1 "strong-only" run as a direct, strong validation baseline for RQ2."""

    responses = _read_jsonl(run_directory / "responses.jsonl")
    calls = _read_jsonl(run_directory / "calls.jsonl")
    preparation_calls = _read_jsonl(preparation_calls_path) if preparation_calls_path else []
    calls_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in (*preparation_calls, *calls):
        calls_by_response[str(call["response_id"])].append(call)
    rows = []
    for response in responses:
        response_id = str(response["response_id"])
        resources = _call_resources(calls_by_response[response_id], pricing=pricing)
        rows.append(
            {
                "response_id": response_id,
                "question_id": str(response["question_id"]),
                "gold_hallucinated": str(response["gold_target"]) == BinaryTarget.UNSUPPORTED.value,
                "stage1_hallucinated": bool(response["final_hallucinated"]),
                "final_hallucinated": bool(response["final_hallucinated"]),
                "routed_factoids": [],
                "metadata": dict(response.get("metadata", {})),
                **resources,
            }
        )
    charge_available = bool(rows) and all(row["cloud_charge"] is not None for row in rows)
    return {
        "final": _confusion(rows),
        "uncertainty": _bootstrap_ci(
            rows, repetitions=bootstrap_repetitions, seed=bootstrap_seed
        ),
        "latency_distribution_seconds": _latency_distribution(rows),
        "resource_means": {
            name: statistics.fmean(float(row[name]) for row in rows) if rows else 0.0
            for name in (
                "cloud_calls",
                "local_calls",
                "cloud_input_tokens",
                "cloud_output_tokens",
                "cloud_tokens",
                "local_tokens",
                "total_tokens",
                "latency_seconds",
                "cloud_latency_seconds",
                "local_wall_latency_seconds",
                "local_ollama_duration_seconds",
                "local_ollama_load_seconds",
                "local_prompt_eval_seconds",
                "local_generation_seconds",
                "routing_latency_seconds",
            )
        },
        "cloud_charge": {
            "available": charge_available,
            "total": (
                sum(float(row["cloud_charge"]) for row in rows)
                if charge_available
                else None
            ),
            "mean_per_response": (
                statistics.fmean(float(row["cloud_charge"]) for row in rows)
                if charge_available
                else None
            ),
        },
        "rows": rows,
        "baseline_scope": "direct strong verifier without Stage-1 local verification",
    }


class MatrixLedger:
    """A complete read-only view of the dual-verifier ledger."""

    def __init__(self, run_directory: str | Path, preparation_calls_path: str | Path | None = None):
        self.run_directory = Path(run_directory)
        self.responses = _read_jsonl(self.run_directory / "responses.jsonl")
        self.factoids = _read_jsonl(self.run_directory / "factoids.jsonl")
        self.calls = _read_jsonl(self.run_directory / "calls.jsonl")
        self.preparation_calls = (
            _read_jsonl(Path(preparation_calls_path)) if preparation_calls_path else []
        )
        self.response_by_id = {str(row["response_id"]): row for row in self.responses}
        if len(self.response_by_id) != len(self.responses):
            raise ValueError("Matrix responses contain duplicate response_id values.")
        self.factoids_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.factoid_by_id: dict[str, dict[str, Any]] = {}
        for factoid in self.factoids:
            factoid_id = str(factoid["factoid_id"])
            if factoid_id in self.factoid_by_id:
                raise ValueError(f"Duplicate factoid_id in matrix: {factoid_id}")
            if factoid.get("stage2") is None:
                raise ValueError(
                    "Policy replay requires Stage-2 output for every factoid; "
                    f"missing for {factoid_id}. Run the matrix with full_escalation."
                )
            self.factoid_by_id[factoid_id] = factoid
            self.factoids_by_response[str(factoid["response_id"])].append(factoid)
        if set(self.factoids_by_response) != set(self.response_by_id):
            raise ValueError("Response/factoid ledger IDs do not match.")
        self.calls_by_factoid_phase: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for call in self.calls:
            self.calls_by_factoid_phase[(str(call["factoid_id"]), str(call["phase"]))].append(call)
        self.preparation_calls_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in self.preparation_calls:
            self.preparation_calls_by_response[str(call["response_id"])].append(call)

    @property
    def factoid_ids(self) -> list[str]:
        return sorted(self.factoid_by_id)

    def route_policy(self, policy: RoutingPolicy, anomaly_count_threshold: int = 2) -> set[str]:
        return {
            factoid_id
            for factoid_id, row in self.factoid_by_id.items()
            if should_escalate(
                _signals(row),
                policy,
                anomaly_count_threshold=anomaly_count_threshold,
            )
        }

    def stage2_costs(self, unit: str = "cloud_tokens") -> dict[str, int]:
        output: dict[str, int] = {}
        for factoid_id in self.factoid_ids:
            calls = self.calls_by_factoid_phase[(factoid_id, "stage2_verify")]
            if unit == "cloud_tokens":
                output[factoid_id] = sum(
                    int((call.get("usage") or {}).get("input_tokens", call.get("input_tokens", 0)))
                    + int((call.get("usage") or {}).get("output_tokens", call.get("output_tokens", 0)))
                    for call in calls
                    if call.get("backend") == "openai_compatible"
                )
            elif unit == "calls":
                output[factoid_id] = sum(
                    int((call.get("usage") or {}).get("calls", call.get("calls", 1)))
                    for call in calls
                )
            else:
                raise ValueError("Budget unit must be cloud_tokens or calls.")
        return output

    def evaluate(
        self,
        selected_factoids: set[str],
        *,
        threshold: float,
        pricing: Mapping[str, Any],
        bootstrap_repetitions: int,
        bootstrap_seed: int,
        routing_latency_total_seconds: float = 0.0,
    ) -> dict[str, Any]:
        unknown = selected_factoids.difference(self.factoid_by_id)
        if unknown:
            raise ValueError(f"Unknown selected factoids: {sorted(unknown)[:10]}")
        rows: list[dict[str, Any]] = []
        trigger_counts: Counter[str] = Counter()
        selected_count = 0
        for response_id, response in self.response_by_id.items():
            factoids = self.factoids_by_response[response_id]
            stage1_scores = [float(item["stage1_score"]) for item in factoids]
            final_scores = []
            response_selected: list[str] = []
            response_calls: list[dict[str, Any]] = list(
                self.preparation_calls_by_response.get(response_id, [])
            )
            for item in factoids:
                factoid_id = str(item["factoid_id"])
                response_calls.extend(self.calls_by_factoid_phase[(factoid_id, "stage1_verify")])
                if factoid_id in selected_factoids:
                    selected_count += 1
                    response_selected.append(factoid_id)
                    response_calls.extend(self.calls_by_factoid_phase[(factoid_id, "stage2_verify")])
                    final_scores.append(float(item["final_score"]))
                    for name in _signals(item).active_names():
                        trigger_counts[name] += 1
                else:
                    final_scores.append(float(item["stage1_score"]))
            stage1_score = aggregate_response_scores(stage1_scores)
            final_score = aggregate_response_scores(final_scores)
            resources = _call_resources(response_calls, pricing=pricing)
            routing_latency = (
                float(routing_latency_total_seconds) * len(factoids) / len(self.factoids)
                if self.factoids
                else 0.0
            )
            resources["routing_latency_seconds"] = routing_latency
            resources["latency_seconds"] = float(resources["latency_seconds"]) + routing_latency
            resources["phase_latency_seconds"]["routing"] = routing_latency
            gold_hallucinated = str(response["gold_target"]) == BinaryTarget.UNSUPPORTED.value
            rows.append(
                {
                    "response_id": response_id,
                    "question_id": str(response["question_id"]),
                    "gold_hallucinated": gold_hallucinated,
                    "stage1_score": stage1_score,
                    "final_score": final_score,
                    "stage1_hallucinated": predicts_hallucination(stage1_score, threshold),
                    "final_hallucinated": predicts_hallucination(final_score, threshold),
                    "routed_factoids": response_selected,
                    "metadata": dict(response.get("metadata", {})),
                    **resources,
                }
            )
        final = _confusion(rows)
        stage1_rows = [dict(row, final_hallucinated=row["stage1_hallucinated"]) for row in rows]
        stage1 = _confusion(stage1_rows)
        transitions = _transition(rows)
        if int(final["tp"]) + int(final["tn"]) - int(stage1["tp"]) - int(stage1["tn"]) != transitions["net_correct_gain"]:
            raise AssertionError("Rescue/harm counts do not reconcile with confusion matrices.")
        summary: dict[str, Any] = {
            "stage1": stage1,
            "final": final,
            "transitions": transitions,
            "routing": {
                "routed_factoids": selected_count,
                "total_factoids": len(self.factoids),
                "factoid_escalation_rate": defined_rate(selected_count, len(self.factoids)),
                "routed_responses": transitions["routed_responses"],
                "total_responses": len(rows),
                "response_escalation_rate": defined_rate(
                    int(transitions["routed_responses"]), len(rows)
                ),
                "active_signal_occurrences_on_routed_factoids": dict(sorted(trigger_counts.items())),
            },
            "uncertainty": (
                _bootstrap_ci(
                    rows,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                )
                if bootstrap_repetitions > 0
                else {"status": "not_computed_for_seeded_random_replicate"}
            ),
            "latency_distribution_seconds": _latency_distribution(rows),
            "resource_means": {
                name: statistics.fmean(float(row[name]) for row in rows) if rows else 0.0
                for name in (
                    "cloud_calls",
                    "local_calls",
                    "cloud_input_tokens",
                    "cloud_output_tokens",
                    "cloud_tokens",
                    "local_tokens",
                    "total_tokens",
                    "latency_seconds",
                    "cloud_latency_seconds",
                    "local_wall_latency_seconds",
                    "local_ollama_duration_seconds",
                    "local_ollama_load_seconds",
                    "local_prompt_eval_seconds",
                    "local_generation_seconds",
                    "routing_latency_seconds",
                )
            },
            "component_latency_means_seconds": {
                phase: statistics.fmean(
                    float((row.get("phase_latency_seconds") or {}).get(phase, 0.0))
                    for row in rows
                )
                for phase in sorted(
                    {
                        name
                        for row in rows
                        for name in (row.get("phase_latency_seconds") or {})
                    }
                )
            },
            "cloud_charge": {
                "available": all(row["cloud_charge"] is not None for row in rows),
                "total": (
                    sum(float(row["cloud_charge"]) for row in rows)
                    if rows and all(row["cloud_charge"] is not None for row in rows)
                    else None
                ),
                "mean_per_response": (
                    statistics.fmean(float(row["cloud_charge"]) for row in rows)
                    if rows and all(row["cloud_charge"] is not None for row in rows)
                    else None
                ),
            },
            "rows": rows,
        }
        by_category: dict[str, Any] = {}
        categories = sorted(
            {
                str(row["metadata"].get("factoid_type"))
                for row in rows
                if row["metadata"].get("factoid_type") is not None
            }
        )
        for category in categories:
            subset = [row for row in rows if str(row["metadata"].get("factoid_type")) == category]
            by_category[category] = {
                "final": _confusion(subset),
                "transitions": _transition(subset),
                "routing": {
                    "responses": len(subset),
                    "routed": sum(bool(row["routed_factoids"]) for row in subset),
                    "escalation_rate": (
                        sum(bool(row["routed_factoids"]) for row in subset) / len(subset)
                        if subset
                        else 0.0
                    ),
                },
            }
        if by_category:
            summary["by_factoid_type"] = by_category
        return summary

    def oracle_minimal_rescue(self, *, threshold: float, max_exhaustive_factoids: int = 14) -> set[str]:
        """Non-deploying Oracle: Routes only the minimal subset of factoids capable of turning an erroneous response into a correct one."""

        selected: set[str] = set()
        for response_id, response in self.response_by_id.items():
            factoids = self.factoids_by_response[response_id]
            stage1_scores = [float(item["stage1_score"]) for item in factoids]
            gold = str(response["gold_target"]) == BinaryTarget.UNSUPPORTED.value
            if predicts_hallucination(aggregate_response_scores(stage1_scores), threshold) == gold:
                continue
            indexes = range(len(factoids))
            best: tuple[int, ...] | None = None
            if len(factoids) <= max_exhaustive_factoids:
                for size in range(1, len(factoids) + 1):
                    for subset in itertools.combinations(indexes, size):
                        scores = [
                            float(item["final_score"]) if index in subset else float(item["stage1_score"])
                            for index, item in enumerate(factoids)
                        ]
                        if predicts_hallucination(aggregate_response_scores(scores), threshold) == gold:
                            best = subset
                            break
                    if best is not None:
                        break
            else:
        
                order = sorted(
                    indexes,
                    key=lambda index: abs(
                        float(factoids[index]["final_score"]) - float(factoids[index]["stage1_score"])
                    ),
                    reverse=True,
                )
                chosen: list[int] = []
                for index in order:
                    chosen.append(index)
                    scores = [
                        float(item["final_score"]) if position in chosen else float(item["stage1_score"])
                        for position, item in enumerate(factoids)
                    ]
                    if predicts_hallucination(aggregate_response_scores(scores), threshold) == gold:
                        best = tuple(chosen)
                        break
            if best is not None:
                selected.update(str(factoids[index]["factoid_id"]) for index in best)
        return selected

    def learned_router(self, *, target_count: int, folds: int, seed: int, threshold: float) -> dict[str, Any]:
        """A lightweight logistic regression router based on cross-fitting; the training labels are single-factoid marginal utility."""

        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import GroupKFold
        except ImportError as error:
            return {"status": "unavailable", "reason": f"missing dependency: {error}"}

        ids = self.factoid_ids
        features: list[list[float]] = []
        labels: list[int] = []
        groups: list[str] = []
        for factoid_id in ids:
            item = self.factoid_by_id[factoid_id]
            signals = stage1_feature_dict(_signals(item))
            features.append([float(signals[name]) for name in sorted(signals)])
            response_id = str(item["response_id"])
            response = self.response_by_id[response_id]
            response_factoids = self.factoids_by_response[response_id]
            baseline_scores = [float(row["stage1_score"]) for row in response_factoids]
            gold = str(response["gold_target"]) == BinaryTarget.UNSUPPORTED.value
            baseline_correct = predicts_hallucination(
                aggregate_response_scores(baseline_scores), threshold
            ) == gold
            counterfactual_scores = [
                float(row["final_score"]) if str(row["factoid_id"]) == factoid_id else float(row["stage1_score"])
                for row in response_factoids
            ]
            counterfactual_correct = predicts_hallucination(
                aggregate_response_scores(counterfactual_scores), threshold
            ) == gold
            labels.append(int((not baseline_correct) and counterfactual_correct))
            groups.append(str(response["question_id"]))
        if len(set(labels)) < 2 or sum(labels) < folds:
            return {
                "status": "unavailable",
                "reason": "insufficient positive marginal-rescue labels for grouped cross-fitting",
                "positive_labels": sum(labels),
            }
        group_count = len(set(groups))
        actual_folds = min(int(folds), group_count)
        splitter = GroupKFold(n_splits=actual_folds)
        probabilities = [0.0] * len(ids)
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        g = np.asarray(groups)
        for train, test in splitter.split(x, y, groups=g):
            if len(set(y[train].tolist())) < 2:
                return {"status": "unavailable", "reason": "a training fold has one class"}
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            )
            model.fit(x[train], y[train])
            for index, probability in zip(test, model.predict_proba(x[test])[:, 1], strict=True):
                probabilities[int(index)] = float(probability)
        ranked = sorted(range(len(ids)), key=lambda index: (-probabilities[index], ids[index]))
        selected = {ids[index] for index in ranked[:target_count]}
        return {
            "status": "ok",
            "selected": selected,
            "folds": actual_folds,
            "seed": seed,
            "target": "single_factoid_marginal_response_rescue",
            "positive_labels": sum(labels),
            "feature_names": sorted(stage1_feature_dict(_signals(self.factoid_by_id[ids[0]]))),
        }


def _holm_adjust(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _aggregate_random_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multi-seed randomized strategies, distinguishing between randomness and instance-sampling uncertainty."""

    metrics = ("accuracy", "precision", "recall", "f1")
    output: dict[str, Any] = {"seeds": len(runs)}
    for metric in metrics:
        values = [float(run["summary"]["final"][metric]) for run in runs]
        output[metric] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values, default=0.0),
            "max": max(values, default=0.0),
            "empirical_q025": _quantile(values, 0.025),
            "empirical_q975": _quantile(values, 0.975),
        }
    routed = [int(run["summary"]["routing"]["routed_factoids"]) for run in runs]
    output["routed_factoids"] = {
        "mean": statistics.fmean(routed) if routed else 0.0,
        "min": min(routed, default=0),
        "max": max(routed, default=0),
    }
    return output


def _comparison(
    first_name: str,
    first_rows: Sequence[dict[str, Any]],
    second_name: str,
    second_rows: Sequence[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    first_by_id = {str(row["response_id"]): row for row in first_rows}
    second_by_id = {str(row["response_id"]): row for row in second_rows}
    if set(first_by_id) != set(second_by_id):
        raise ValueError("Paired comparison response IDs do not match.")
    pairs = [
        PairedPrediction(
            question_id=str(first_by_id[item_id]["question_id"]),
            gold_hallucinated=bool(first_by_id[item_id]["gold_hallucinated"]),
            first_hallucinated=bool(first_by_id[item_id]["final_hallucinated"]),
            second_hallucinated=bool(second_by_id[item_id]["final_hallucinated"]),
        )
        for item_id in sorted(first_by_id)
    ]
    return {
        "first": first_name,
        "second": second_name,
        "mcnemar": mcnemar_exact(pairs),
        "accuracy_difference_second_minus_first": clustered_paired_difference(
            pairs, metric="accuracy", repetitions=repetitions, seed=seed
        ),
        "f1_difference_second_minus_first": clustered_paired_difference(
            pairs, metric="f1", repetitions=repetitions, seed=seed
        ),
    }


def run_policy_suite(config: dict[str, Any], *, repository_root: str | Path) -> Path:
    root = Path(repository_root)
    run_directory = root / str(config["matrix_run_directory"])
    preparation_calls = config.get("preparation_calls_path")
    matrix = MatrixLedger(
        run_directory,
        root / str(preparation_calls) if preparation_calls else None,
    )
    threshold = float(config.get("threshold", 0.5))
    repetitions = int(config.get("bootstrap_repetitions", 2000))
    seed = int(config.get("bootstrap_seed", 42))
    pricing = config.get("pricing", {})
    policies = [
        RoutingPolicy(str(name))
        for name in config.get(
            "policies",
            ["strict", "paradox_only", "structure_aware", "uncertainty_only", "anomaly_count"],
        )
    ]
    summaries: dict[str, Any] = {}
    selected_by_name: dict[str, set[str]] = {}
    routing_microbenchmarks: dict[str, Any] = {}
    for policy in policies:
        routing_started = time.perf_counter()
        selected = matrix.route_policy(
            policy,
            anomaly_count_threshold=int(config.get("anomaly_count_threshold", 2)),
        )
        routing_elapsed = time.perf_counter() - routing_started
        routing_microbenchmarks[policy.value] = {
            "total_seconds": routing_elapsed,
            "factoids": len(matrix.factoids),
            "mean_seconds_per_factoid": (
                routing_elapsed / len(matrix.factoids) if matrix.factoids else 0.0
            ),
            "scope": "offline Python routing-rule replay only; excludes all model calls",
        }
        selected_by_name[policy.value] = selected
        summaries[policy.value] = matrix.evaluate(
            selected,
            threshold=threshold,
            pricing=pricing,
            bootstrap_repetitions=repetitions,
            bootstrap_seed=seed,
            routing_latency_total_seconds=routing_elapsed,
        )

    main_selected = selected_by_name.get("structure_aware")
    if main_selected is None:
        raise ValueError("Policy suite must include structure_aware as the primary policy.")
    stage2_cost = matrix.stage2_costs(str(config.get("budget_unit", "cloud_tokens")))
    target_budget = sum(stage2_cost[item] for item in main_selected)
    include_controls = bool(config.get("include_routing_controls", True))
    if include_controls:
        configured_seeds = config.get("random_seeds")
        random_seeds = (
            [int(value) for value in configured_seeds]
            if configured_seeds is not None
            else list(range(int(config.get("random_seed_count", 1000))))
        )
    else:
        random_seeds = []
    random_rate_runs = []
    random_budget_runs = []
    for random_seed in random_seeds:
        rate_selected = random_matched_count(matrix.factoid_ids, len(main_selected), seed=random_seed)
        budget_selected = random_matched_budget(stage2_cost, target_budget, seed=random_seed)
        rate_summary = matrix.evaluate(
            rate_selected,
            threshold=threshold,
            pricing=pricing,
            bootstrap_repetitions=0,
            bootstrap_seed=random_seed,
        )
        budget_summary = matrix.evaluate(
            budget_selected,
            threshold=threshold,
            pricing=pricing,
            bootstrap_repetitions=0,
            bootstrap_seed=random_seed,
        )
        rate_summary.pop("rows", None)
        budget_summary.pop("rows", None)
        random_rate_runs.append({"seed": random_seed, "summary": rate_summary})
        random_budget_runs.append(
            {
                "seed": random_seed,
                "used_budget": sum(stage2_cost[item] for item in budget_selected),
                "target_budget": target_budget,
                "summary": budget_summary,
            }
        )

    learned: dict[str, Any] = {"status": "not_requested"}
    if include_controls:
        oracle_selected = matrix.oracle_minimal_rescue(threshold=threshold)
        summaries["oracle_minimal_rescue"] = matrix.evaluate(
            oracle_selected,
            threshold=threshold,
            pricing=pricing,
            bootstrap_repetitions=repetitions,
            bootstrap_seed=seed,
        )
        learned = matrix.learned_router(
            target_count=len(main_selected),
            folds=int(config.get("learned_router_folds", 5)),
            seed=int(config.get("learned_router_seed", 42)),
            threshold=threshold,
        )
    if learned.get("status") == "ok":
            learned_selected = set(learned.pop("selected"))
            summaries["learned_router_matched_rate"] = matrix.evaluate(
                learned_selected,
                threshold=threshold,
                pricing=pricing,
                bootstrap_repetitions=repetitions,
                bootstrap_seed=seed,
            )

    strong_only_run = config.get("strong_only_run_directory")
    if strong_only_run:
        strong_prep = config.get("strong_only_preparation_calls_path", preparation_calls)
        summaries["strong_only"] = _external_single_stage_summary(
            root / str(strong_only_run),
            preparation_calls_path=root / str(strong_prep) if strong_prep else None,
            pricing=pricing,
            bootstrap_repetitions=repetitions,
            bootstrap_seed=seed,
        )

    comparison_names = [name for name in summaries if name != "structure_aware"]
    comparisons = {}
    raw_main_rows = summaries["structure_aware"]["rows"]
    for name in comparison_names:
        comparisons[name + "_vs_structure_aware"] = _comparison(
            name,
            summaries[name]["rows"],
            "structure_aware",
            raw_main_rows,
            repetitions=repetitions,
            seed=seed,
        )
    p_values = [
        (name, float(value["mcnemar"]["p_value_two_sided_exact"]))
        for name, value in comparisons.items()
    ]
    holm = _holm_adjust(p_values)
    for name, value in comparisons.items():
        value["mcnemar"]["holm_adjusted_p"] = holm[name]

    include_rows = bool(config.get("include_response_rows", True))
    if not include_rows:
        for summary in summaries.values():
            summary.pop("rows", None)
    report = {
        "analysis_schema_version": "2026-08-22.metrics-v2",
        "matrix_run_directory": Path(str(config["matrix_run_directory"])).as_posix(),
        "method": {
            "threshold": threshold,
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed,
            "cluster_unit": "question_id",
            "random_seed_count": len(random_seeds),
            "random_seeds": random_seeds,
            "multiple_comparison_correction": "Holm within policy-vs-main McNemar family",
            "latency_scope": "sum of measured per-call sequential latencies in the frozen source ledger",
            "pricing_scope": "computed only when explicit per-model prices are supplied",
            "rq2_metric_definitions": {
                "response_escalation_rate": "routed_responses / total_responses",
                "factoid_escalation_rate": "routed_factoids / total_factoids",
                "rescue_rate_response": "rescued / routed_responses",
                "error_conditional_rescue_rate": (
                    "rescued / routed_stage1_wrong_responses"
                ),
                "harm_rate_response": "harmed / routed_responses",
                "zero_denominator": "null (not estimable), never coerced to zero",
            },
        },
        "policies": summaries,
        "routing_rule_microbenchmarks": routing_microbenchmarks,
        "random_matched_factoid_rate": random_rate_runs,
        "random_matched_cloud_budget": random_budget_runs,
        "random_matched_factoid_rate_aggregate": _aggregate_random_runs(random_rate_runs),
        "random_matched_cloud_budget_aggregate": _aggregate_random_runs(random_budget_runs),
        "learned_router": {key: value for key, value in learned.items() if key != "selected"},
        "confidence_router": (
            {
                "status": "unavailable",
                "reason": "The frozen three-way verifier protocol records labels, not calibrated token probabilities/logprobs.",
            }
            if include_controls
            else {"status": "not_requested_for_rq3"}
        ),
        "oracle": {
            "status": "computed" if include_controls else "not_requested_for_rq3",
            "deployable": False,
            "definition": "minimum factoid subset that changes a wrong Stage-1 response to correct using gold labels and Stage-2 outputs",
        },
        "paired_comparisons": comparisons,
    }
    output = root / str(config["output_path"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite policy report: {output}")
    atomic_write_json(output, report)
    return output
