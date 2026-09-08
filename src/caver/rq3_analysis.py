"""RQ3 Statistics on the factoid types, failure mechanisms, and repairs of the main strategy."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .labels import BinaryTarget
from .ledger import atomic_write_json
from .metrics import defined_rate
from .policy_analysis import (
    _bootstrap_ci,
    _call_resources,
    _confusion,
    _read_jsonl,
    _transition,
)
from .taxonomy import taxonomy_manifest


def _failure_family(factoid: dict[str, Any], stage1_correct: bool) -> str:
    

    signals = factoid["signals"]
    if bool(signals["original_uncertainty"]):
        return "Original verdict uncertainty"
    if bool(signals["explicit_paradox"]):
        return "Explicit paradox"
    if bool(signals["cross_side_anomaly"]):
        return "Compound probe failure"
    if bool(signals["two_antonym_anomalies"]):
        return "Antonym/reversal failure"
    if int(signals["synonym_anomaly_count"]) > 0:
        return "Synonym preservation failure"
    if not stage1_correct:
        return "Baseline decision error without routing signal"
    return "No detected failure"


def _category_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    

    transitions = _transition(rows)
    routed = sum(bool(row["routed_factoids"]) for row in rows)
    stage1_wrong = sum(
        bool(row["stage1_hallucinated"]) != bool(row["gold_hallucinated"]) for row in rows
    )
    routed_stage1_wrong = sum(
        bool(row["routed_factoids"])
        and bool(row["stage1_hallucinated"]) != bool(row["gold_hallucinated"])
        for row in rows
    )
    over_escalation = sum(
        bool(row["routed_factoids"])
        and bool(row["stage1_hallucinated"]) == bool(row["gold_hallucinated"])
        for row in rows
    )
    missed = sum(
        not bool(row["routed_factoids"])
        and bool(row["stage1_hallucinated"]) != bool(row["gold_hallucinated"])
        for row in rows
    )
    correct_non_escalation = sum(
        not bool(row["routed_factoids"])
        and bool(row["stage1_hallucinated"]) == bool(row["gold_hallucinated"])
        for row in rows
    )
    total = len(rows)
    # The four behavioral states must constitute a mutually exclusive and collectively exhaustive partition of all factoids.
    if missed + over_escalation + correct_non_escalation + routed_stage1_wrong != total:
        raise AssertionError("RQ3 routing outcomes do not partition all factoids.")
    if routed != over_escalation + routed_stage1_wrong:
        raise AssertionError("RQ3 escalation count must equal over + valid escalation.")
    if stage1_wrong != missed + routed_stage1_wrong:
        raise AssertionError("RQ3 Stage-1 errors must equal missed + valid escalation.")

    escalation_efficiency = (
        100.0 * float(transitions["net_correct_gain"]) / routed if routed else None
    )
    return {
        "n": total,
        "supported": sum(not bool(row["gold_hallucinated"]) for row in rows),
        "unsupported": sum(bool(row["gold_hallucinated"]) for row in rows),
        "stage1": _confusion(
            [dict(row, final_hallucinated=row["stage1_hallucinated"]) for row in rows]
        ),
        "final": _confusion(rows),
        "routing": {
            "routed": routed,
            "escalation_rate": defined_rate(routed, total),
            "missed_detection": missed,
            "missed_detection_rate": defined_rate(missed, total),
            "over_escalation": over_escalation,
            "over_escalation_rate": defined_rate(over_escalation, total),
            "success_pass": correct_non_escalation,
            "success_pass_rate": defined_rate(correct_non_escalation, total),
            "valid_escalation": routed_stage1_wrong,
            "valid_escalation_rate": defined_rate(routed_stage1_wrong, total),
            "stage1_errors": stage1_wrong,
            "routed_stage1_errors": routed_stage1_wrong,
            "missed_escalation": missed,
            "correct_non_escalation": correct_non_escalation,
            "rate_denominator": total,
        },
        "repair": {
            **transitions,
            "rescue_rate": transitions["rescue_rate_response"],
            "error_conditional_rescue_rate": transitions[
                "error_conditional_rescue_rate"
            ],
            "harm_rate": transitions["harm_rate_response"],
            "escalation_efficiency_per_100": escalation_efficiency,
        },
        "failure_families": dict(sorted(Counter(row["failure_family"] for row in rows).items())),
    }


def run_rq3_analysis(config: dict[str, Any], *, repository_root: str | Path) -> Path:
    root = Path(repository_root)
    run_directory = root / str(config["run_directory"])
    responses = _read_jsonl(run_directory / "responses.jsonl")
    factoids = _read_jsonl(run_directory / "factoids.jsonl")
    calls = _read_jsonl(run_directory / "calls.jsonl")
    response_by_id = {str(row["response_id"]): row for row in responses}
    factoids_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for factoid in factoids:
        factoids_by_response[str(factoid["response_id"])].append(factoid)
    if any(len(items) != 1 for items in factoids_by_response.values()):
        raise ValueError("RQ3 final700 must contain exactly one source factoid per response.")
    if set(factoids_by_response) != set(response_by_id):
        raise ValueError("RQ3 response and factoid ledgers do not align.")
    calls_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        response_id = str(call.get("response_id", ""))
        if not response_id:
            raise ValueError("RQ3 call ledger contains a row without response_id.")
        calls_by_response[response_id].append(call)
    unknown_call_responses = set(calls_by_response).difference(response_by_id)
    if unknown_call_responses:
        raise ValueError(
            "RQ3 call ledger contains unknown response IDs: "
            f"{sorted(unknown_call_responses)[:10]}"
        )
    missing_call_responses = set(response_by_id).difference(calls_by_response)
    if missing_call_responses:
        raise ValueError(
            "RQ3 call ledger is missing response IDs: "
            f"{sorted(missing_call_responses)[:10]}"
        )
    pricing = config.get("pricing", {})
    if not isinstance(pricing, dict):
        raise TypeError("RQ3 analysis pricing must be an object.")
    rows: list[dict[str, Any]] = []
    for response_id, response in response_by_id.items():
        factoid = factoids_by_response[response_id][0]
        gold = str(response["gold_target"]) == BinaryTarget.UNSUPPORTED.value
        stage1 = bool(response["stage1_hallucinated"])
        final = bool(response["final_hallucinated"])
        routed = bool(factoid["routed"])
        stage1_correct = stage1 == gold
        metadata = response.get("metadata") or {}
        resources = _call_resources(calls_by_response[response_id], pricing=pricing)
        routing_latency = float(factoid.get("routing_latency_seconds", 0.0))
        resources["routing_latency_seconds"] = routing_latency
        resources["latency_seconds"] = (
            float(resources["latency_seconds"]) + routing_latency
        )
        resources["phase_latency_seconds"]["routing"] = routing_latency
        rows.append(
            {
                "response_id": response_id,
                "question_id": str(response["question_id"]),
                "gold_hallucinated": gold,
                "stage1_hallucinated": stage1,
                "final_hallucinated": final,
                "routed_factoids": [str(factoid["factoid_id"])] if routed else [],
                "category": str(metadata.get("factoid_type")),
                "source_dataset": metadata.get("source_dataset"),
                "failure_family": _failure_family(factoid, stage1_correct),

                **resources,
            }
        )
    categories = sorted({row["category"] for row in rows})
    category_results = {
        category: _category_summary([row for row in rows if row["category"] == category])
        for category in categories
    }
    overall = _category_summary(rows)
    uncertainty = _bootstrap_ci(
        rows,
        repetitions=int(config.get("bootstrap_repetitions", 2000)),
        seed=int(config.get("bootstrap_seed", 42)),
    )
    report = {
        "analysis_schema_version": "2026-08-28.metrics-v3",
        "run_directory": Path(str(config["run_directory"])).as_posix(),
        "method": {
            "routing_policy": "structure_aware",
            "unit": "source factoid",
            "decomposition_in_rq3": False,
            "cluster_unit": "source question_id",
            "resource_scope": (
                "per-response policy-equivalent verification resources from calls.jsonl; "
                "routing latency included; preparation excluded; cache-hit execution counts "
                "remain reported separately in the run summary/manifest"
            ),
            "metric_definitions": {
                "escalation_rate": "routed / n",
                "missed_detection_rate": "not_routed_and_stage1_wrong / n",
                "over_escalation_rate": "routed_and_stage1_correct / n",
                "success_pass_rate": "not_routed_and_stage1_correct / n",
                "valid_escalation_rate": "routed_and_stage1_wrong / n",
                "rescue_rate": "rescued / routed",
                "error_conditional_rescue_rate": "rescued / valid_escalation",
                "harm_rate": "harmed / routed",
                "escalation_efficiency_per_100": (
                    "100 * (rescued - harmed) / routed"
                ),
                "zero_denominator": "null (not estimable), never coerced to zero",
            },
            "failure_family_priority": [
                "Original verdict uncertainty",
                "Explicit paradox",
                "Compound probe failure",
                "Antonym/reversal failure",
                "Synonym preservation failure",
                "Baseline decision error without routing signal",
                "No detected failure",
            ],
            "rq3_taxonomy": taxonomy_manifest(),
        },
        "overall": overall,
        "overall_cluster_bootstrap_95ci": uncertainty,
        "by_category": category_results,
    }
    if bool(config.get("include_factoid_rows", True)):
        report["factoid_rows"] = rows
    output = root / str(config["output_path"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RQ3 report: {output}")
    atomic_write_json(output, report)
    return output
