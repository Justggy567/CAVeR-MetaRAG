"""RQ1 4  Unified aggregation of validator results and paired statistics."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .labels import BinaryTarget
from .ledger import atomic_write_json
from .policy_analysis import (
    _bootstrap_ci,
    _call_resources,
    _comparison,
    _confusion,
    _holm_adjust,
    _latency_distribution,
    _read_jsonl,
)


def _analyze_run(
    run_directory: Path,
    *,
    preparation_calls_path: Path | None,
    pricing: Mapping[str, Any],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
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
                "final_hallucinated": bool(response["final_hallucinated"]),
                **resources,
            }
        )
    charge_available = bool(rows) and all(row["cloud_charge"] is not None for row in rows)
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    return {
        "manifest": {
            "run_id": manifest.get("run_id"),
            "prepared_sha256": manifest.get("prepared_sha256"),
            "stage1": manifest.get("stage1"),
            "threshold": manifest.get("threshold"),
            "prompt_bundle": manifest.get("prompt_bundle"),
        },
        "confusion": _confusion(rows),
        "uncertainty": _bootstrap_ci(
            rows,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
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
            "available": charge_available,
            "total": sum(float(row["cloud_charge"]) for row in rows) if charge_available else None,
            "mean_per_response": (
                statistics.fmean(float(row["cloud_charge"]) for row in rows)
                if charge_available
                else None
            ),
        },
        "rows": rows,
    }


def run_rq1_analysis(config: dict[str, Any], *, repository_root: str | Path) -> Path:
    root = Path(repository_root)
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("RQ1 analysis config requires a non-empty runs array.")
    repetitions = int(config.get("bootstrap_repetitions", 2000))
    seed = int(config.get("bootstrap_seed", 42))
    pricing = config.get("pricing", {})
    results: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in runs:
        dataset = str(item["dataset"])
        model = str(item["model_label"])
        preparation_calls = item.get("preparation_calls_path")
        results[dataset][model] = _analyze_run(
            root / str(item["run_directory"]),
            preparation_calls_path=root / str(preparation_calls) if preparation_calls else None,
            pricing=pricing,
            bootstrap_repetitions=repetitions,
            bootstrap_seed=seed,
        )

    comparisons: dict[str, Any] = {}
    reference = str(config.get("reference_model", "DeepSeek-V4-Flash"))
    for dataset, model_results in results.items():
        if reference not in model_results:
            raise ValueError(f"Reference model {reference!r} missing for dataset {dataset!r}.")
        family_p_values: list[tuple[str, float]] = []
        for model, value in model_results.items():
            if model == reference:
                continue
            name = f"{dataset}:{model}_vs_{reference}"
            comparison = _comparison(
                model,
                value["rows"],
                reference,
                model_results[reference]["rows"],
                repetitions=repetitions,
                seed=seed,
            )
            comparisons[name] = comparison
            family_p_values.append((name, float(comparison["mcnemar"]["p_value_two_sided_exact"])))
        adjusted = _holm_adjust(family_p_values)
        for name, p_value in adjusted.items():
            comparisons[name]["mcnemar"]["holm_adjusted_p"] = p_value

    if not bool(config.get("include_response_rows", True)):
        for model_results in results.values():
            for value in model_results.values():
                value.pop("rows", None)
    report = {
        "method": {
            "cluster_unit": "question_id",
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed,
            "paired_reference_model": reference,
            "multiple_comparison_correction": "Holm within each dataset/model-vs-reference family",
            "cost_scope": "cloud charges are omitted unless explicit prices are supplied",
        },
        "datasets": {key: value for key, value in sorted(results.items())},
        "paired_comparisons": comparisons,
    }
    output = root / str(config["output_path"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RQ1 report: {output}")
    atomic_write_json(output, report)
    return output
