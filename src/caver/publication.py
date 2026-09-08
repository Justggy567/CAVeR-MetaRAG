"""Export the actual analysis report as a CSV file for direct verification in a paper."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ledger import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(root: Path, value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(
            f"Analysis report does not exist:{path}. You must first complete the corresponding experiments and the analyze command; you cannot export placeholder results."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError(f"The analysis report must be a JSON object：{path}")
    return path, report


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _require_keys(mapping: Any, keys: Iterable[str], *, context: str) -> None:
    """Reject outdated or incomplete analysis reports, and avoid silently exporting missing indicators as empty columns."""

    if not isinstance(mapping, Mapping):
        raise ValueError(f"{context} must be an object.")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(
            f"{context} missing required metrics {missing}. "
            "Re-run the offline analyze command from existing JSONL ledgers; "
            "model calls are not required."
        )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
   
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _metric_columns(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{name}": metrics.get(name)
        for name in ("tp", "fp", "tn", "fn", "n", "accuracy", "precision", "recall", "f1")
    }


def _ci_columns(uncertainty: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("accuracy", "precision", "recall", "f1"):
        value = uncertainty.get(metric, {})
        output[f"{metric}_ci_low"] = value.get("ci_low") if isinstance(value, Mapping) else None
        output[f"{metric}_ci_high"] = value.get("ci_high") if isinstance(value, Mapping) else None
    return output


def _rq1_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = report.get("datasets", {})
    if not isinstance(datasets, Mapping):
        raise ValueError("RQ1 report missing datasets object.")
    for dataset, models in sorted(datasets.items()):
        if not isinstance(models, Mapping):
            continue
        for model, value in sorted(models.items()):
            if not isinstance(value, Mapping):
                continue
            latency = value.get("latency_distribution_seconds", {})
            resources = value.get("resource_means", {})
            charge = value.get("cloud_charge", {})
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **_metric_columns("final", value.get("confusion", {})),
                    **_ci_columns(value.get("uncertainty", {})),
                    "latency_mean_s": _nested(latency, "mean"),
                    "latency_median_s": _nested(latency, "median"),
                    "latency_p95_s": _nested(latency, "p95"),
                    "cloud_input_tokens_mean": _nested(resources, "cloud_input_tokens"),
                    "cloud_output_tokens_mean": _nested(resources, "cloud_output_tokens"),
                    "cloud_calls_mean": _nested(resources, "cloud_calls"),
                    "local_calls_mean": _nested(resources, "local_calls"),
                    "local_tokens_mean": _nested(resources, "local_tokens"),
                    "total_tokens_mean": _nested(resources, "total_tokens"),
                    "cloud_latency_mean_s": _nested(resources, "cloud_latency_seconds"),
                    "local_wall_latency_mean_s": _nested(
                        resources, "local_wall_latency_seconds"
                    ),
                    "local_gpu_duration_mean_s": _nested(
                        resources, "local_ollama_duration_seconds"
                    ),
                    "local_load_duration_mean_s": _nested(
                        resources, "local_ollama_load_seconds"
                    ),
                    "cloud_charge_available": _nested(charge, "available"),
                    "cloud_charge_total": _nested(charge, "total"),
                }
            )
    if not rows:
        raise ValueError("RQ1 report contains no model results.")
    return rows


def _rq2_rows(
    dataset: str,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = report.get("policies", {})
    if not isinstance(policies, Mapping):
        raise ValueError(f"RQ2 report for {dataset} missing policies object.")
    # The published CSV is a complete machine-verifiable results table: the main strategy, control, and ablation are all retained. 
    # The subset needed for paper formatting should be selected by downstream plotting/formatting scripts and should not change the baseline of the published data.
    policy_items = sorted(policies.items())
    for policy, value in policy_items:
        if not isinstance(value, Mapping):
            continue
        transitions = value.get("transitions", {})
        routing = value.get("routing", {})
        if policy != "strong_only":
            _require_keys(
                routing,
                (
                    "routed_factoids",
                    "total_factoids",
                    "factoid_escalation_rate",
                    "routed_responses",
                    "total_responses",
                    "response_escalation_rate",
                ),
                context=f"RQ2 {dataset}/{policy} routing",
            )
            _require_keys(
                transitions,
                (
                    "rescued",
                    "harmed",
                    "routed_stage1_wrong",
                    "net_correct_gain",
                    "rescue_rate_response",
                    "error_conditional_rescue_rate",
                    "harm_rate_response",
                    "rescue_rate_all_responses",
                    "harm_rate_all_responses",
                ),
                context=f"RQ2 {dataset}/{policy} transitions",
            )
        latency = value.get("latency_distribution_seconds", {})
        resources = value.get("resource_means", {})
        charge = value.get("cloud_charge", {})
        rows.append(
            {
                "dataset": dataset,
                "policy": policy,
                **_metric_columns("stage1", value.get("stage1", {})),
                **_metric_columns("final", value.get("final", {})),
                **_ci_columns(value.get("uncertainty", {})),
                "routed_factoids": _nested(routing, "routed_factoids"),
                "total_factoids": _nested(routing, "total_factoids"),
                "factoid_escalation_rate": _nested(routing, "factoid_escalation_rate"),
                "routed_responses": _nested(routing, "routed_responses"),
                "total_responses": _nested(routing, "total_responses"),
                "response_escalation_rate": _nested(routing, "response_escalation_rate"),
                "rescued": _nested(transitions, "rescued"),
                "harmed": _nested(transitions, "harmed"),
                "routed_stage1_errors": _nested(transitions, "routed_stage1_wrong"),
                "net_correct_gain": _nested(transitions, "net_correct_gain"),               
                "rescue_rate_response": _nested(transitions, "rescue_rate_response"),
                "error_conditional_rescue_rate": _nested(
                    transitions, "error_conditional_rescue_rate"
                ),
                "harm_rate_response": _nested(transitions, "harm_rate_response"),
                "rescue_rate_all_responses": _nested(
                    transitions, "rescue_rate_all_responses"
                ),
                "harm_rate_all_responses": _nested(
                    transitions, "harm_rate_all_responses"
                ),
                "latency_mean_s": _nested(latency, "mean"),
                "latency_median_s": _nested(latency, "median"),
                "latency_p95_s": _nested(latency, "p95"),
                "cloud_tokens_mean": _nested(resources, "cloud_tokens"),
                "cloud_calls_mean": _nested(resources, "cloud_calls"),
                "local_calls_mean": _nested(resources, "local_calls"),
                "local_tokens_mean": _nested(resources, "local_tokens"),
                "total_tokens_mean": _nested(resources, "total_tokens"),
                "cloud_latency_mean_s": _nested(resources, "cloud_latency_seconds"),
                "local_wall_latency_mean_s": _nested(
                    resources, "local_wall_latency_seconds"
                ),
                "local_gpu_duration_mean_s": _nested(
                    resources, "local_ollama_duration_seconds"
                ),
                "local_load_duration_mean_s": _nested(
                    resources, "local_ollama_load_seconds"
                ),
                "routing_latency_mean_s": _nested(resources, "routing_latency_seconds"),
                "cloud_charge_available": _nested(charge, "available"),
                "cloud_charge_total": _nested(charge, "total"),
            }
        )
    if not rows:
        raise ValueError(f"RQ2 report for {dataset} contains no policy results.")
    return rows


def _rq3_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: list[tuple[str, Any]] = [("ALL", report.get("overall", {}))]
    by_category = report.get("by_category", {})
    if not isinstance(by_category, Mapping):
        raise ValueError("RQ3 report missing by_category object.")
    groups.extend(sorted((str(name), value) for name, value in by_category.items()))
    rows: list[dict[str, Any]] = []
    for category, value in groups:
        if not isinstance(value, Mapping):
            continue
        routing = value.get("routing", {})
        repair = value.get("repair", {})
        _require_keys(
            routing,
            (
                "routed",
                "escalation_rate",
                "missed_detection",
                "missed_detection_rate",
                "over_escalation",
                "over_escalation_rate",
                "success_pass",
                "success_pass_rate",
                "valid_escalation",
                "valid_escalation_rate",
            ),
            context=f"RQ3 {category} routing",
        )
        _require_keys(
            repair,
            (
                "rescued",
                "rescue_rate",
                "harmed",
                "harm_rate",
                "net_correct_gain",
                "error_conditional_rescue_rate",
                "escalation_efficiency_per_100",
            ),
            context=f"RQ3 {category} repair",
        )
        rows.append(
            {
                "category": category,
                "n": value.get("n"),
                "supported": value.get("supported"),
                "unsupported": value.get("unsupported"),
                **_metric_columns("stage1", value.get("stage1", {})),
                **_metric_columns("final", value.get("final", {})),
                "routed": _nested(routing, "routed"),
                "escalation_rate": _nested(routing, "escalation_rate"),
                "missed_detection": _nested(routing, "missed_detection"),
                "missed_detection_rate": _nested(routing, "missed_detection_rate"),
                "over_escalation": _nested(routing, "over_escalation"),
                "over_escalation_rate": _nested(routing, "over_escalation_rate"),
                "success_pass": _nested(routing, "success_pass"),
                "success_pass_rate": _nested(routing, "success_pass_rate"),
                "valid_escalation": _nested(routing, "valid_escalation"),
                "valid_escalation_rate": _nested(routing, "valid_escalation_rate"),
                "rescued": _nested(repair, "rescued"),
                "rescue_rate": _nested(repair, "rescue_rate"),
                "harmed": _nested(repair, "harmed"),
                "harm_rate": _nested(repair, "harm_rate"),
                "net_correct_gain": _nested(repair, "net_correct_gain"),
                "error_conditional_rescue_rate": _nested(
                    repair, "error_conditional_rescue_rate"
                ),
                "escalation_efficiency_per_100": _nested(
                    repair, "escalation_efficiency_per_100"
                ),
                "failure_families_json": json.dumps(
                    value.get("failure_families", {}), ensure_ascii=False, sort_keys=True
                ),
            }
        )
    if not rows:
        raise ValueError("RQ3 report contains no results.")
    return rows


def export_publication_tables(config: dict[str, Any], *, repository_root: str | Path) -> Path:
    """Export RQ1/RQ2/RQ3 tables; fail immediately if any input is missing, do not write placeholder numbers."""

    root = Path(repository_root)
    output = Path(str(config["output_directory"]))
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite publication tables: {output}")

    rq1_path, rq1_report = _load_report(root, str(config["rq1_report"]))
    rq3_path, rq3_report = _load_report(root, str(config["rq3_report"]))
    rq2_inputs = config.get("rq2_reports")
    if not isinstance(rq2_inputs, list) or not rq2_inputs:
        raise ValueError("publication config requires a non-empty rq2_reports array.")
    rq2_reports: list[tuple[str, Path, dict[str, Any]]] = []
    for item in rq2_inputs:
        if not isinstance(item, Mapping):
            raise TypeError("Each rq2_reports item must be an object.")
        path, report = _load_report(root, str(item["path"]))
        rq2_reports.append((str(item["dataset"]), path, report))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        rq1_rows = _rq1_rows(rq1_report)
        rq2_rows = [
            row
            for dataset, _, report in rq2_reports
            for row in _rq2_rows(dataset, report)
        ]
        rq3_rows = _rq3_rows(rq3_report)
        _write_csv(temporary / "rq1_effectiveness_resources.csv", rq1_rows, list(rq1_rows[0]))
        _write_csv(temporary / "rq2_policy_comparison.csv", rq2_rows, list(rq2_rows[0]))
        _write_csv(temporary / "rq3_category_analysis.csv", rq3_rows, list(rq3_rows[0]))

        input_paths = [rq1_path, rq3_path, *(path for _, path, _ in rq2_reports)]
        manifest = {
            "status": "complete",
            "analysis_schema_version": "2026-08-22.metrics-v2",
            "source_reports": [
                {
                    "path": path.resolve().relative_to(root.resolve()).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in input_paths
            ],
            "tables": {
                "rq1_effectiveness_resources.csv": len(rq1_rows),
                "rq2_policy_comparison.csv": len(rq2_rows),
                "rq3_category_analysis.csv": len(rq3_rows),
            },
            "notes": [
                "All values are exported from completed analysis JSON reports.",
                "RQ2 contains every policy in each source analysis report, including primary strategies, controls, and trigger ablations.",
                "Blank charge fields mean prices were not supplied; they are not zero-cost claims.",
                "RQ1 contains four verifier models across two datasets, i.e. eight run configurations.",
            ],
        }
        atomic_write_json(temporary / "publication_manifest.json", manifest)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
