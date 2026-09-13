
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ledger import atomic_write_json
from .labels import BinaryTarget
from .policy_analysis import _comparison, _confusion, _read_jsonl


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _portable_path(root: Path, path: Path) -> str:
    """Return a repository-relative path when the target is inside the package."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _portable_reference(root: Path, value: Any) -> Any:
    """Normalize legacy in-package absolute references without exposing a workstation path."""

    if not isinstance(value, str) or not value:
        return value
    normalized = value.replace("\\", "/")
    marker = f"/{root.name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _categorical_agreement(first: Sequence[Any], second: Sequence[Any]) -> dict[str, Any]:
    if len(first) != len(second) or not first:
        raise ValueError("Agreement inputs must have the same non-zero length.")
    first_values = [str(value) for value in first]
    second_values = [str(value) for value in second]
    n = len(first_values)
    observed_count = sum(a == b for a, b in zip(first_values, second_values, strict=True))
    first_counts = Counter(first_values)
    second_counts = Counter(second_values)
    labels = sorted(set(first_counts).union(second_counts))
    expected = sum(
        (first_counts[label] / n) * (second_counts[label] / n) for label in labels
    )
    kappa = None if math.isclose(1.0 - expected, 0.0) else (
        observed_count / n - expected
    ) / (1.0 - expected)
    return {
        "n": n,
        "agreement_count": observed_count,
        "agreement_rate": observed_count / n,
        "cohen_kappa": kappa,
        "labels": labels,
    }


def _response_rows(run_directory: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(run_directory / "responses.jsonl")
    return [
        {
            "response_id": str(row["response_id"]),
            "question_id": str(row["question_id"]),
            "gold_hallucinated": str(row["gold_target"]) == BinaryTarget.UNSUPPORTED.value,
            "final_hallucinated": bool(row["final_hallucinated"]),
        }
        for row in rows
    ]


def _probe_verdicts(run_directory: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(run_directory / "factoids.jsonl"):
        factoid_id = str(row["factoid_id"])
        bundle = row["stage1"]
        values = (
            str(bundle["original"]),
            str(bundle["synonyms"][0]),
            str(bundle["synonyms"][1]),
            str(bundle["antonyms"][0]),
            str(bundle["antonyms"][1]),
        )
        output[factoid_id] = {
            "probe_hash": str(row["probe_hash"]),
            "response_id": str(row["response_id"]),
            "verdicts": values,
        }
    return output


def run_shared_model_bias_analysis(
    config: dict[str, Any], *, repository_root: str | Path
) -> Path:

    root = Path(repository_root)
    repetitions = int(config.get("bootstrap_repetitions", 2000))
    seed = int(config.get("bootstrap_seed", 42))
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("shared-bias config requires a non-empty datasets array.")

    report_datasets: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for dataset_item in datasets:
        dataset = str(dataset_item["name"])
        models = dataset_item.get("models")
        if not isinstance(models, list) or len(models) < 2:
            raise ValueError(f"{dataset}: at least two verifier models are required.")
        response_by_model: dict[str, list[dict[str, Any]]] = {}
        probes_by_model: dict[str, dict[str, dict[str, Any]]] = {}
        shared_labels = [
            str(item["label"]) for item in models if bool(item.get("shared_model", False))
        ]
        if len(shared_labels) != 1:
            raise ValueError(f"{dataset}: exactly one shared_model verifier is required.")
        shared_label = shared_labels[0]
        independent_labels = [
            str(item["label"]) for item in models if not bool(item.get("shared_model", False))
        ]
        for item in models:
            label = str(item["label"])
            run_directory = _resolve(root, str(item["run_directory"]))
            response_by_model[label] = _response_rows(run_directory)
            probes_by_model[label] = _probe_verdicts(run_directory)

        reference_ids = {row["response_id"] for row in response_by_model[shared_label]}
        reference_factoids = set(probes_by_model[shared_label])
        for label in response_by_model:
            if {row["response_id"] for row in response_by_model[label]} != reference_ids:
                raise ValueError(f"{dataset}: response IDs differ for {label}.")
            if set(probes_by_model[label]) != reference_factoids:
                raise ValueError(f"{dataset}: factoid IDs differ for {label}.")
            for factoid_id in reference_factoids:
                if (
                    probes_by_model[label][factoid_id]["probe_hash"]
                    != probes_by_model[shared_label][factoid_id]["probe_hash"]
                ):
                    raise ValueError(
                        f"{dataset}: probe hash mismatch for {label}/{factoid_id}."
                    )

        rows_by_model = {
            label: {row["response_id"]: row for row in rows}
            for label, rows in response_by_model.items()
        }
        ensemble_rows: list[dict[str, Any]] = []
        for response_id in sorted(reference_ids):
            votes = [
                bool(rows_by_model[label][response_id]["final_hallucinated"])
                for label in independent_labels
            ]
            predicted = sum(votes) > len(votes) / 2
            shared_row = rows_by_model[shared_label][response_id]
            ensemble_rows.append(
                {
                    "response_id": response_id,
                    "question_id": shared_row["question_id"],
                    "gold_hallucinated": shared_row["gold_hallucinated"],
                    "final_hallucinated": predicted,
                }
            )

        response_agreement: dict[str, Any] = {}
        probe_agreement: dict[str, Any] = {}
        comparisons: dict[str, Any] = {}
        shared_rows = response_by_model[shared_label]
        for label in independent_labels:
            independent_rows = response_by_model[label]
            response_agreement[label] = _categorical_agreement(
                [
                    row["final_hallucinated"]
                    for row in sorted(shared_rows, key=lambda item: item["response_id"])
                ],
                [
                    row["final_hallucinated"]
                    for row in sorted(independent_rows, key=lambda item: item["response_id"])
                ],
            )
            shared_probe_values: list[str] = []
            independent_probe_values: list[str] = []
            for factoid_id in sorted(reference_factoids):
                shared_probe_values.extend(
                    probes_by_model[shared_label][factoid_id]["verdicts"]
                )
                independent_probe_values.extend(
                    probes_by_model[label][factoid_id]["verdicts"]
                )
            probe_agreement[label] = _categorical_agreement(
                shared_probe_values, independent_probe_values
            )
            comparisons[label] = _comparison(
                label,
                independent_rows,
                shared_label,
                shared_rows,
                repetitions=repetitions,
                seed=seed,
            )

        ensemble_agreement = _categorical_agreement(
            [
                row["final_hallucinated"]
                for row in sorted(shared_rows, key=lambda item: item["response_id"])
            ],
            [
                row["final_hallucinated"]
                for row in sorted(ensemble_rows, key=lambda item: item["response_id"])
            ],
        )
        ensemble_comparison = _comparison(
            "Independent-local-majority",
            ensemble_rows,
            shared_label,
            shared_rows,
            repetitions=repetitions,
            seed=seed,
        )
        model_confusions = {
            label: _confusion(rows) for label, rows in response_by_model.items()
        }
        ensemble_confusion = _confusion(ensemble_rows)

        report_datasets[dataset] = {
            "n_responses": len(reference_ids),
            "n_factoids": len(reference_factoids),
            "n_probe_verdicts_per_model": 5 * len(reference_factoids),
            "shared_model_verifier": shared_label,
            "independent_verifiers": independent_labels,
            "model_confusions": model_confusions,
            "independent_majority_confusion": ensemble_confusion,
            "response_prediction_agreement_with_shared_model": response_agreement,
            "probe_verdict_agreement_with_shared_model": probe_agreement,
            "independent_majority_agreement_with_shared_model": ensemble_agreement,
            "paired_comparisons_independent_vs_shared": comparisons,
            "paired_comparison_majority_vs_shared": ensemble_comparison,
        }
        for label in independent_labels:
            csv_rows.append(
                {
                    "dataset": dataset,
                    "comparison": label,
                    "unit": "response",
                    **response_agreement[label],
                    "independent_accuracy": model_confusions[label]["accuracy"],
                    "independent_f1": model_confusions[label]["f1"],
                    "deepseek_accuracy": model_confusions[shared_label]["accuracy"],
                    "deepseek_f1": model_confusions[shared_label]["f1"],
                }
            )
            csv_rows.append(
                {
                    "dataset": dataset,
                    "comparison": label,
                    "unit": "probe_verdict",
                    **probe_agreement[label],
                }
            )
        csv_rows.append(
            {
                "dataset": dataset,
                "comparison": "Independent-local-majority",
                "unit": "response",
                **ensemble_agreement,
                "independent_accuracy": ensemble_confusion["accuracy"],
                "independent_f1": ensemble_confusion["f1"],
                "deepseek_accuracy": model_confusions[shared_label]["accuracy"],
                "deepseek_f1": model_confusions[shared_label]["f1"],
            }
        )

    report = {
        "analysis_type": "independent_verifier_sensitivity_on_deepseek_generated_probes",
        "scope_limitation": (
            "This analysis changes verifier identity while keeping the DeepSeek-generated "
            "factoid decomposition and mutations frozen. It tests shared-verifier sensitivity "
            "but is not an independent probe-generation experiment."
        ),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "datasets": report_datasets,
    }
    output = _resolve(root, str(config["output_path"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite shared-bias report: {output}")
    atomic_write_json(output, report)
    csv_path = _resolve(root, str(config["csv_path"]))
    _write_csv(csv_path, csv_rows)
    return output


def run_human_audit_analysis(
    config: dict[str, Any], *, repository_root: str | Path
) -> Path:

    root = Path(repository_root)
    a_path = _resolve(root, str(config["annotator_a"]))
    b_path = _resolve(root, str(config["annotator_b"]))
    adjudication_path = _resolve(root, str(config["adjudication"]))
    dataset_path = _resolve(root, str(config["frozen_dataset"]))

    with a_path.open(encoding="utf-8-sig", newline="") as handle:
        annotator_a = list(csv.DictReader(handle))
    with b_path.open(encoding="utf-8-sig", newline="") as handle:
        annotator_b = list(csv.DictReader(handle))
    with adjudication_path.open(encoding="utf-8-sig", newline="") as handle:
        adjudication = [row for row in csv.DictReader(handle) if str(row.get("id", "")).strip()]
    frozen = _read_jsonl(dataset_path)
    if len(annotator_a) != 700 or len(annotator_b) != 700 or len(frozen) != 700:
        raise ValueError("Main-700 human audit requires exactly 700 A, B, and frozen rows.")

    a_by_id = {str(row["id"]): row for row in annotator_a}
    b_by_id = {
        f"MAIN_{int(row['source_order']):04d}": row for row in annotator_b
    }
    adjudication_by_id = {str(row["id"]): row for row in adjudication}
    frozen_by_id = {str(row["id"]): row for row in frozen}
    expected_ids = set(frozen_by_id)
    if set(a_by_id) != expected_ids or set(b_by_id) != expected_ids:
        raise ValueError("Annotator IDs do not align with the frozen Main-700 IDs.")

    fields = {
        "primary_category": (
            [a_by_id[item]["primary_category"] for item in sorted(expected_ids)],
            [b_by_id[item]["primary_category"] for item in sorted(expected_ids)],
        ),
        "context_support": (
            [a_by_id[item]["context_support"].upper() for item in sorted(expected_ids)],
            [b_by_id[item]["context_support"].upper() for item in sorted(expected_ids)],
        ),
        "question_relevance_descriptive": (
            [a_by_id[item]["question_relevance"].upper() for item in sorted(expected_ids)],
            [b_by_id[item]["question_relevance"].upper() for item in sorted(expected_ids)],
        ),
        "atomicity": (
            [a_by_id[item]["atomicity"].upper() for item in sorted(expected_ids)],
            [b_by_id[item]["atomicity"].upper() for item in sorted(expected_ids)],
        ),
        "evidence_clarity_descriptive": (
            [a_by_id[item]["evidence_clarity"].upper() for item in sorted(expected_ids)],
            [b_by_id[item]["evidence_clarity"].upper() for item in sorted(expected_ids)],
        ),
    }
    agreement = {
        name: _categorical_agreement(first, second)
        for name, (first, second) in fields.items()
    }
    category_support_disagreements = {
        item
        for item in expected_ids
        if (
            a_by_id[item]["primary_category"] != b_by_id[item]["primary_category"]
            or a_by_id[item]["context_support"].upper()
            != b_by_id[item]["context_support"].upper()
        )
    }
    core_disagreements = category_support_disagreements.union(
        item
        for item in expected_ids
        if a_by_id[item]["atomicity"].upper()
        != b_by_id[item]["atomicity"].upper()
    )
    unresolved = sorted(core_disagreements.difference(adjudication_by_id))
    if unresolved:
        raise ValueError(f"Unadjudicated core disagreements: {unresolved[:10]}")

    category_mismatch: list[str] = []
    support_mismatch: list[str] = []
    invalid_adjudicated_support: list[str] = []
    text_mismatch: list[str] = []
    for item in sorted(expected_ids):
        a_row = a_by_id[item]
        b_row = b_by_id[item]
        final_row = frozen_by_id[item]
        adjudicated = adjudication_by_id.get(item)
        expected_category = (
            adjudicated["adjudicator_final_category"]
            if adjudicated
            else a_row["primary_category"]
        )
        expected_support = a_row["context_support"].upper()
        if adjudicated:
            candidate_support = str(
                adjudicated.get("adjudicator_context_support", "")
            ).strip().upper()
            if candidate_support in {"TRUE", "FALSE"}:
                expected_support = candidate_support
            else:
                invalid_adjudicated_support.append(item)
        final_support = "TRUE" if bool(final_row["context_support"]) else "FALSE"
        if str(final_row["category"]) != expected_category:
            category_mismatch.append(item)
        if item not in invalid_adjudicated_support and final_support != expected_support:
            support_mismatch.append(item)
        canonical_final = tuple(
            str(final_row[name]).strip() for name in ("question", "context", "factoid")
        )
        canonical_a = tuple(
            str(a_row[name]).strip() for name in ("question", "context", "factoid")
        )
        canonical_b = tuple(
            str(b_row[name]).strip() for name in ("question", "context", "factoid")
        )
        if canonical_final != canonical_a or canonical_final != canonical_b:
            text_mismatch.append(item)

    relevance_counts = Counter(
        (
            a_by_id[item]["question_relevance"].upper(),
            b_by_id[item]["question_relevance"].upper(),
        )
        for item in expected_ids
    )
    report = {
        "audit_scope": {
            "hard_final_fields": ["atomicity", "primary_category", "context_support"],
            "descriptive_non_filter_fields": [
                "question_relevance",
                "evidence_clarity",
            ],
            "question_relevance_filtering": False,
        },
        "files": {
            "annotator_a": _portable_path(root, a_path),
            "annotator_b": _portable_path(root, b_path),
            "adjudication": _portable_path(root, adjudication_path),
            "frozen_dataset": _portable_path(root, dataset_path),
        },
        "rows": {
            "annotator_a": len(annotator_a),
            "annotator_b": len(annotator_b),
            "adjudication_nonempty": len(adjudication),
            "frozen": len(frozen),
        },
        "agreement": agreement,
        "core_hard_field_disagreements": len(core_disagreements),
        "core_category_or_support_disagreements": len(category_support_disagreements),
        "unadjudicated_core_disagreements": unresolved,
        "final_consistency": {
            "atomicity_status": "verified_by_double_review",
            "atomicity_fail_ids": sorted(
                item
                for item in expected_ids
                if a_by_id[item]["atomicity"].upper() != "PASS"
                or b_by_id[item]["atomicity"].upper() != "PASS"
            ),
            "category_mismatch_ids": category_mismatch,
            "context_support_status": (
                "unavailable_due_to_invalid_adjudication_export"
                if invalid_adjudicated_support
                else "verified"
            ),
            "invalid_adjudicated_context_support_ids": invalid_adjudicated_support,
            "context_support_mismatch_ids_among_valid_rows": support_mismatch,
            "trimmed_text_mismatch_ids": text_mismatch,
        },
        "question_relevance_descriptive_pair_counts": {
            f"A={first},B={second}": count
            for (first, second), count in sorted(relevance_counts.items())
        },
    }
    output = _resolve(root, str(config["output_path"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite human audit report: {output}")
    atomic_write_json(output, report)
    csv_path = _resolve(root, str(config["csv_path"]))
    csv_rows = [
        {
            "field": field,
            **value,
            "hard_filter": field in {"atomicity", "primary_category", "context_support"},
        }
        for field, value in agreement.items()
    ]
    _write_csv(csv_path, csv_rows)
    return output


def _token_price_breakdown(
    call: Mapping[str, Any], pricing: Mapping[str, Any]
) -> dict[str, Any] | None:
    backend = str(call.get("backend", ""))
    if backend != "openai_compatible":
        return None
    model = str(call.get("model_id") or call.get("requested_model_id") or "")
    model_price = pricing.get("models", {}).get(model)
    if not isinstance(model_price, Mapping):
        return None
    usage = call.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", call.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", call.get("output_tokens", 0)) or 0)
    details = call.get("usage_details") or {}
    cached_value = details.get("prompt_cache_hit_tokens")
    uncached_value = details.get("prompt_cache_miss_tokens")
    details_available = cached_value is not None and uncached_value is not None
    cached_tokens = int(cached_value or 0)
    uncached_tokens = int(uncached_value or 0)
    if not details_available:
        uncached_tokens = input_tokens
    unclassified = max(0, input_tokens - cached_tokens - uncached_tokens)
    uncached_tokens += unclassified
    cached_price = float(model_price["cached_input_per_million"])
    uncached_price = float(model_price["uncached_input_per_million"])
    output_price = float(model_price["output_per_million"])
    cached_cost = cached_tokens * cached_price / 1_000_000.0
    uncached_cost = uncached_tokens * uncached_price / 1_000_000.0
    output_cost = output_tokens * output_price / 1_000_000.0
    return {
        "model": model,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        "output_tokens": output_tokens,
        "cached_input_cost": cached_cost,
        "uncached_input_cost": uncached_cost,
        "output_cost": output_cost,
        "total_cost": cached_cost + uncached_cost + output_cost,
        "usage_details_available": details_available,
    }


def run_cloud_cost_audit(
    config: dict[str, Any], *, repository_root: str | Path
) -> Path:

    root = Path(repository_root)
    pricing = config.get("pricing")
    if not isinstance(pricing, Mapping):
        raise ValueError("cost audit requires a pricing object.")
    ledgers = config.get("call_ledgers")
    if not isinstance(ledgers, list) or not ledgers:
        raise ValueError("cost audit requires non-empty call_ledgers.")

    summary_rows: list[dict[str, Any]] = []
    unique_actual: dict[str, dict[str, Any]] = {}
    for item in ledgers:
        scope = str(item["scope"])
        path = _resolve(root, str(item["path"]))
        rows = _read_jsonl(path)
        policy = Counter()
        actual = Counter()
        remote_rows = 0
        executed_rows = 0
        missing_details = 0
        for row in rows:
            breakdown = _token_price_breakdown(row, pricing)
            if breakdown is None:
                continue
            remote_rows += 1
            if not breakdown["usage_details_available"]:
                missing_details += 1
            for key in (
                "cached_input_tokens",
                "uncached_input_tokens",
                "output_tokens",
                "cached_input_cost",
                "uncached_input_cost",
                "output_cost",
                "total_cost",
            ):
                policy[key] += breakdown[key]
            if not bool(row.get("cache_hit", False)):
                executed_rows += 1
                unique_key = str(
                    row.get("provider_response_id")
                    or f"{row.get('request_hash')}|{row.get('response_hash')}"
                )
                if unique_key not in unique_actual:
                    unique_actual[unique_key] = breakdown
                    for key in (
                        "cached_input_tokens",
                        "uncached_input_tokens",
                        "output_tokens",
                        "cached_input_cost",
                        "uncached_input_cost",
                        "output_cost",
                        "total_cost",
                    ):
                        actual[key] += breakdown[key]
        summary_rows.append(
            {
                "scope": scope,
                "path": _portable_path(root, path),
                "remote_ledger_rows": remote_rows,
                "non_application_cache_rows": executed_rows,
                "rows_without_provider_cache_details": missing_details,
                **{f"policy_equivalent_{key}": value for key, value in policy.items()},
                **{f"actual_executed_unique_{key}": value for key, value in actual.items()},
            }
        )

    total_actual = Counter()
    for breakdown in unique_actual.values():
        for key in (
            "cached_input_tokens",
            "uncached_input_tokens",
            "output_tokens",
            "cached_input_cost",
            "uncached_input_cost",
            "output_cost",
            "total_cost",
        ):
            total_actual[key] += breakdown[key]
    total_policy = Counter()
    for row in summary_rows:
        for key in (
            "cached_input_tokens",
            "uncached_input_tokens",
            "output_tokens",
            "cached_input_cost",
            "uncached_input_cost",
            "output_cost",
            "total_cost",
        ):
            total_policy[key] += row.get(f"policy_equivalent_{key}", 0)

    report = {
        "pricing": dict(pricing),
        "accounting_definitions": {
            "policy_equivalent": (
                "All cloud calls represented by each ledger, including application-cache "
                "replays, priced as if that policy were executed independently."
            ),
            "actual_executed_unique": (
                "Only non-application-cache cloud calls, deduplicated across ledgers by "
                "provider response ID or request/response hash."
            ),
            "provider_prompt_cache": (
                "Input tokens use provider-reported prompt_cache_hit_tokens and "
                "prompt_cache_miss_tokens. Missing details are conservatively priced as uncached."
            ),
        },
        "ledgers": summary_rows,
        "totals": {
            "policy_equivalent_sum_across_listed_ledgers": dict(total_policy),
            "actual_executed_unique_across_listed_ledgers": dict(total_actual),
            "unique_executed_cloud_responses": len(unique_actual),
        },
    }
    output = _resolve(root, str(config["output_path"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cloud cost audit: {output}")
    atomic_write_json(output, report)
    csv_path = _resolve(root, str(config["csv_path"]))
    _write_csv(csv_path, summary_rows)
    return output


def run_preparation_quality_audit(
    config: dict[str, Any], *, repository_root: str | Path
) -> Path:
    """Summary unresolved、warnings and response-level exclusions."""

    root = Path(repository_root)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("preparation audit requires a non-empty datasets array.")
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for item in datasets:
        name = str(item["name"])
        manifest_path = _resolve(root, str(item["prepared_manifest"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        warning_types = Counter(
            warning
            for row in manifest.get("probe_warnings", [])
            for warning in row.get("warnings", [])
        )
        excluded_ids: set[str] = set()
        run_statuses: list[str] = []
        for value in item.get("run_manifests", []):
            run_path = _resolve(root, str(value))
            run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
            run_statuses.append(str(run_manifest.get("status")))
            excluded_ids.update(
                str(response_id)
                for response_id in run_manifest.get("excluded_zero_probe_responses", [])
            )
        row = {
            "dataset": name,
            "preparation_status": manifest.get("status"),
            "source_records": manifest.get("source_records"),
            "prepared_responses": manifest.get("prepared_responses"),
            "source_factoids": manifest.get("source_factoids"),
            "prepared_factoids": manifest.get("prepared_factoids"),
            "unresolved_decompositions": manifest.get("unresolved_decompositions"),
            "unresolved_decomposition_rate": manifest.get("unresolved_decomposition_rate"),
            "unresolved_mutations": manifest.get("unresolved_mutations"),
            "unresolved_mutation_rate": manifest.get("unresolved_mutation_rate"),
            "probe_warning_factoids": manifest.get("probe_warning_count"),
            "responses_without_valid_probes": len(
                manifest.get("responses_without_valid_probes", [])
            ),
            "excluded_zero_probe_responses": len(excluded_ids),
            "all_run_manifests_complete": bool(run_statuses)
            and all(status == "complete" for status in run_statuses),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "prepared_sha256": manifest.get("prepared_sha256"),
        }
        rows.append(row)
        detail[name] = {
            "prepared_manifest": _portable_path(root, manifest_path),
            "warning_type_counts": dict(sorted(warning_types.items())),
            "responses_without_valid_probes": manifest.get(
                "responses_without_valid_probes", []
            ),
            "excluded_zero_probe_response_ids": sorted(excluded_ids),
            "run_statuses": run_statuses,
            "unresolved_mutations_path": _portable_reference(
                root, manifest.get("unresolved_mutations_path")
            ),
            "unresolved_decompositions_path": _portable_reference(
                root, manifest.get("unresolved_decompositions_path")
            ),
        }
    report = {
        "denominator_rule": (
            "Effectiveness metrics use successfully evaluated responses. Source count, "
            "zero-probe exclusions, unresolved decomposition/mutation counts, and warning "
            "factoids are reported separately and are never silently folded into accuracy."
        ),
        "datasets": rows,
        "details": detail,
    }
    output = _resolve(root, str(config["output_path"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preparation audit: {output}")
    atomic_write_json(output, report)
    csv_path = _resolve(root, str(config["csv_path"]))
    _write_csv(csv_path, rows)
    return output
