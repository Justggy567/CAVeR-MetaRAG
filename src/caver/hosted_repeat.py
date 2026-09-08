"""Reproducible sampling, configuration generation, and consistency statistics for hosted DeepSeek."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .datasets import file_sha256
from .ledger import atomic_write_json
from .policy_analysis import _confusion, _read_jsonl
from .reviewer_analysis import _categorical_agreement, _resolve, _token_price_breakdown


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite JSONL: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _valid_prepared(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("probe_sets")]


def prepare_hosted_repeat_package(
    config: dict[str, Any], *, repository_root: str | Path
) -> Path:
    

    root = Path(repository_root)
    output = _resolve(root, str(config["output_directory"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite hosted repeat package: {output}")
    repeats = int(config.get("repeats", 3))
    if repeats < 2:
        raise ValueError("Hosted repeat analysis requires at least two repeats.")
    seed = int(config.get("sample_seed", 20260829))
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("hosted repeat package requires a non-empty datasets array.")
    stage1 = config.get("stage1")
    if not isinstance(stage1, Mapping) or stage1.get("backend") != "openai_compatible":
        raise ValueError("hosted repeat package requires an openai_compatible stage1.")

    output.mkdir(parents=True)
    prepared_directory = output / "prepared"
    config_directory = output / "configs"
    prepared_directory.mkdir()
    config_directory.mkdir()
    manifest_datasets: list[dict[str, Any]] = []
    generated_configs: list[str] = []
    pricing = config.get("pricing", {})
    for dataset_index, item in enumerate(datasets):
        name = str(item["name"])
        slug = str(item["slug"])
        source_path = _resolve(root, str(item["source_prepared_path"]))
        source_rows = _read_jsonl(source_path)
        valid_rows = _valid_prepared(source_rows)
        by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            by_question[str(row["question_id"])].append(row)
        eligible_questions = sorted(by_question)
        required_cluster_size = item.get("required_cluster_size")
        if required_cluster_size is not None:
            eligible_questions = [
                question_id
                for question_id in eligible_questions
                if len(by_question[question_id]) == int(required_cluster_size)
            ]
        sample_mode = str(item.get("sample_mode", "question_sample"))
        if sample_mode == "all":
            selected_questions = eligible_questions
        elif sample_mode == "question_sample":
            count = int(item["question_count"])
            if len(eligible_questions) < count:
                raise ValueError(
                    f"{name}: requested {count} question clusters, only "
                    f"{len(eligible_questions)} are eligible."
                )
            generator = random.Random(seed + dataset_index)
            selected_questions = sorted(generator.sample(eligible_questions, count))
        else:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        selected_rows = [
            row
            for question_id in selected_questions
            for row in by_question[question_id]
        ]
        selected_ids = {str(row["response_id"]) for row in selected_rows}
        prepared_path = prepared_directory / f"{slug}.hosted_repeat.probes.jsonl"
        _write_jsonl(prepared_path, selected_rows)

        reference_calls_path = _resolve(root, str(item["reference_calls_path"]))
        forecast = Counter()
        for call in _read_jsonl(reference_calls_path):
            if str(call.get("response_id")) not in selected_ids:
                continue
            breakdown = _token_price_breakdown(call, pricing)
            if breakdown is None:
                continue
            for key in (
                "cached_input_tokens",
                "uncached_input_tokens",
                "output_tokens",
                "cached_input_cost",
                "uncached_input_cost",
                "output_cost",
                "total_cost",
            ):
                forecast[key] += breakdown[key]

        dataset_configs: list[str] = []
        for repeat in range(1, repeats + 1):
            relative_prepared = prepared_path.relative_to(root)
            cache_path = (
                output
                / "fresh_caches"
                / slug
                / f"replicate_{repeat:02d}.jsonl"
            )
            run_config = {
                "prepared_path": str(relative_prepared).replace("\\", "/"),
                "output_root": str(
                    (output / "runs" / slug).relative_to(root)
                ).replace("\\", "/"),
                "run_id": f"replicate_{repeat:02d}",
                "resume": True,
                "routing_policy": "no_escalation",
                "threshold": float(config.get("threshold", 0.5)),
                "scoring_profile": "paper_five_statement_mean",
                "response_aggregation": "max_factoid_score",
                "stage1": {
                    **dict(stage1),
                    "cache_path": str(cache_path.relative_to(root)).replace("\\", "/"),
                },
                "stage2": None,
                "environment": {
                    "concurrency": 1,
                    "execution": "sequential",
                    "repeat_design": "fresh application cache per replicate",
                },
                "pricing": config.get("run_pricing", {}),
            }
            run_config_path = config_directory / f"{slug}_replicate_{repeat:02d}.json"
            run_config_path.write_text(
                json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            relative_config = str(run_config_path.relative_to(root)).replace("\\", "/")
            dataset_configs.append(relative_config)
            generated_configs.append(relative_config)

        manifest_datasets.append(
            {
                "name": name,
                "slug": slug,
                "source_prepared_path": str(source_path.resolve()),
                "source_prepared_sha256": file_sha256(source_path),
                "selected_prepared_path": str(prepared_path.resolve()),
                "selected_prepared_sha256": file_sha256(prepared_path),
                "source_valid_responses": len(valid_rows),
                "eligible_question_clusters": len(eligible_questions),
                "selected_question_clusters": len(selected_questions),
                "selected_responses": len(selected_rows),
                "selected_factoids": sum(len(row["probe_sets"]) for row in selected_rows),
                "selected_question_ids": selected_questions,
                "selected_response_ids": sorted(selected_ids),
                "reference_calls_path": str(reference_calls_path.resolve()),
                "forecast_per_replicate": dict(forecast),
                "forecast_all_replicates_total_cost": (
                    float(forecast["total_cost"]) * repeats
                ),
                "run_configs": dataset_configs,
            }
        )

    guide_lines = [
        "# Hosted DeepSeek repeat runs",
        "",
        "The package uses frozen probes and a fresh application cache for each replicate.",
        "Do not copy a primary-run cache into any replicate cache path.",
        "",
        "Run the following commands from the repository root after setting DEEPSEEK_API_KEY:",
        "",
    ]
    guide_lines.extend(f"    python main.py run --config {path}" for path in generated_configs)
    guide_lines.extend(
        [
            "",
            "After all run manifests report status=complete:",
            "",
            (
                "    python main.py analyze-hosted-repeats --config "
                + str(config["analysis_config_path"])
            ),
            "",
        ]
    )
    (output / "RUN_GUIDE.md").write_text("\n".join(guide_lines), encoding="utf-8")
    manifest = {
        "status": "package_complete_api_calls_not_started",
        "repeats": repeats,
        "sample_seed": seed,
        "stage1_model_id": stage1.get("model_id"),
        "fresh_cache_required": True,
        "pricing": pricing,
        "datasets": manifest_datasets,
        "generated_run_configs": generated_configs,
    }
    atomic_write_json(output / "package_manifest.json", manifest)
    return output


def analyze_hosted_repeats(
    config: dict[str, Any], *, repository_root: str | Path
) -> Path:
    """Aggregate prediction stability across repeated runs; only fully completed runs are accepted."""

    root = Path(repository_root)
    package_manifest_path = _resolve(root, str(config["package_manifest"]))
    package = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    repeats = int(package["repeats"])
    report_datasets: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for dataset in package["datasets"]:
        slug = str(dataset["slug"])
        expected_ids = set(str(value) for value in dataset["selected_response_ids"])
        replicate_predictions: dict[str, dict[str, bool]] = {}
        replicate_summaries: dict[str, Any] = {}
        for repeat in range(1, repeats + 1):
            label = f"replicate_{repeat:02d}"
            run_directory = _resolve(
                root,
                str(Path(config["runs_root"]) / slug / label),
            )
            manifest = json.loads(
                (run_directory / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("status") != "complete":
                raise ValueError(f"{slug}/{label} is not complete.")
            responses = _read_jsonl(run_directory / "responses.jsonl")
            calls = _read_jsonl(run_directory / "calls.jsonl")
            if {str(row["response_id"]) for row in responses} != expected_ids:
                raise ValueError(f"{slug}/{label} response IDs do not match the package.")
            predictions = {
                str(row["response_id"]): bool(row["final_hallucinated"])
                for row in responses
            }
            replicate_predictions[label] = predictions
            metric_rows = [
                {
                    "response_id": str(row["response_id"]),
                    "question_id": str(row["question_id"]),
                    "gold_hallucinated": str(row["gold_target"]) == "UNSUPPORTED",
                    "final_hallucinated": bool(row["final_hallucinated"]),
                }
                for row in responses
            ]
            confusion = _confusion(metric_rows)
            cloud_calls = [
                row for row in calls if str(row.get("backend")) == "openai_compatible"
            ]
            replicate_summaries[label] = {
                "confusion": confusion,
                "cloud_ledger_rows": len(cloud_calls),
                "application_cache_hits": sum(
                    bool(row.get("cache_hit", False)) for row in cloud_calls
                ),
                "provider_fingerprints": sorted(
                    {
                        str(row.get("system_fingerprint"))
                        for row in cloud_calls
                        if row.get("system_fingerprint")
                    }
                ),
                "returned_model_ids": sorted(
                    {
                        str(row.get("returned_model_id"))
                        for row in cloud_calls
                        if row.get("returned_model_id")
                    }
                ),
                "manifest": {
                    "config_hash": manifest.get("config_hash"),
                    "prepared_sha256": manifest.get("prepared_sha256"),
                    "prompt_bundle": manifest.get("prompt_bundle"),
                },
            }
            csv_rows.append(
                {
                    "dataset": dataset["name"],
                    "replicate": label,
                    **confusion,
                    "cloud_ledger_rows": len(cloud_calls),
                    "application_cache_hits": replicate_summaries[label][
                        "application_cache_hits"
                    ],
                }
            )

        labels = sorted(replicate_predictions)
        pairwise_agreement: dict[str, Any] = {}
        for index, first in enumerate(labels):
            for second in labels[index + 1 :]:
                pairwise_agreement[f"{first}_vs_{second}"] = _categorical_agreement(
                    [
                        replicate_predictions[first][response_id]
                        for response_id in sorted(expected_ids)
                    ],
                    [
                        replicate_predictions[second][response_id]
                        for response_id in sorted(expected_ids)
                    ],
                )
        stable = sum(
            len(
                {
                    replicate_predictions[label][response_id]
                    for label in labels
                }
            )
            == 1
            for response_id in expected_ids
        )
        metric_summary: dict[str, Any] = {}
        for metric in ("accuracy", "precision", "recall", "f1"):
            values = [
                float(replicate_summaries[label]["confusion"][metric])
                for label in labels
            ]
            metric_summary[metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        report_datasets[str(dataset["name"])] = {
            "selected_responses": len(expected_ids),
            "replicates": replicate_summaries,
            "pairwise_prediction_agreement": pairwise_agreement,
            "stable_predictions_all_replicates": stable,
            "stable_prediction_rate": stable / len(expected_ids),
            "metric_across_replicates": metric_summary,
        }

    report = {
        "analysis_type": "hosted_api_repeatability",
        "package_manifest": Path(str(config["package_manifest"])).as_posix(),
        "fresh_cache_definition": (
            "Each replicate uses its own initially absent application cache. Within-replicate "
            "cache hits may occur only for duplicate exact requests and are reported."
        ),
        "datasets": report_datasets,
    }
    output = _resolve(root, str(config["output_path"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite hosted repeat report: {output}")
    atomic_write_json(output, report)
    _write_csv(_resolve(root, str(config["csv_path"])), csv_rows)
    return output
