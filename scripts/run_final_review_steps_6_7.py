#!/usr/bin/env python3
"""

The script is intentionally analysis-only.  It reads the completed runtime tree,
writes a new versioned review bundle, and refuses to overwrite an existing one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEWER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEWER_ROOT / "src"))

from caver.audit import (  # noqa: E402
    audit_data_config,
    environment_report,
    parameter_report,
)
from caver.datasets import file_sha256  # noqa: E402
from caver.hosted_repeat import analyze_hosted_repeats  # noqa: E402
from caver.ledger import atomic_write_json  # noqa: E402
from caver.policy_analysis import run_policy_suite  # noqa: E402
from caver.reviewer_analysis import (  # noqa: E402
    run_cloud_cost_audit,
    run_human_audit_analysis,
    run_preparation_quality_audit,
    run_shared_model_bias_analysis,
)
from caver.rq1_analysis import run_rq1_analysis  # noqa: E402
from caver.rq3_analysis import run_rq3_analysis  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    atomic_write_json(path, value)


def absolute(runtime: Path, value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else runtime / path)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def completed_manifest_rows(runtime: Path) -> list[dict[str, Any]]:
    roots = [
        runtime / "artifacts/runs/rq1",
        runtime / "artifacts/runs/rq2",
        runtime / "artifacts/runs/rq3",
        runtime / "artifacts/reviewer/hosted_repeats_20260829/runs",
    ]
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("manifest.json")):
            manifest = read_json(path)
            row = {
                "path": str(path.resolve()),
                "status": manifest.get("status"),
                "completed_responses": manifest.get("completed_responses"),
                "expected_responses": manifest.get("expected_responses"),
                "sha256": file_sha256(path),
            }
            if row["status"] != "complete":
                raise ValueError(f"Run is not complete: {path}")
            if row["completed_responses"] != row["expected_responses"]:
                raise ValueError(f"Run count mismatch: {path}")
            rows.append(row)
    if len(rows) != 18:
        raise ValueError(f"Expected 18 completed run manifests, found {len(rows)}")
    return rows


def load_analysis_config(runtime: Path, name: str, output: Path) -> dict[str, Any]:
    config = read_json(runtime / "configs/analysis" / name)
    for key in (
        "matrix_run_directory",
        "preparation_calls_path",
        "strong_only_run_directory",
        "strong_only_preparation_calls_path",
        "run_directory",
    ):
        if key in config:
            config[key] = absolute(runtime, config[key])
    if "runs" in config:
        for item in config["runs"]:
            item["run_directory"] = absolute(runtime, item["run_directory"])
            item["preparation_calls_path"] = absolute(
                runtime, item["preparation_calls_path"]
            )
    config["output_path"] = str(output)
    return config


def analysis_step(runtime: Path, output_root: Path, configs_root: Path) -> list[Path]:
    analysis_root = output_root / "analysis"
    analyses: list[Path] = []

    rq1_output = analysis_root / "rq1_all_models.json"
    rq1_config = load_analysis_config(
        runtime, "rq1_all_models.json", rq1_output
    )
    write_json(configs_root / "analysis_rq1_all_models.json", rq1_config)
    analyses.append(run_rq1_analysis(rq1_config, repository_root=REVIEWER_ROOT))

    rq2_outputs: list[Path] = []
    for stem in ("rq2_asqa_top5_76", "rq2_haluevalqa_3206"):
        output = analysis_root / f"{stem}.json"
        config = load_analysis_config(runtime, f"{stem}.json", output)
        write_json(configs_root / f"analysis_{stem}.json", config)
        rq2_outputs.append(run_policy_suite(config, repository_root=REVIEWER_ROOT))
    analyses.extend(rq2_outputs)

    rq3_outputs: dict[str, Path] = {}
    for stem in ("rq3_main700", "rq3_replication140"):
        output = analysis_root / f"{stem}.json"
        config = load_analysis_config(runtime, f"{stem}.json", output)
        write_json(configs_root / f"analysis_{stem}.json", config)
        rq3_outputs[stem] = run_rq3_analysis(config, repository_root=REVIEWER_ROOT)
        analyses.append(rq3_outputs[stem])

    hosted_config = {
        "package_manifest": str(
            runtime
            / "artifacts/reviewer/hosted_repeats_20260829/package_manifest.json"
        ),
        "runs_root": str(
            runtime / "artifacts/reviewer/hosted_repeats_20260829/runs"
        ),
        "output_path": str(analysis_root / "hosted_repeatability.json"),
        "csv_path": str(analysis_root / "hosted_repeatability.csv"),
    }
    write_json(configs_root / "analysis_hosted_repeatability.json", hosted_config)
    analyses.append(
        analyze_hosted_repeats(hosted_config, repository_root=REVIEWER_ROOT)
    )
    analyses.append(Path(hosted_config["csv_path"]))

    shared_config = read_json(
        REVIEWER_ROOT / "configs/reviewer/shared_model_bias_20260829.json"
    )
    shared_config["output_path"] = str(analysis_root / "shared_model_bias.json")
    shared_config["csv_path"] = str(analysis_root / "shared_model_bias.csv")
    write_json(configs_root / "analysis_shared_model_bias.json", shared_config)
    analyses.append(
        run_shared_model_bias_analysis(
            shared_config, repository_root=REVIEWER_ROOT
        )
    )
    analyses.append(Path(shared_config["csv_path"]))

    main = read_json(rq3_outputs["rq3_main700"])
    replication = read_json(rq3_outputs["rq3_replication140"])
    comparison = {
        "analysis_type": "rq3_main700_vs_independent_replication140",
        "main_report": str(rq3_outputs["rq3_main700"].resolve()),
        "replication_report": str(rq3_outputs["rq3_replication140"].resolve()),
        "main_overall": main["overall"],
        "replication_overall": replication["overall"],
        "main_overall_cluster_bootstrap_95ci": main[
            "overall_cluster_bootstrap_95ci"
        ],
        "replication_overall_cluster_bootstrap_95ci": replication[
            "overall_cluster_bootstrap_95ci"
        ],
        "category_reports": {
            category: {
                "main": main["by_category"].get(category),
                "replication": replication["by_category"].get(category),
            }
            for category in sorted(
                set(main["by_category"]) | set(replication["by_category"])
            )
        },
        "interpretation_scope": (
            "Replication140 is independent of Main700 by frozen overlap checks. "
            "Dataset-level estimates must be reported separately; the smaller replication "
            "set is corroborative rather than a replacement for Main700."
        ),
    }
    comparison_path = analysis_root / "rq3_main_vs_replication.json"
    write_json(comparison_path, comparison)
    analyses.append(comparison_path)
    return analyses


def data_audit_step(runtime: Path, audit_root: Path, configs_root: Path) -> Path:
    main_counts = {
        f"Cat{index}": {"n": 100, "supported": 50, "unsupported": 50}
        for index in range(1, 8)
    }
    replication_counts = {
        f"Cat{index}": {"n": 20, "supported": 10, "unsupported": 10}
        for index in range(1, 8)
    }
    config = {
        "datasets": [
            {
                "name": "HaluEvalQA-3206-afterHuman",
                "path": "data/frozen/haluevalqa_3206_afterHuman.json",
                "format": "paired_answers",
                "expected_count": 3206,
                "expected_sha256": "52b474d61f76f72b267a8a88985cdb638893a401fd0b7c62fdb5db697aa4d699",
            },
            {
                "name": "ASQA-top5-76",
                "path": "data/frozen/asqa_top5_76_cleaned_final.json",
                "format": "paired_answers",
                "expected_count": 76,
                "expected_sha256": "b4189a7be2798cee0f6f1cf43a07597682ac7eb199ae29f9d6f105a670b69f78",
            },
            {
                "name": "RQ3-main700-final",
                "path": "data/frozen/rq3_main700_final.jsonl",
                "format": "factoids",
                "expected_count": 700,
                "expected_sha256": "c3f868a3af409dbf61838f4edce8820c01a8a62f7c7fe70c3457a94dd5db331a",
                "expected_category_counts": main_counts,
            },
            {
                "name": "RQ3-replication140-final",
                "path": "data/frozen/rq3_replication140_final.jsonl",
                "format": "factoids",
                "expected_count": 140,
                "expected_sha256": "337595d8d4f4dd8ea6240c4b3aded1d851afe36f84bfdc9c9f5c517aba4f302e",
                "expected_category_counts": replication_counts,
            },
        ]
    }
    write_json(configs_root / "audit_data.json", config)
    report = audit_data_config(config, repository_root=runtime)
    report["freeze_manifests"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for name, path in {
            "asqa": runtime / "data/frozen/asqa_top5_76_cleaned_final.freeze.json",
            "haluevalqa": runtime
            / "data/frozen/haluevalqa_3206_afterHuman.freeze.json",
            "rq3_main700": runtime / "data/frozen/rq3_main700_final.freeze.json",
            "rq3_replication140": runtime
            / "release/RQ3_main700_replication140_audit_package/data/replication140/rq3_replication140_final.freeze.json",
        }.items()
    }
    output = audit_root / "data_audit.json"
    write_json(output, report)
    return output


def parameter_audit_step(runtime: Path, audit_root: Path, configs_root: Path) -> Path:
    report = parameter_report(runtime)
    hosted_root = runtime / "artifacts/reviewer/hosted_repeats_20260829"
    hosted_rows: list[dict[str, Any]] = []
    for path in sorted((hosted_root / "configs").glob("*.json")):
        config = read_json(path)
        stage1 = config["stage1"]
        hosted_rows.append(
            {
                "config": str(path.resolve()),
                "config_sha256": file_sha256(path),
                "run_id": config["run_id"],
                "prepared_path": absolute(runtime, config["prepared_path"]),
                "fresh_cache_path": absolute(runtime, stage1["cache_path"]),
                "backend": stage1["backend"],
                "model_id": stage1["model_id"],
                "base_url": stage1["base_url"],
                "api_key_env": stage1["api_key_env"],
                "api_key_value_recorded": False,
                "temperature": stage1["temperature"],
                "top_p": stage1["top_p"],
                "max_tokens": stage1["max_tokens"],
                "seed": stage1["seed"],
                "thinking_mode": stage1["thinking_mode"],
                "timeout_seconds": stage1["timeout_seconds"],
                "max_retries": stage1["max_retries"],
                "concurrency": config["environment"]["concurrency"],
                "execution": config["environment"]["execution"],
            }
        )
    if len(hosted_rows) != 6:
        raise ValueError(f"Expected six hosted repeat configs, found {len(hosted_rows)}")
    report["hosted_repeat_design"] = {
        "package_manifest": str((hosted_root / "package_manifest.json").resolve()),
        "package_manifest_sha256": file_sha256(hosted_root / "package_manifest.json"),
        "config_count": len(hosted_rows),
        "configs": hosted_rows,
    }
    report["final_analysis_configs"] = {
        path.name: file_sha256(path) for path in sorted(configs_root.glob("analysis_*.json"))
    }
    report["analysis_runner"] = {
        "root": str(REVIEWER_ROOT),
        "source_hashes": {
            name: file_sha256(REVIEWER_ROOT / "src/caver" / name)
            for name in (
                "rq1_analysis.py",
                "policy_analysis.py",
                "rq3_analysis.py",
                "hosted_repeat.py",
                "reviewer_analysis.py",
                "audit.py",
            )
        },
    }
    output = audit_root / "parameters_audit.json"
    write_json(output, report)
    return output


def environment_audit_step(runtime: Path, audit_root: Path) -> Path:
    report = environment_report()
    report["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["runtime_root"] = str(runtime.resolve())
    report["analysis_code_root"] = str(REVIEWER_ROOT.resolve())
    output = audit_root / "environment_audit.json"
    write_json(output, report)
    return output


def preparation_audit_step(
    runtime: Path, audit_root: Path, configs_root: Path
) -> list[Path]:
    def manifest(path: str) -> str:
        return str((runtime / path).resolve())

    config = {
        "output_path": str(audit_root / "preparation_audit.json"),
        "csv_path": str(audit_root / "preparation_audit.csv"),
        "datasets": [
            {
                "name": "ASQA-top5-76",
                "prepared_manifest": manifest(
                    "artifacts/prepared/asqa_top5_76.probes.jsonl.manifest.json"
                ),
                "run_manifests": [
                    manifest(f"artifacts/runs/rq1/asqa_top5_76/{model}/primary/manifest.json")
                    for model in (
                        "gemma3_4b",
                        "gemma3_12b",
                        "phi4_14b",
                        "deepseek_v4_flash",
                    )
                ]
                + [manifest("artifacts/runs/rq2/asqa_top5_76/matrix/primary/manifest.json")],
            },
            {
                "name": "HaluEvalQA-3206-afterHuman",
                "prepared_manifest": manifest(
                    "artifacts/prepared/haluevalqa_3206.probes.jsonl.manifest.json"
                ),
                "run_manifests": [
                    manifest(f"artifacts/runs/rq1/haluevalqa_3206/{model}/primary/manifest.json")
                    for model in (
                        "gemma3_4b",
                        "gemma3_12b",
                        "phi4_14b",
                        "deepseek_v4_flash",
                    )
                ]
                + [manifest("artifacts/runs/rq2/haluevalqa_3206/matrix/primary/manifest.json")],
            },
            {
                "name": "RQ3-main700-final",
                "prepared_manifest": manifest(
                    "artifacts/prepared/rq3_main700.probes.jsonl.manifest.json"
                ),
                "run_manifests": [
                    manifest(
                        "artifacts/runs/rq3/main700/structure_aware/primary/manifest.json"
                    )
                ],
            },
            {
                "name": "RQ3-replication140-final",
                "prepared_manifest": manifest(
                    "artifacts/prepared/rq3_replication140.probes.jsonl.manifest.json"
                ),
                "run_manifests": [
                    manifest(
                        "artifacts/runs/rq3/replication140/structure_aware/primary/manifest.json"
                    )
                ],
            },
        ],
    }
    write_json(configs_root / "audit_preparation.json", config)
    output = run_preparation_quality_audit(config, repository_root=REVIEWER_ROOT)
    return [output, Path(config["csv_path"])]


def cost_audit_step(runtime: Path, audit_root: Path, configs_root: Path) -> list[Path]:
    pricing_source = audit_root / "pricing_source_deepseek_peak_20260829.png"
    source_image = (
        REVIEWER_ROOT / "artifacts/audits/pricing_source_deepseek_peak_20260829.png"
    )
    if pricing_source.exists():
        raise FileExistsError(f"Refusing to overwrite: {pricing_source}")
    shutil.copy2(source_image, pricing_source)
    ledgers = [
        ("Preparation-ASQA", "artifacts/prepared/asqa_top5_76.probes.jsonl.calls.jsonl"),
        ("Preparation-HaluEvalQA", "artifacts/prepared/haluevalqa_3206.probes.jsonl.calls.jsonl"),
        ("Preparation-RQ3-Main700", "artifacts/prepared/rq3_main700.probes.jsonl.calls.jsonl"),
        ("Preparation-RQ3-Replication140", "artifacts/prepared/rq3_replication140.probes.jsonl.calls.jsonl"),
        ("RQ1-ASQA-DeepSeek", "artifacts/runs/rq1/asqa_top5_76/deepseek_v4_flash/primary/calls.jsonl"),
        ("RQ1-HaluEvalQA-DeepSeek", "artifacts/runs/rq1/haluevalqa_3206/deepseek_v4_flash/primary/calls.jsonl"),
        ("RQ2-ASQA-matrix", "artifacts/runs/rq2/asqa_top5_76/matrix/primary/calls.jsonl"),
        ("RQ2-HaluEvalQA-matrix", "artifacts/runs/rq2/haluevalqa_3206/matrix/primary/calls.jsonl"),
        ("RQ3-Main700", "artifacts/runs/rq3/main700/structure_aware/primary/calls.jsonl"),
        ("RQ3-Replication140", "artifacts/runs/rq3/replication140/structure_aware/primary/calls.jsonl"),
    ]
    for dataset in ("asqa_top5_76", "haluevalqa_3206"):
        for repeat in range(1, 4):
            label = f"replicate_{repeat:02d}"
            ledgers.append(
                (
                    f"Hosted-{dataset}-{label}",
                    f"artifacts/reviewer/hosted_repeats_20260829/runs/{dataset}/{label}/calls.jsonl",
                )
            )
    config = {
        "output_path": str(audit_root / "cost_audit.json"),
        "csv_path": str(audit_root / "cost_audit.csv"),
        "pricing": {
            "currency": "CNY",
            "price_snapshot_date": "2026-08-29",
            "time_band": "peak",
            "source": str(pricing_source.resolve()),
            "models": {
                "deepseek-v4-flash": {
                    "cached_input_per_million": 0.1,
                    "uncached_input_per_million": 3,
                    "output_per_million": 9,
                }
            },
        },
        "call_ledgers": [
            {"scope": scope, "path": str((runtime / path).resolve())}
            for scope, path in ledgers
        ],
    }
    require_files([Path(item["path"]) for item in config["call_ledgers"]])
    write_json(configs_root / "audit_cost.json", config)
    output = run_cloud_cost_audit(config, repository_root=REVIEWER_ROOT)
    return [output, Path(config["csv_path"]), pricing_source]


def human_audit_step(
    runtime: Path, audit_root: Path, configs_root: Path
) -> list[Path]:
    main_annotations = runtime / "data/rq3/annotations/main700"
    main_config = {
        "annotator_a": str((main_annotations / "annotator_A.csv").resolve()),
        "annotator_b": str((main_annotations / "annotator_B.csv").resolve()),
        "adjudication": str((main_annotations / "adjudication.csv").resolve()),
        "frozen_dataset": str(
            (runtime / "data/frozen/rq3_main700_final.jsonl").resolve()
        ),
        "output_path": str(audit_root / "rq3_main700_human_audit.json"),
        "csv_path": str(audit_root / "rq3_main700_human_agreement.csv"),
    }
    require_files(
        [
            Path(main_config["annotator_a"]),
            Path(main_config["annotator_b"]),
            Path(main_config["adjudication"]),
            Path(main_config["frozen_dataset"]),
        ]
    )
    write_json(configs_root / "audit_rq3_main700_human.json", main_config)
    main_output = run_human_audit_analysis(
        main_config, repository_root=REVIEWER_ROOT
    )

    replication_output = replication_human_audit_step(runtime, audit_root)
    return [main_output, Path(main_config["csv_path"]), replication_output]


def replication_human_audit_step(runtime: Path, audit_root: Path) -> Path:
    replication_package = runtime / "release/RQ3_main700_replication140_audit_package"
    replication_output = audit_root / "rq3_replication140_human_audit.json"
    # The final-only release owns the current A/B/C exports and their audit.
    # Verify that package without rewriting its frozen reports or hash files,
    # then reuse its authoritative human audit in the new review bundle.
    replication_script = replication_package / "scripts/verify_package.py"
    replication_audit = (
        replication_package / "audit/replication140/rq3_replication140_human_audit.json"
    )
    runtime_dataset = runtime / "data/frozen/rq3_replication140_final.jsonl"
    package_dataset = replication_package / "data/replication140/rq3_replication140_final.jsonl"
    require_files([replication_script, replication_audit, runtime_dataset, package_dataset])
    if file_sha256(runtime_dataset) != file_sha256(package_dataset):
        raise ValueError("Replication-140 runtime dataset differs from the authoritative review package")
    command = [
        sys.executable,
        "-B",
        str(replication_script),
        "--no-write",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Replication human audit failed:\n"
            + result.stdout
            + "\n"
            + result.stderr
        )
    if replication_output.exists():
        raise FileExistsError(f"Refusing to overwrite: {replication_output}")
    shutil.copyfile(replication_audit, replication_output)
    return replication_output


def file_manifest(output_root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(output_root)).replace("\\", "/"): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite final review bundle: {output_root}")
    output_root.mkdir(parents=True)
    configs_root = output_root / "configs"
    audit_root = output_root / "audits"
    configs_root.mkdir()
    audit_root.mkdir()

    run_manifests = completed_manifest_rows(runtime)
    analyses = analysis_step(runtime, output_root, configs_root)
    audits = [
        data_audit_step(runtime, audit_root, configs_root),
        parameter_audit_step(runtime, audit_root, configs_root),
        environment_audit_step(runtime, audit_root),
    ]
    audits.extend(preparation_audit_step(runtime, audit_root, configs_root))
    audits.extend(cost_audit_step(runtime, audit_root, configs_root))
    audits.extend(human_audit_step(runtime, audit_root, configs_root))

    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": {
            "step_6": "Unified offline analysis rerun from frozen completed ledgers",
            "step_7": "Final data/parameters/environment/preparation/cost audits",
            "model_inference_rerun": False,
            "network_model_calls": False,
        },
        "runtime_root": str(runtime),
        "analysis_code_root": str(REVIEWER_ROOT),
        "completed_run_manifests": run_manifests,
        "analysis_outputs": [str(path.resolve()) for path in analyses],
        "audit_outputs": [str(path.resolve()) for path in audits],
        "files": file_manifest(output_root),
    }
    write_json(output_root / "bundle_manifest.json", manifest)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
