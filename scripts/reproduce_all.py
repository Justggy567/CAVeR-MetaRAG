#!/usr/bin/env python3
"""Rebuild all RQ1/RQ2/RQ3 analyses and publication tables offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.caver.policy_analysis import run_policy_suite  # noqa: E402
from src.caver.publication import export_publication_tables  # noqa: E402
from src.caver.rq1_analysis import run_rq1_analysis  # noqa: E402
from src.caver.rq3_analysis import run_rq3_analysis  # noqa: E402


def load_config(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_rq3_comparison(main_path: Path, replication_path: Path, output: Path) -> None:
    main_report = json.loads(main_path.read_text(encoding="utf-8"))
    replication_report = json.loads(replication_path.read_text(encoding="utf-8"))
    categories = sorted(
        set(main_report["by_category"]) | set(replication_report["by_category"])
    )
    payload = {
        "analysis_schema_version": "2026-08-22.metrics-v2",
        "main_report": relative(main_path),
        "replication_report": relative(replication_path),
        "main_overall": main_report["overall"],
        "replication_overall": replication_report["overall"],
        "main_overall_cluster_bootstrap_95ci": main_report[
            "overall_cluster_bootstrap_95ci"
        ],
        "replication_overall_cluster_bootstrap_95ci": replication_report[
            "overall_cluster_bootstrap_95ci"
        ],
        "category_reports": {
            category: {
                "main": main_report["by_category"].get(category),
                "replication": replication_report["by_category"].get(category),
            }
            for category in categories
        },
        "interpretation_scope": (
            "Replication-140 is independent of Main-700 by frozen overlap checks. "
            "Dataset-level estimates must be reported separately; the smaller replication "
            "set is corroborative rather than a replacement for Main-700."
        ),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline one-command reproduction from the frozen ledgers. No model or API "
            "requests are made."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "New directory under this repository. Default: "
            "artifacts/reproduced/run-YYYYMMDD-HHMMSS"
        ),
    )
    args = parser.parse_args()

    requested = args.output_root or Path(
        "artifacts/reproduced/" + datetime.now().strftime("run-%Y%m%d-%H%M%S")
    )
    output_root = requested if requested.is_absolute() else ROOT / requested
    try:
        relative(output_root)
    except ValueError as exc:
        raise ValueError("--output-root must be inside the repository") from exc
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite reproduction output: {output_root}")

    analysis = output_root / "analysis"
    analysis.mkdir(parents=True)
    outputs = {
        "rq1": analysis / "rq1_all_models.json",
        "rq2_asqa": analysis / "rq2_asqa_top5_76.json",
        "rq2_haluevalqa": analysis / "rq2_haluevalqa_3206.json",
        "rq3_main": analysis / "rq3_main700.json",
        "rq3_replication": analysis / "rq3_replication140.json",
        "rq3_comparison": analysis / "rq3_main_vs_replication.json",
    }

    rq1 = load_config("configs/analysis/rq1_all_models.json")
    rq1["output_path"] = relative(outputs["rq1"])
    run_rq1_analysis(rq1, repository_root=ROOT)

    for key, config_path in (
        ("rq2_asqa", "configs/analysis/rq2_asqa_top5_76.json"),
        ("rq2_haluevalqa", "configs/analysis/rq2_haluevalqa_3206.json"),
    ):
        config = load_config(config_path)
        config["output_path"] = relative(outputs[key])
        run_policy_suite(config, repository_root=ROOT)

    for key, config_path in (
        ("rq3_main", "configs/analysis/rq3_main700.json"),
        ("rq3_replication", "configs/analysis/rq3_replication140.json"),
    ):
        config = load_config(config_path)
        config["output_path"] = relative(outputs[key])
        run_rq3_analysis(config, repository_root=ROOT)

    write_rq3_comparison(
        outputs["rq3_main"], outputs["rq3_replication"], outputs["rq3_comparison"]
    )

    publication = {
        "output_directory": relative(output_root / "publication_tables"),
        "rq1_report": relative(outputs["rq1"]),
        "rq2_reports": [
            {"dataset": "ASQA-top5-76", "path": relative(outputs["rq2_asqa"])},
            {
                "dataset": "HaluEvalQA-3206-afterHuman",
                "path": relative(outputs["rq2_haluevalqa"]),
            },
        ],
        "rq3_report": relative(outputs["rq3_main"]),
    }
    publication_dir = export_publication_tables(publication, repository_root=ROOT)

    generated = sorted(
        path for path in output_root.rglob("*") if path.is_file()
    )
    manifest = {
        "status": "complete",
        "mode": "offline_from_frozen_ledgers",
        "model_or_api_calls": False,
        "outputs": [
            {"path": relative(path), "sha256": sha256(path)} for path in generated
        ],
    }
    manifest_path = output_root / "reproduction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": relative(output_root),
                "publication_tables": relative(publication_dir),
                "rq2_rows": 28,
                "model_or_api_calls": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
