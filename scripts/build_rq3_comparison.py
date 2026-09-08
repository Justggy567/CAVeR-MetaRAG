#!/usr/bin/env python3
"""Build the Main-700 versus Replication-140 comparison from canonical analyses."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/analysis/rq3_main_vs_replication.json"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    main_report = read("artifacts/analysis/rq3_main700.json")
    replication_report = read("artifacts/analysis/rq3_replication140.json")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    categories = sorted(
        set(main_report["by_category"]) | set(replication_report["by_category"])
    )
    payload = {
        "analysis_schema_version": "2026-08-22.metrics-v2",
        "main_report": "artifacts/analysis/rq3_main700.json",
        "replication_report": "artifacts/analysis/rq3_replication140.json",
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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
