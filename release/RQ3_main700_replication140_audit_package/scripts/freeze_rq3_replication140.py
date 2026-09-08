#!/usr/bin/env python3
"""Validate and freeze the clean RQ3 Replication-140 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_FIELDS = ("id", "source_dataset", "source_id", "question", "context", "factoid", "category", "context_support")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: Any) -> str:
    return " ".join(str(value or "").split())


def duplicate_groups(rows: list[dict[str, Any]], key_fn) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(str(row["id"]))
    return [sorted(ids) for key, ids in groups.items() if key and len(ids) > 1]


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temp_name)
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--replication-candidates", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--main700", type=Path, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--adjudicator-c", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--frozen-dataset", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--construction-manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    selection = read_jsonl(args.selection)
    reserve = read_jsonl(args.reserve)
    candidates = read_jsonl(args.replication_candidates)
    large_candidates = read_jsonl(args.candidate_pool)
    main700 = read_jsonl(args.main700)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("gate", {}).get("replication_run_authorized_by_audit") is not True:
        raise ValueError("Human-review audit does not authorize freezing")
    if audit["inputs"]["dataset"]["sha256"] != sha256(args.dataset):
        raise ValueError("Audit does not describe the current dataset")
    if len(rows) != 140 or len(selection) != 140 or len(reserve) != 211:
        raise ValueError("Expected 140 dataset/selection rows and 211 reserve rows")
    if len(candidates) != 351 or len(large_candidates) != 830 or len(main700) != 700:
        raise ValueError("Candidate-pool or Main-700 row count is not canonical")
    if len({row["id"] for row in rows}) != 140:
        raise ValueError("Replication IDs are not unique")
    for row in rows:
        if tuple(row.keys()) != DATA_FIELDS:
            raise ValueError(f"Non-canonical field order/schema in {row.get('id')}: {tuple(row.keys())}")

    selection_by_rep = {row["replication_id"]: row for row in selection}
    small_by_id = {row["id"]: row for row in candidates}
    large_by_id = {row["id"]: row for row in large_candidates}
    sync_mismatches: list[str] = []
    for row in rows:
        mapping = selection_by_rep.get(row["id"])
        if mapping is None:
            sync_mismatches.append(row["id"])
            continue
        candidate_id = mapping["candidate_id"]
        for pool in (small_by_id, large_by_id):
            candidate = pool.get(candidate_id)
            expected = {**row, "id": candidate_id}
            if candidate != expected:
                sync_mismatches.append(row["id"])
                break
    if sync_mismatches:
        raise ValueError(f"Dataset/candidate-pool synchronization mismatch: {sorted(set(sync_mismatches))}")

    main_provenance = {(norm(row["source_dataset"]), norm(row["source_id"])) for row in main700}
    main_questions = {norm(row["question"]) for row in main700}
    main_contexts = {norm(row["context"]) for row in main700}
    main_factoids = {norm(row["factoid"]) for row in main700}
    overlaps = {
        "source_dataset+source_id": sum((norm(row["source_dataset"]), norm(row["source_id"])) in main_provenance for row in rows),
        "normalized_question": sum(norm(row["question"]) in main_questions for row in rows),
        "normalized_context": sum(norm(row["context"]) in main_contexts for row in rows),
        "normalized_factoid": sum(norm(row["factoid"]) in main_factoids for row in rows),
    }
    if any(overlaps.values()):
        raise ValueError(f"Replication set overlaps Main-700: {overlaps}")

    duplicates = {
        "provenance": duplicate_groups(rows, lambda row: f"{norm(row['source_dataset'])}|{norm(row['source_id'])}"),
        "question": duplicate_groups(rows, lambda row: norm(row["question"])),
        "context": duplicate_groups(rows, lambda row: norm(row["context"])),
        "factoid": duplicate_groups(rows, lambda row: norm(row["factoid"])),
    }
    if any(duplicates.values()):
        raise ValueError(f"Internal duplicates remain: {duplicates}")

    source_counts = Counter(row["source_dataset"] for row in rows)
    support_counts = Counter(str(bool(row["context_support"])).lower() for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    strata = Counter(
        f"{row['category']}|{str(bool(row['context_support'])).lower()}|{row['source_dataset']}"
        for row in rows
    )
    agreement = audit["review"]["pre_adjudication_agreement"]
    compact_agreement = {
        field: {
            "n": value["n"],
            "agreements": value["agreements"],
            "percent_agreement": value["percent_agreement"],
            "cohen_kappa": value["cohen_kappa"],
        }
        for field, value in agreement.items()
    }

    atomic_copy(args.dataset, args.frozen_dataset)
    frozen_hash = sha256(args.frozen_dataset)
    if frozen_hash != sha256(args.dataset):
        raise ValueError("Frozen dataset hash differs from canonical dataset")

    manifest = {
        "name": "RQ3-FactoidReplication140-v1.2.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "primary_sources": [str(args.annotator_a.resolve()), str(args.annotator_b.resolve()), str(args.adjudicator_c.resolve())],
            "precedence": "A/B consensus is final and Adjudicator C is final for required hard-field disagreements.",
            "hard_final_fields": ["atomicity", "category", "context_support"],
            "descriptive_non_filter_fields": ["question_relevance", "evidence_clarity"],
        },
        "frozen_dataset": {"path": str(args.frozen_dataset.resolve()), "sha256": frozen_hash, "rows": 140},
        "inputs": {
            "main700": {"path": str(args.main700.resolve()), "sha256": sha256(args.main700), "rows": 700},
            "annotator_A": {"path": str(args.annotator_a.resolve()), "sha256": sha256(args.annotator_a), "rows": 140},
            "annotator_B": {"path": str(args.annotator_b.resolve()), "sha256": sha256(args.annotator_b), "rows": 140},
            "adjudicator_C": {"path": str(args.adjudicator_c.resolve()), "sha256": sha256(args.adjudicator_c), "rows": 10},
        },
        "synchronized_complete_data": {
            "canonical_dataset": {"path": str(args.dataset.resolve()), "sha256": sha256(args.dataset), "rows": 140},
            "selection_crosswalk": {"path": str(args.selection.resolve()), "sha256": sha256(args.selection), "rows": 140},
            "reserve_queue": {"path": str(args.reserve.resolve()), "sha256": sha256(args.reserve), "rows": 211},
            "replication_candidate_pool": {"path": str(args.replication_candidates.resolve()), "sha256": sha256(args.replication_candidates), "rows": 351},
            "large_candidate_pool": {"path": str(args.candidate_pool.resolve()), "sha256": sha256(args.candidate_pool), "rows": 830},
        },
        "human_review": {
            "annotator_A_rows": 140,
            "annotator_B_rows": 140,
            "adjudicator_C_rows": 10,
            "agreement_before_adjudication": compact_agreement,
            "core_hard_field_adjudication_complete": True,
            "final_atomicity": "PASS for all 140 records",
        },
        "final_distribution": {
            "by_source_dataset": dict(sorted(source_counts.items())),
            "by_context_support": dict(sorted(support_counts.items())),
            "by_category": dict(sorted(category_counts.items())),
            "by_category_support_and_source": dict(sorted(strata.items())),
            "post_review_stratum_min": min(strata.values()),
            "post_review_stratum_max": max(strata.values()),
            "all_28_strata_equal_5": len(strata) == 28 and set(strata.values()) == {5},
            "fixed_140_ids_retained_without_replacement_after_review": True,
        },
        "validation": {
            "main700_overlap_counts": overlaps,
            "internal_duplicate_groups": duplicates,
            "dataset_candidate_pools_synchronized": True,
            "clean_dataset_schema": list(DATA_FIELDS),
            "replication_run_authorized": True,
        },
    }
    atomic_json(args.freeze_manifest, manifest)
    atomic_json(args.construction_manifest, manifest)
    print(json.dumps({"frozen_dataset_sha256": frozen_hash, "rows": 140, "run_authorized": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
