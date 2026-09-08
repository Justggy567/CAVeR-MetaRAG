#!/usr/bin/env python3
"""Verify the self-contained RQ3 Main-700 and Replication-140 audit package.

Uses only the Python standard library. It recomputes final-dataset balance,
inter-annotator agreement, adjudication coverage, Main/replication disjointness,
legacy reconstruction counts, and the deterministic replication selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = tuple(f"Cat{i}" for i in range(1, 8))
SOURCES = ("ASQA", "HALUEVALQA")
SUPPORTS = (True, False)
SAMPLE_SALT = "rq3-replication140-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path, *, errors: str = "strict") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors=errors, newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def normalize_label(value: Any) -> str:
    return str(value).strip().upper()


def normalize_bool_label(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    label = normalize_label(value)
    if label in {"TRUE", "SUPPORTED", "ENTAILED"}:
        return "TRUE"
    if label in {"FALSE", "UNSUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}:
        return "FALSE"
    return label


def legacy_binary_support(value: Any) -> str:
    return "TRUE" if normalize_label(value) == "ENTAILED" else "FALSE"


def agreement(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Agreement inputs must have equal non-zero length")
    n = len(labels_a)
    agreements = sum(a == b for a, b in zip(labels_a, labels_b))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = sorted(set(labels_a) | set(labels_b))
    observed = agreements / n
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in labels)
    kappa = None if abs(1.0 - expected) < 1e-15 else (observed - expected) / (1.0 - expected)
    return {
        "n": n,
        "agreements": agreements,
        "disagreements": n - agreements,
        "percent_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "annotator_A_distribution": dict(sorted(counts_a.items())),
        "annotator_B_distribution": dict(sorted(counts_b.items())),
    }


def by_id(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field]).strip()
        if not key or key in result:
            raise ValueError(f"Missing or duplicate identifier in field {field}: {key!r}")
        result[key] = row
    return result


def provenance_key(row: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(row["source_dataset"]), normalize_text(row["source_id"])


def sample_priority(candidate_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SALT}|{candidate_id}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "reports" / "verification_results.json",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        item: dict[str, Any] = {"name": name, "status": "PASS" if condition else "FAIL"}
        if details is not None:
            item["details"] = details
        checks.append(item)

    main_rows = read_jsonl(PACKAGE_ROOT / "data/main700/rq3_main700_final.jsonl")
    replication_rows = read_jsonl(PACKAGE_ROOT / "data/replication140/rq3_replication140_final.jsonl")
    large_candidates = read_jsonl(PACKAGE_ROOT / "data/construction/rq3_candidate_pool_final.jsonl")
    replication_candidates = read_jsonl(
        PACKAGE_ROOT / "data/construction/rq3_replication_candidate_pool_351.jsonl"
    )
    selection_rows = read_jsonl(
        PACKAGE_ROOT / "data/construction/rq3_replication140_selection.jsonl"
    )
    reserve_rows = read_jsonl(
        PACKAGE_ROOT / "data/construction/rq3_replication140_reserve_queue.jsonl"
    )

    # Main-700 structure.
    main_ids = [str(row["id"]) for row in main_rows]
    main_category = Counter(str(row["category"]) for row in main_rows)
    main_support = Counter(normalize_bool_label(row["context_support"]) for row in main_rows)
    main_strata = Counter(
        (str(row["category"]), normalize_bool_label(row["context_support"]))
        for row in main_rows
    )
    check("main700_row_count", len(main_rows) == 700, len(main_rows))
    check("main700_unique_ids", len(set(main_ids)) == 700, len(set(main_ids)))
    check("main700_category_balance", all(main_category[c] == 100 for c in CATEGORIES), main_category)
    check("main700_support_balance", main_support == {"FALSE": 350, "TRUE": 350}, main_support)
    check(
        "main700_14_strata_equal_50",
        len(main_strata) == 14 and all(value == 50 for value in main_strata.values()),
        {f"{k[0]}|{k[1]}": v for k, v in sorted(main_strata.items())},
    )

    # Main-700 final annotation agreement and adjudication.
    main_a_rows = read_csv(PACKAGE_ROOT / "audit/main700/annotator_A.csv")
    main_b_rows = read_csv(PACKAGE_ROOT / "audit/main700/annotator_B.csv")
    main_adj_rows = read_csv(PACKAGE_ROOT / "audit/main700/adjudication.csv")
    main_a = by_id(main_a_rows, "id")
    main_b: dict[str, dict[str, Any]] = {}
    main_b_order_errors: list[str] = []
    for row in main_b_rows:
        try:
            position = int(str(row["source_order"]).strip())
        except (KeyError, ValueError):
            main_b_order_errors.append(str(row.get("annotation_id", "")))
            continue
        if position < 1 or position > len(main_ids):
            main_b_order_errors.append(str(row.get("annotation_id", "")))
            continue
        item_id = main_ids[position - 1]
        if item_id in main_b:
            main_b_order_errors.append(str(row.get("annotation_id", "")))
            continue
        main_b[item_id] = row
    main_adj = by_id(main_adj_rows, "id")
    check(
        "main700_annotation_id_coverage",
        set(main_a) == set(main_b) == set(main_ids) and not main_b_order_errors,
        {"main_b_order_errors": main_b_order_errors},
    )
    main_text_mismatches = {"annotator_A": [], "annotator_B": []}
    main_by_identifier = by_id(main_rows, "id")
    for item_id in main_ids:
        canonical = main_by_identifier[item_id]
        for field in ("question", "context", "factoid"):
            if main_a[item_id][field] != canonical[field]:
                main_text_mismatches["annotator_A"].append(f"{item_id}:{field}")
            if main_b[item_id][field] != canonical[field]:
                main_text_mismatches["annotator_B"].append(f"{item_id}:{field}")
    check(
        "main700_A_B_inputs_match_frozen_text",
        not any(main_text_mismatches.values()),
        main_text_mismatches,
    )

    main_fields = {
        "atomicity": ("atomicity", "atomicity", normalize_label),
        "question_relevance": ("question_relevance", "question_relevance", normalize_label),
        "primary_category": ("primary_category", "primary_category", normalize_label),
        "context_support": ("context_support", "context_support", normalize_bool_label),
        "evidence_clarity": ("evidence_clarity", "evidence_clarity", normalize_label),
    }
    main_iaa: dict[str, Any] = {}
    main_disagreement_ids: dict[str, list[str]] = {}
    for output_field, (a_field, b_field, transform) in main_fields.items():
        labels_a = [transform(main_a[item_id][a_field]) for item_id in main_ids]
        labels_b = [transform(main_b[item_id][b_field]) for item_id in main_ids]
        main_iaa[output_field] = agreement(labels_a, labels_b)
        main_disagreement_ids[output_field] = [
            item_id
            for item_id in main_ids
            if transform(main_a[item_id][a_field]) != transform(main_b[item_id][b_field])
        ]
    main_hard_union = set().union(
        main_disagreement_ids["atomicity"],
        main_disagreement_ids["primary_category"],
        main_disagreement_ids["context_support"],
    )
    check("main700_hard_disagreement_count", len(main_hard_union) == 47, len(main_hard_union))
    check("main700_adjudication_rows", len(main_adj_rows) == 53, len(main_adj_rows))
    check(
        "main700_all_hard_disagreements_adjudicated",
        main_hard_union.issubset(main_adj),
        sorted(main_hard_union - set(main_adj)),
    )
    main_final_mismatches: list[str] = []
    for row in main_rows:
        item_id = str(row["id"])
        if item_id in main_adj:
            category = normalize_label(main_adj[item_id]["adjudicator_final_category"])
            support = normalize_bool_label(main_adj[item_id]["adjudicator_context_support"])
            atomicity = normalize_label(main_adj[item_id]["adjudicator__atomicity"])
        else:
            category = normalize_label(main_a[item_id]["primary_category"])
            support = normalize_bool_label(main_a[item_id]["context_support"])
            atomicity = normalize_label(main_a[item_id]["atomicity"])
        if category != normalize_label(row["category"]):
            main_final_mismatches.append(f"{item_id}:category")
        if support != normalize_bool_label(row["context_support"]):
            main_final_mismatches.append(f"{item_id}:support")
        if atomicity != "PASS":
            main_final_mismatches.append(f"{item_id}:atomicity")
    check("main700_adjudicated_labels_match_frozen", not main_final_mismatches, main_final_mismatches)

    # Legacy audit and revision rates.
    legacy_a_rows = read_csv(
        PACKAGE_ROOT / "audit/legacy700/legacy_factoid700_annotator_A.csv", errors="replace"
    )
    legacy_b_rows = read_csv(
        PACKAGE_ROOT / "audit/legacy700/legacy_factoid700_annotator_B.csv", errors="replace"
    )
    legacy_adj_rows = read_csv(PACKAGE_ROOT / "audit/legacy700/legacy_factoid700_adjudication.csv")
    legacy_change_rows = read_csv(
        PACKAGE_ROOT / "audit/legacy700/rq3_legacy_change_log_after_human_end.csv"
    )
    legacy_a = by_id(legacy_a_rows, "id")
    legacy_b = by_id(legacy_b_rows, "id")
    legacy_ids = sorted(legacy_a, key=lambda value: int(value))
    check("legacy_annotation_id_coverage", set(legacy_a) == set(legacy_b) and len(legacy_ids) == 700)
    legacy_fields = {
        "atomicity": ("atomicity", normalize_label),
        "question_relevance": ("question_relevance", normalize_label),
        "evidence_clarity": ("evidence_clarity", normalize_label),
        "evidence_relation": ("evidence_relation", normalize_label),
        "primary_category": ("primary_category", normalize_label),
    }
    legacy_iaa: dict[str, Any] = {}
    legacy_any_disagreement: set[str] = set()
    for output_field, (field, transform) in legacy_fields.items():
        labels_a = [transform(legacy_a[item_id][field]) for item_id in legacy_ids]
        labels_b = [transform(legacy_b[item_id][field]) for item_id in legacy_ids]
        legacy_iaa[output_field] = agreement(labels_a, labels_b)
        legacy_any_disagreement.update(
            item_id
            for item_id in legacy_ids
            if transform(legacy_a[item_id][field]) != transform(legacy_b[item_id][field])
        )
    legacy_iaa["binary_support_entailed_vs_nonentailed"] = agreement(
        [legacy_binary_support(legacy_a[item_id]["evidence_relation"]) for item_id in legacy_ids],
        [legacy_binary_support(legacy_b[item_id]["evidence_relation"]) for item_id in legacy_ids],
    )
    legacy_adj_ids = {str(row["legacy_id"]).strip() for row in legacy_adj_rows}
    check("legacy_disagreement_union_count", len(legacy_any_disagreement) == 222, len(legacy_any_disagreement))
    check("legacy_adjudication_exact_coverage", legacy_adj_ids == legacy_any_disagreement)
    legacy_actions = Counter(normalize_label(row["action"]) for row in legacy_change_rows)
    legacy_changed_dimensions = Counter(str(row["changed_dimensions"]).strip() for row in legacy_change_rows)
    check(
        "legacy_action_counts",
        legacy_actions == {"REMOVE": 368, "RELABEL": 126, "RETAIN_UNCHANGED": 206},
        legacy_actions,
    )

    # Replication-140 structure and independence.
    replication_ids = [str(row["id"]) for row in replication_rows]
    rep_category = Counter(str(row["category"]) for row in replication_rows)
    rep_support = Counter(normalize_bool_label(row["context_support"]) for row in replication_rows)
    rep_source = Counter(str(row["source_dataset"]) for row in replication_rows)
    rep_strata = Counter(
        (
            str(row["category"]),
            normalize_bool_label(row["context_support"]),
            str(row["source_dataset"]),
        )
        for row in replication_rows
    )
    check("replication140_row_count", len(replication_rows) == 140, len(replication_rows))
    check("replication140_unique_ids", len(set(replication_ids)) == 140)
    check("replication140_category_balance", all(rep_category[c] == 20 for c in CATEGORIES), rep_category)
    check("replication140_support_balance", rep_support == {"FALSE": 70, "TRUE": 70}, rep_support)
    check("replication140_source_balance", rep_source == {"ASQA": 70, "HALUEVALQA": 70}, rep_source)
    check(
        "replication140_28_strata_equal_5",
        len(rep_strata) == 28 and all(value == 5 for value in rep_strata.values()),
    )
    main_keys = {
        "provenance": {provenance_key(row) for row in main_rows},
        "question": {normalize_text(row["question"]) for row in main_rows},
        "context": {normalize_text(row["context"]) for row in main_rows},
        "factoid": {normalize_text(row["factoid"]) for row in main_rows},
    }
    overlap_counts = {
        "source_dataset+source_id": sum(provenance_key(row) in main_keys["provenance"] for row in replication_rows),
        "normalized_question": sum(normalize_text(row["question"]) in main_keys["question"] for row in replication_rows),
        "normalized_context": sum(normalize_text(row["context"]) in main_keys["context"] for row in replication_rows),
        "normalized_factoid": sum(normalize_text(row["factoid"]) in main_keys["factoid"] for row in replication_rows),
    }
    check("replication140_zero_main_overlap", not any(overlap_counts.values()), overlap_counts)

    # Verify that package-facing freeze manifests resolve to the packaged files.
    main_freeze = read_json(PACKAGE_ROOT / "data/main700/rq3_main700_final.freeze.json")
    main_expected_hash = main_freeze.get("sha256") or main_freeze.get("frozen_dataset", {}).get("sha256")
    check(
        "main700_freeze_hash_matches",
        main_expected_hash == sha256(PACKAGE_ROOT / "data/main700/rq3_main700_final.jsonl"),
        {"manifest_sha256": main_expected_hash},
    )
    rep_freeze = read_json(
        PACKAGE_ROOT / "data/replication140/rq3_replication140_final.freeze.json"
    )
    rep_freeze_failures: list[str] = []
    freeze_entries = [rep_freeze.get("frozen_dataset", {})]
    freeze_entries.extend(rep_freeze.get("inputs", {}).values())
    freeze_entries.extend(rep_freeze.get("synchronized_complete_data", {}).values())
    for entry in freeze_entries:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not relative or not expected:
            continue
        target = PACKAGE_ROOT / Path(relative)
        if not target.exists():
            rep_freeze_failures.append(f"missing:{relative}")
        elif sha256(target) != expected:
            rep_freeze_failures.append(f"hash:{relative}")
    check(
        "replication140_package_freeze_hashes_match",
        not rep_freeze_failures,
        rep_freeze_failures,
    )

    # Deterministically reconstruct the disjoint 351-pool.
    filtered_candidates = []
    for row in large_candidates:
        if provenance_key(row) in main_keys["provenance"]:
            continue
        if normalize_text(row["question"]) in main_keys["question"]:
            continue
        if normalize_text(row["context"]) in main_keys["context"]:
            continue
        if normalize_text(row["factoid"]) in main_keys["factoid"]:
            continue
        filtered_candidates.append(row)
    check(
        "replication_candidate_pool_deterministic_rebuild",
        filtered_candidates == replication_candidates,
        {"rebuilt": len(filtered_candidates), "packaged": len(replication_candidates)},
    )

    # Deterministically reconstruct Replication-140 and its audit crosswalk.
    grouped: dict[tuple[str, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for row in replication_candidates:
        grouped[(str(row["category"]), bool(row["context_support"]), str(row["source_dataset"]))].append(row)
    selected_candidates: list[dict[str, Any]] = []
    rebuilt_reserve: list[dict[str, Any]] = []
    selected_contexts: set[str] = set()
    for category in CATEGORIES:
        for support in SUPPORTS:
            for source in SOURCES:
                ranked = sorted(
                    grouped[(category, support, source)],
                    key=lambda row: (sample_priority(str(row["id"])), str(row["id"])),
                )
                selected_in_stratum: list[dict[str, Any]] = []
                skipped_duplicate_context_ids: set[str] = set()
                for row in ranked:
                    context_key = normalize_text(row["context"])
                    if len(selected_in_stratum) < 5:
                        if context_key in selected_contexts:
                            skipped_duplicate_context_ids.add(str(row["id"]))
                            continue
                        selected_in_stratum.append(row)
                        selected_contexts.add(context_key)
                selected_ids = {str(row["id"]) for row in selected_in_stratum}
                selected_candidates.extend(selected_in_stratum)
                for rank, row in enumerate(ranked, start=1):
                    if str(row["id"]) in selected_ids:
                        continue
                    rebuilt_reserve.append(
                        {
                            **row,
                            "sample_priority": sample_priority(str(row["id"])),
                            "stratum_rank": rank,
                            "initial_selection_status": (
                                "SKIPPED_DUPLICATE_CONTEXT"
                                if str(row["id"]) in skipped_duplicate_context_ids
                                else "RESERVE"
                            ),
                        }
                    )
    rebuilt_replication: list[dict[str, Any]] = []
    rebuilt_selection: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected_candidates, start=1):
        item_id = f"REP_{index:04d}"
        rebuilt_replication.append({**candidate, "id": item_id})
        rebuilt_selection.append(
            {
                "replication_id": item_id,
                "candidate_id": candidate["id"],
                "category": candidate["category"],
                "context_support": candidate["context_support"],
                "source_dataset": candidate["source_dataset"],
                "source_id": candidate["source_id"],
                "sample_priority": sample_priority(str(candidate["id"])),
            }
        )
    check("replication140_deterministic_rebuild", rebuilt_replication == replication_rows)
    check("replication140_selection_crosswalk_rebuild", rebuilt_selection == selection_rows)
    check("replication140_reserve_queue_rebuild", rebuilt_reserve == reserve_rows)

    # Replication annotation agreement and hard-field adjudication.
    rep_a_rows = read_csv(PACKAGE_ROOT / "audit/replication140/annotator_A_reaudited.csv")
    rep_b_rows = read_csv(PACKAGE_ROOT / "audit/replication140/annotator_B_reaudited.csv")
    rep_c_rows = read_csv(PACKAGE_ROOT / "audit/replication140/adjudicator_C.csv")
    rep_a = by_id(rep_a_rows, "id")
    rep_b_id_field = "annotation_id" if "annotation_id" in rep_b_rows[0] else "id"
    rep_b = by_id(rep_b_rows, rep_b_id_field)
    rep_c = by_id(rep_c_rows, "id")
    check("replication_annotation_id_coverage", set(rep_a) == set(rep_b) == set(replication_ids))
    rep_text_mismatches = {"annotator_A": [], "annotator_B": [], "adjudicator_C": []}
    rep_by_identifier = by_id(replication_rows, "id")
    for item_id in replication_ids:
        canonical = rep_by_identifier[item_id]
        for field in ("question", "context", "factoid"):
            if rep_a[item_id].get(field, "") != canonical[field]:
                rep_text_mismatches["annotator_A"].append(f"{item_id}:{field}")
            if rep_b[item_id].get(field, "") != canonical[field]:
                rep_text_mismatches["annotator_B"].append(f"{item_id}:{field}")
        if item_id in rep_c:
            for field in ("question", "context", "factoid"):
                if rep_c[item_id].get(field, "") != canonical[field]:
                    rep_text_mismatches["adjudicator_C"].append(f"{item_id}:{field}")
    check(
        "replication_A_B_C_inputs_match_frozen",
        not any(rep_text_mismatches.values()),
        rep_text_mismatches,
    )
    forbidden_a_fields = {"original_category", "original_context_support"}
    check(
        "replication_annotator_A_blind_packet_schema",
        all("context" in row for row in rep_a_rows)
        and all(not forbidden_a_fields.intersection(row) for row in rep_a_rows),
        {"forbidden_fields": sorted(forbidden_a_fields)},
    )
    rep_fields = {
        "atomicity": ("Atomicity", "atomicity", normalize_label),
        "question_relevance": ("question_relevance", "question_relevance", normalize_label),
        "primary_category": ("primary_category", "primary_category", normalize_label),
        "context_support": ("context_support", "context_support", normalize_bool_label),
        "evidence_clarity": ("evidence_clarity", "evidence_clarity", normalize_label),
    }
    rep_iaa: dict[str, Any] = {}
    rep_disagreement_ids: dict[str, list[str]] = {}
    for output_field, (a_field, b_field, transform) in rep_fields.items():
        labels_a = [transform(rep_a[item_id][a_field]) for item_id in replication_ids]
        labels_b = [transform(rep_b[item_id][b_field]) for item_id in replication_ids]
        rep_iaa[output_field] = agreement(labels_a, labels_b)
        rep_disagreement_ids[output_field] = [
            item_id
            for item_id in replication_ids
            if transform(rep_a[item_id][a_field]) != transform(rep_b[item_id][b_field])
        ]
    rep_any_union = set().union(*rep_disagreement_ids.values())
    rep_hard_union = set().union(
        rep_disagreement_ids["atomicity"],
        rep_disagreement_ids["primary_category"],
        rep_disagreement_ids["context_support"],
    )
    check("replication_any_disagreement_union_count", len(rep_any_union) == 43, len(rep_any_union))
    check("replication_hard_disagreement_union_count", len(rep_hard_union) == 8, len(rep_hard_union))
    check("replication_adjudicator_rows", len(rep_c_rows) == 10, len(rep_c_rows))
    check("replication_all_hard_disagreements_adjudicated", rep_hard_union.issubset(rep_c))
    rep_final_mismatches: list[str] = []
    for row in replication_rows:
        item_id = str(row["id"])
        if item_id in rep_c:
            category = normalize_label(rep_c[item_id]["adjudicator_final_category"])
            support = normalize_bool_label(rep_c[item_id]["adjudicator_context_support"])
            atomicity = normalize_label(rep_c[item_id]["adjudicator__atomicity"])
        else:
            category = normalize_label(rep_a[item_id]["primary_category"])
            support = normalize_bool_label(rep_a[item_id]["context_support"])
            atomicity = normalize_label(rep_a[item_id]["Atomicity"])
        if category != normalize_label(row["category"]):
            rep_final_mismatches.append(f"{item_id}:category")
        if support != normalize_bool_label(row["context_support"]):
            rep_final_mismatches.append(f"{item_id}:support")
        if atomicity != "PASS":
            rep_final_mismatches.append(f"{item_id}:atomicity")
    check("replication_adjudicated_labels_match_frozen", not rep_final_mismatches, rep_final_mismatches)

    # Hash manifest, when present.
    sums_path = PACKAGE_ROOT / "SHA256SUMS.txt"
    hash_failures: list[dict[str, str]] = []
    hash_verified = 0
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            target = PACKAGE_ROOT / Path(relative)
            if not target.exists():
                hash_failures.append({"path": relative, "problem": "missing"})
                continue
            actual = sha256(target)
            hash_verified += 1
            if actual != expected:
                hash_failures.append({"path": relative, "expected": expected, "actual": actual})
        check("sha256_manifest", not hash_failures, {"verified": hash_verified, "failures": hash_failures})
    else:
        check("sha256_manifest", False, "SHA256SUMS.txt is missing")

    metrics = {
        "main700": {
            "records": len(main_rows),
            "pre_adjudication_iaa": main_iaa,
            "hard_disagreement_records": len(main_hard_union),
            "adjudication_rows": len(main_adj_rows),
            "legacy_pre_adjudication_iaa": legacy_iaa,
            "legacy_actions": dict(sorted(legacy_actions.items())),
            "legacy_changed_dimensions": dict(sorted(legacy_changed_dimensions.items())),
            "legacy_relabel_rate": legacy_actions["RELABEL"] / 700,
            "legacy_exclusion_rate": legacy_actions["REMOVE"] / 700,
            "legacy_overall_intervention_rate": (legacy_actions["RELABEL"] + legacy_actions["REMOVE"]) / 700,
            "final_atomicity_validation": (
                "All 700 frozen records are validated as atomic by the final A/B review "
                "and adjudication chain. Development-stage pre-rewrite snapshots are not "
                "part of this final-only package."
            ),
        },
        "replication140": {
            "records": len(replication_rows),
            "pre_adjudication_iaa": rep_iaa,
            "any_field_disagreement_records": len(rep_any_union),
            "hard_field_disagreement_records": len(rep_hard_union),
            "adjudicator_rows": len(rep_c_rows),
            "category_or_support_changes_vs_sampled_set": 0,
            "final_records_with_matching_A_B_text": len(replication_rows)
            - len(
                {
                    item.split(":", 1)[0]
                    for key in ("annotator_A", "annotator_B")
                    for item in rep_text_mismatches[key]
                }
            ),
            "main700_overlap_counts": overlap_counts,
            "candidate_pool": {
                "input": len(large_candidates),
                "excluded_as_main_overlap": len(large_candidates) - len(replication_candidates),
                "retained": len(replication_candidates),
                "selected": len(replication_rows),
                "reserve": len(reserve_rows),
            },
        },
        "auditability_notes": {
            "replication_A_B_text_inputs_match_frozen": not any(
                rep_text_mismatches[key]
                for key in ("annotator_A", "annotator_B")
            ),
            "replication_annotator_A_blind_packet_schema": all("context" in row for row in rep_a_rows)
            and all(not forbidden_a_fields.intersection(row) for row in rep_a_rows),
            "replication_descriptive_disagreements_not_fully_adjudicated": True,
        },
    }

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    result = {
        "schema_version": "rq3-main700-replication140-package-verification-v3",
        "verification_status": overall,
        "checks": checks,
        "metrics": metrics,
    }
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        replication_audit = {
            "audit_schema_version": "rq3-replication140-package-audit-v3",
            "inputs": {
                "dataset": {
                    "path": "data/replication140/rq3_replication140_final.jsonl",
                    "sha256": sha256(PACKAGE_ROOT / "data/replication140/rq3_replication140_final.jsonl"),
                },
                "annotator_A": {
                    "path": "audit/replication140/annotator_A_reaudited.csv",
                    "sha256": sha256(PACKAGE_ROOT / "audit/replication140/annotator_A_reaudited.csv"),
                },
                "annotator_B": {
                    "path": "audit/replication140/annotator_B_reaudited.csv",
                    "sha256": sha256(PACKAGE_ROOT / "audit/replication140/annotator_B_reaudited.csv"),
                },
                "adjudicator_C": {
                    "path": "audit/replication140/adjudicator_C.csv",
                    "sha256": sha256(PACKAGE_ROOT / "audit/replication140/adjudicator_C.csv"),
                },
            },
            "input_alignment": {
                "id_coverage_complete": set(rep_a) == set(rep_b) == set(replication_ids),
                "text_mismatches": rep_text_mismatches,
                "annotator_A_blind_packet_schema": all("context" in row for row in rep_a_rows)
                and all(not forbidden_a_fields.intersection(row) for row in rep_a_rows),
            },
            "pre_adjudication_agreement": rep_iaa,
            "disagreement_ids_by_field": rep_disagreement_ids,
            "any_field_disagreement_records": len(rep_any_union),
            "hard_field_disagreement_records": len(rep_hard_union),
            "adjudication": {
                "rows": len(rep_c_rows),
                "all_hard_disagreements_covered": rep_hard_union.issubset(rep_c),
                "hard_disagreements_missing": sorted(rep_hard_union - set(rep_c)),
                "descriptive_disagreements_not_adjudicated": sorted(rep_any_union - set(rep_c)),
            },
            "final_consistency": {
                "category_support_atomicity_mismatches": rep_final_mismatches,
                "matches_frozen_dataset": not rep_final_mismatches,
                "records_matching_A_B_text": len(replication_rows)
                - len(
                    {
                        item.split(":", 1)[0]
                        for key in ("annotator_A", "annotator_B")
                        for item in rep_text_mismatches[key]
                    }
                ),
            },
        }
        replication_audit_path = (
            PACKAGE_ROOT / "audit/replication140/rq3_replication140_human_audit.json"
        )
        replication_audit_path.write_text(
            json.dumps(replication_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"verification_status": overall, "checks": len(checks), "failed": [item["name"] for item in checks if item["status"] == "FAIL"]}, ensure_ascii=False))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
