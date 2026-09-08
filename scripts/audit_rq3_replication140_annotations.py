#!/usr/bin/env python3
"""Review the completeness of RQ3 replica set construction and manual review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELDS = ("atomicity", "question_relevance", "category", "context_support", "evidence_clarity")
HARD_FIELDS = ("atomicity", "category", "context_support")
DESCRIPTIVE_FIELDS = ("question_relevance", "evidence_clarity")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def norm_label(value: Any) -> str:
    return norm_text(value).upper()


def norm_category(value: Any) -> str:
    value = norm_text(value)
    if value.upper() in {"N/A", "NA", ""}:
        return "N/A"
    if value.lower().startswith("cat"):
        suffix = value[3:].strip()
        return f"Cat{suffix}"
    return value


def norm_support(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    value = norm_label(value)
    mapping = {"1": "TRUE", "0": "FALSE", "YES": "TRUE", "NO": "FALSE"}
    return mapping.get(value, value)


def index_unique(rows: Iterable[dict[str, Any]], id_key: str, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        row_id = norm_text(row.get(id_key))
        if row_id in indexed:
            duplicates.append(row_id)
        indexed[row_id] = row
    if duplicates:
        raise ValueError(f"Duplicate IDs in {source}: {sorted(set(duplicates))}")
    return indexed


def labels_a(row: dict[str, Any]) -> dict[str, str]:
    return {
        "atomicity": norm_label(row.get("Atomicity")),
        "question_relevance": norm_label(row.get("question_relevance")),
        "category": norm_category(row.get("primary_category")),
        "context_support": norm_support(row.get("context_support")),
        "evidence_clarity": norm_label(row.get("evidence_clarity")),
    }


def labels_b(row: dict[str, Any]) -> dict[str, str]:
    return {
        "atomicity": norm_label(row.get("atomicity")),
        "question_relevance": norm_label(row.get("question_relevance")),
        "category": norm_category(row.get("primary_category")),
        "context_support": norm_support(row.get("context_support")),
        "evidence_clarity": norm_label(row.get("evidence_clarity")),
    }


def labels_c(row: dict[str, Any]) -> dict[str, str]:
    return {
        "atomicity": norm_label(row.get("adjudicator__atomicity")),
        "question_relevance": norm_label(row.get("adjudicator__question_relevance")),
        "category": norm_category(row.get("adjudicator_final_category")),
        "context_support": norm_support(row.get("adjudicator_context_support")),
        "evidence_clarity": norm_label(row.get("adjudicator_evidence_clarity")),
    }


def cohen_kappa(a_values: list[str], b_values: list[str]) -> dict[str, Any]:
    if len(a_values) != len(b_values):
        raise ValueError("Kappa inputs differ in length")
    n = len(a_values)
    agreements = sum(a == b for a, b in zip(a_values, b_values))
    observed = agreements / n if n else math.nan
    a_counts, b_counts = Counter(a_values), Counter(b_values)
    labels = set(a_counts) | set(b_counts)
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels) if n else math.nan
    if not n or math.isclose(1.0 - expected, 0.0):
        kappa: float | None = None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "n": n,
        "agreements": agreements,
        "disagreements": n - agreements,
        "percent_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "annotator_A_distribution": dict(sorted(a_counts.items())),
        "annotator_B_distribution": dict(sorted(b_counts.items())),
    }


def field_mismatches(
    annotation: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    mapping: dict[str, str],
    exclude_ids: set[str] | None = None,
) -> dict[str, list[str]]:
    exclude_ids = exclude_ids or set()
    result: dict[str, list[str]] = {}
    for annotation_field, base_field in mapping.items():
        bad = [
            row_id
            for row_id, row in annotation.items()
            if row_id in base
            and row_id not in exclude_ids
            and norm_text(row.get(annotation_field)) != norm_text(base[row_id].get(base_field))
        ]
        result[annotation_field] = sorted(bad)
    return result


def duplicate_groups(rows: list[dict[str, Any]], key_fn) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(str(row.get("id", "")))
    return [sorted(ids) for value, ids in groups.items() if value and len(ids) > 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--main700", type=Path, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--adjudicator-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accept-post-review-stratum-drift",
        action="store_true",
        help="Keep the fixed 140 IDs after human reclassification instead of replacing rows.",
    )
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    candidate_rows = read_jsonl(args.candidate_pool)
    main_rows = read_jsonl(args.main700)
    a_rows = read_csv(args.annotator_a)
    b_rows = read_csv(args.annotator_b)
    c_rows = read_csv(args.adjudicator_c)

    base = index_unique(dataset_rows, "id", "replication dataset")
    a_index = index_unique(a_rows, "id", "Annotator A")
    b_id_field = "annotation_id" if b_rows and "annotation_id" in b_rows[0] else "id"
    b_index = index_unique(b_rows, b_id_field, "Annotator B")
    c_index = index_unique(c_rows, "id", "Adjudicator C")

    base_ids = set(base)
    a_ids, b_ids, c_ids = set(a_index), set(b_index), set(c_index)
    common_ids = sorted(base_ids & a_ids & b_ids)

    a_label_map = {row_id: labels_a(row) for row_id, row in a_index.items()}
    b_label_map = {row_id: labels_b(row) for row_id, row in b_index.items()}
    c_label_map = {row_id: labels_c(row) for row_id, row in c_index.items()}

    disagreements: dict[str, list[str]] = {}
    agreement_stats: dict[str, Any] = {}
    c_missing_by_field: dict[str, list[str]] = {}
    c_blank_by_field: dict[str, list[str]] = {}
    for field in FIELDS:
        disagreements[field] = [
            row_id for row_id in common_ids if a_label_map[row_id][field] != b_label_map[row_id][field]
        ]
        agreement_stats[field] = cohen_kappa(
            [a_label_map[row_id][field] for row_id in common_ids],
            [b_label_map[row_id][field] for row_id in common_ids],
        )
        c_missing_by_field[field] = [row_id for row_id in disagreements[field] if row_id not in c_ids]
        c_blank_by_field[field] = [
            row_id
            for row_id in disagreements[field]
            if row_id in c_ids and c_label_map[row_id][field] in {"", "N/A"}
        ]

    consensus_vs_original: dict[str, list[str]] = {"category": [], "context_support": []}
    for row_id in common_ids:
        if a_label_map[row_id]["category"] == b_label_map[row_id]["category"]:
            original_category = norm_category(a_index[row_id].get("original_category"))
            if original_category not in {"", "N/A"} and a_label_map[row_id]["category"] != original_category:
                consensus_vs_original["category"].append(row_id)
        if a_label_map[row_id]["context_support"] == b_label_map[row_id]["context_support"]:
            original_support = norm_support(a_index[row_id].get("original_context_support"))
            if original_support and a_label_map[row_id]["context_support"] != original_support:
                consensus_vs_original["context_support"].append(row_id)

    required_by_field = {field: list(disagreements[field]) for field in FIELDS}
    for field in ("category", "context_support"):
        required_by_field[field] = sorted(set(required_by_field[field]) | set(consensus_vs_original[field]))
    for field in HARD_FIELDS:
        c_missing_by_field[field] = [row_id for row_id in required_by_field[field] if row_id not in c_ids]
        c_blank_by_field[field] = [
            row_id
            for row_id in required_by_field[field]
            if row_id in c_ids and c_label_map[row_id][field] in {"", "N/A"}
        ]

    union_disagreements = sorted({row_id for ids in disagreements.values() for row_id in ids})
    missing_union = sorted({row_id for ids in c_missing_by_field.values() for row_id in ids})
    blank_union = sorted({row_id for ids in c_blank_by_field.values() for row_id in ids})

    strata = Counter(
        f"{row.get('category')}|{str(row.get('context_support')).lower()}|{row.get('source_dataset')}"
        for row in dataset_rows
    )
    source_counts = Counter(str(row.get("source_dataset")) for row in dataset_rows)

    main_provenance = {(norm_text(row.get("source_dataset")), norm_text(row.get("source_id"))) for row in main_rows}
    main_questions = {norm_text(row.get("question")) for row in main_rows}
    main_contexts = {norm_text(row.get("context")) for row in main_rows}
    main_factoids = {norm_text(row.get("factoid")) for row in main_rows}
    overlap = {
        "source_dataset+source_id": sum(
            (norm_text(row.get("source_dataset")), norm_text(row.get("source_id"))) in main_provenance
            for row in dataset_rows
        ),
        "normalized_question": sum(norm_text(row.get("question")) in main_questions for row in dataset_rows),
        "normalized_context": sum(norm_text(row.get("context")) in main_contexts for row in dataset_rows),
        "normalized_factoid": sum(norm_text(row.get("factoid")) in main_factoids for row in dataset_rows),
    }

    c_vs_base: dict[str, list[dict[str, str]]] = {"category": [], "context_support": []}
    for row_id in sorted(c_ids & base_ids):
        base_category = norm_category(base[row_id].get("category"))
        final_category = c_label_map[row_id]["category"]
        if final_category not in {"", "N/A"} and final_category != base_category:
            c_vs_base["category"].append({"id": row_id, "base": base_category, "adjudicated": final_category})
        base_support = norm_support(base[row_id].get("context_support"))
        final_support = c_label_map[row_id]["context_support"]
        if final_support and final_support != base_support:
            c_vs_base["context_support"].append(
                {"id": row_id, "base": base_support, "adjudicated": final_support}
            )

    source_alignment = {
        "annotator_A": field_mismatches(
            a_index,
            base,
            {"source_dataset": "source_dataset", "source_id": "source_id", "question": "question", "factoid": "factoid"},
        ),
        "annotator_B": field_mismatches(
            b_index,
            base,
            {
                "source_dataset": "source_dataset",
                "source_id": "source_id",
                "question": "question",
                "context": "context",
                "factoid": "factoid",
            },
        ),
        "adjudicator_C": field_mismatches(
            c_index,
            base,
            {"question": "question", "context": "context", "factoid": "factoid"},
        ),
    }

    final_category_mismatch: list[str] = []
    final_support_mismatch: list[str] = []
    final_atomicity_fail: list[str] = []
    for row_id in common_ids:
        expected: dict[str, str] = {}
        for field in HARD_FIELDS:
            expected[field] = (
                c_label_map[row_id][field]
                if row_id in required_by_field[field] and row_id in c_label_map
                else a_label_map[row_id][field]
            )
        if expected["atomicity"] != "PASS":
            final_atomicity_fail.append(row_id)
        if norm_category(base[row_id].get("category")) != expected["category"]:
            final_category_mismatch.append(row_id)
        if norm_support(base[row_id].get("context_support")) != expected["context_support"]:
            final_support_mismatch.append(row_id)

    hard_missing_union = sorted({row_id for field in HARD_FIELDS for row_id in c_missing_by_field[field]})
    hard_blank_union = sorted({row_id for field in HARD_FIELDS for row_id in c_blank_by_field[field]})

    report = {
        "audit_schema_version": "rq3-replication140-annotation-audit-v2",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "dataset": args.dataset,
                "candidate_pool": args.candidate_pool,
                "main700": args.main700,
                "annotator_A": args.annotator_a,
                "annotator_B": args.annotator_b,
                "adjudicator_C": args.adjudicator_c,
            }.items()
        },
        "construction": {
            "dataset_rows": len(dataset_rows),
            "candidate_pool_rows": len(candidate_rows),
            "main700_rows": len(main_rows),
            "source_distribution": dict(sorted(source_counts.items())),
            "stratum_distribution": dict(sorted(strata.items())),
            "all_28_strata_equal_5": len(strata) == 28 and set(strata.values()) == {5},
            "post_review_stratum_min": min(strata.values()) if strata else 0,
            "post_review_stratum_max": max(strata.values()) if strata else 0,
            "fixed_set_retained_after_review": args.accept_post_review_stratum_drift,
            "main700_overlap_counts": overlap,
            "internal_duplicate_groups": {
                "provenance": duplicate_groups(
                    dataset_rows,
                    lambda row: f"{norm_text(row.get('source_dataset'))}|{norm_text(row.get('source_id'))}",
                ),
                "question": duplicate_groups(dataset_rows, lambda row: norm_text(row.get("question"))),
                "context": duplicate_groups(dataset_rows, lambda row: norm_text(row.get("context"))),
                "factoid": duplicate_groups(dataset_rows, lambda row: norm_text(row.get("factoid"))),
            },
        },
        "review": {
            "audit_scope": {
                "hard_final_fields": list(HARD_FIELDS),
                "descriptive_non_filter_fields": list(DESCRIPTIVE_FIELDS),
                "question_relevance_filtering": False,
            },
            "row_counts": {"dataset": len(base), "annotator_A": len(a_index), "annotator_B": len(b_index), "adjudicator_C": len(c_index)},
            "id_coverage": {
                "annotator_A_missing": sorted(base_ids - a_ids),
                "annotator_A_extra": sorted(a_ids - base_ids),
                "annotator_B_missing": sorted(base_ids - b_ids),
                "annotator_B_extra": sorted(b_ids - base_ids),
                "adjudicator_C_extra": sorted(c_ids - base_ids),
            },
            "source_alignment_mismatches": source_alignment,
            "pre_adjudication_agreement": agreement_stats,
            "disagreement_ids_by_field": disagreements,
            "disagreement_union_count": len(union_disagreements),
            "disagreement_union_ids": union_disagreements,
            "consensus_vs_original_core_ids_by_field": consensus_vs_original,
            "required_core_adjudication_ids_by_field": {
                field: required_by_field[field] for field in HARD_FIELDS
            },
            "adjudication": {
                "rows_provided": len(c_index),
                "covered_disagreement_ids": sorted(c_ids & set(union_disagreements)),
                "missing_disagreement_ids_by_field": c_missing_by_field,
                "blank_decisions_by_field": c_blank_by_field,
                "missing_disagreement_union_count": len(missing_union),
                "missing_disagreement_union_ids": missing_union,
                "blank_decision_union_ids": blank_union,
                "missing_hard_field_disagreement_union_ids": hard_missing_union,
                "blank_hard_field_decision_union_ids": hard_blank_union,
                "complete_for_core_hard_fields": not hard_missing_union and not hard_blank_union,
                "complete_for_all_review_fields": not missing_union and not blank_union,
                "final_category_or_support_changes_vs_sampled_dataset": c_vs_base,
            },
            "final_consistency": {
                "category_mismatch_ids": final_category_mismatch,
                "context_support_mismatch_ids": final_support_mismatch,
                "final_atomicity_fail_ids": final_atomicity_fail,
            },
        },
    }

    construction_ok = (
        len(dataset_rows) == 140
        and len(candidate_rows) == 351
        and len(main_rows) == 700
        and (
            report["construction"]["all_28_strata_equal_5"]
            or args.accept_post_review_stratum_drift
        )
        and all(value == 0 for value in overlap.values())
        and all(not groups for groups in report["construction"]["internal_duplicate_groups"].values())
    )
    double_review_ok = (
        len(a_index) == 140
        and len(b_index) == 140
        and not report["review"]["id_coverage"]["annotator_A_missing"]
        and not report["review"]["id_coverage"]["annotator_A_extra"]
        and not report["review"]["id_coverage"]["annotator_B_missing"]
        and not report["review"]["id_coverage"]["annotator_B_extra"]
        and all(not ids for ids in source_alignment["annotator_A"].values())
        and all(not ids for ids in source_alignment["annotator_B"].values())
    )
    adjudication_ok = report["review"]["adjudication"]["complete_for_core_hard_fields"]
    final_consistency_ok = (
        not final_category_mismatch
        and not final_support_mismatch
        and not final_atomicity_fail
    )
    report["gate"] = {
        "construction_complete": construction_ok,
        "exact_post_review_stratum_balance": report["construction"]["all_28_strata_equal_5"],
        "post_review_stratum_drift_accepted": args.accept_post_review_stratum_drift,
        "double_review_complete": double_review_ok,
        "adjudication_complete": adjudication_ok,
        "final_consistency_complete": final_consistency_ok,
        "replication_run_authorized_by_audit": construction_ok and double_review_ok and adjudication_ok and final_consistency_ok,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], ensure_ascii=False))
    print(f"report={args.output.resolve()}")
    return 0 if report["gate"]["replication_run_authorized_by_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
