"""Build an RQ3 replication candidate pool disjoint from Main-700.

The filter operates at source-question level, not merely record level. It also
removes normalized exact question, context, and factoid overlaps as safeguards
against duplicated records with inconsistent provenance identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "id",
    "source_dataset",
    "source_id",
    "question",
    "context",
    "factoid",
    "category",
    "context_support",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Missing fields at {path}:{line_number}: {missing}")
            rows.append(row)
    return rows


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def provenance_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_text(row["source_dataset"]),
        normalize_text(row["source_id"]),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temp_path, path)


def duplicate_groups(
    rows: list[dict[str, Any]], key_name: str
) -> list[list[str]]:
    grouped: dict[Any, list[str]] = defaultdict(list)
    for row in rows:
        if key_name == "provenance":
            key = provenance_key(row)
        else:
            key = normalize_text(row[key_name])
        grouped[key].append(row["id"])
    return [ids for ids in grouped.values() if len(ids) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_rows = read_jsonl(args.main)
    candidate_rows = read_jsonl(args.candidate_pool)

    main_keys = {
        "provenance": {provenance_key(row) for row in main_rows},
        "question": {normalize_text(row["question"]) for row in main_rows},
        "context": {normalize_text(row["context"]) for row in main_rows},
        "factoid": {normalize_text(row["factoid"]) for row in main_rows},
    }

    kept_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    reason_combinations: Counter[str] = Counter()
    for row in candidate_rows:
        reasons: list[str] = []
        if provenance_key(row) in main_keys["provenance"]:
            reasons.append("source_dataset+source_id")
        if normalize_text(row["question"]) in main_keys["question"]:
            reasons.append("normalized_question")
        if normalize_text(row["context"]) in main_keys["context"]:
            reasons.append("normalized_context")
        if normalize_text(row["factoid"]) in main_keys["factoid"]:
            reasons.append("normalized_factoid")

        if reasons:
            excluded_rows.append(row)
            reason_counts.update(reasons)
            reason_combinations[" + ".join(reasons)] += 1
        else:
            kept_rows.append(row)

    if len(kept_rows) + len(excluded_rows) != len(candidate_rows):
        raise AssertionError("Candidate row accounting failed")
    if len({row["id"] for row in kept_rows}) != len(kept_rows):
        raise ValueError("Replication candidate pool contains duplicate candidate IDs")

    write_jsonl_atomic(args.output, kept_rows)

    output_rows = read_jsonl(args.output)
    overlap_checks = {
        "source_dataset+source_id": sum(
            provenance_key(row) in main_keys["provenance"] for row in output_rows
        ),
        "normalized_question": sum(
            normalize_text(row["question"]) in main_keys["question"]
            for row in output_rows
        ),
        "normalized_context": sum(
            normalize_text(row["context"]) in main_keys["context"]
            for row in output_rows
        ),
        "normalized_factoid": sum(
            normalize_text(row["factoid"]) in main_keys["factoid"]
            for row in output_rows
        ),
    }
    if any(overlap_checks.values()):
        raise AssertionError(f"Main-700 overlap remains after filtering: {overlap_checks}")

    stratum_counts = Counter(
        (row["category"], str(bool(row["context_support"])).lower())
        for row in output_rows
    )
    category_counts = Counter(row["category"] for row in output_rows)
    source_counts = Counter(row["source_dataset"] for row in output_rows)
    all_categories = [f"Cat{number}" for number in range(1, 8)]
    category_deficits = {
        category: max(0, 20 - category_counts.get(category, 0))
        for category in all_categories
    }
    stratum_deficits = {
        f"{category}|{support}": max(
            0, 10 - stratum_counts.get((category, support), 0)
        )
        for category in all_categories
        for support in ("true", "false")
    }
    can_sample_20_per_category = all(
        category_counts.get(category, 0) >= 20 for category in all_categories
    )
    can_sample_10_per_stratum = all(
        stratum_counts.get((category, support), 0) >= 10
        for category in all_categories
        for support in ("true", "false")
    )
    manifest = {
        "name": "RQ3-replication-candidate-pool-351",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Candidate pool for an RQ3 replication split that is independent "
            "of the latest Main-700 at source-question level."
        ),
        "source_candidate_pool": {
            "path": str(args.candidate_pool),
            "sha256": sha256(args.candidate_pool),
            "rows": len(candidate_rows),
        },
        "excluded_main_set": {
            "path": str(args.main),
            "sha256": sha256(args.main),
            "rows": len(main_rows),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": len(output_rows),
        },
        "filter_policy": [
            "Exclude a candidate sharing normalized source_dataset + source_id with Main-700.",
            "Exclude a candidate sharing a normalized exact question with Main-700.",
            "Exclude a candidate sharing a normalized exact context with Main-700.",
            "Exclude a candidate sharing a normalized exact factoid with Main-700.",
            "Preserve the original candidate ID and all record fields for retained rows.",
        ],
        "normalization": "Unicode NFKC, case-fold, replace non-alphanumeric characters with spaces, collapse whitespace.",
        "counts": {
            "input_candidates": len(candidate_rows),
            "excluded_candidates": len(excluded_rows),
            "retained_candidates": len(output_rows),
            "exclusion_reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
            "exclusion_reason_combinations": dict(sorted(reason_combinations.items())),
        },
        "independence_checks": overlap_checks,
        "retained_distribution": {
            "by_source_dataset": dict(sorted(source_counts.items())),
            "by_category": {
                category: category_counts.get(category, 0)
                for category in all_categories
            },
            "by_category_and_context_support": {
                f"{category}|{support}": stratum_counts.get((category, support), 0)
                for category in all_categories
                for support in ("true", "false")
            },
        },
        "internal_duplicate_groups": {
            "provenance": duplicate_groups(output_rows, "provenance"),
            "question": duplicate_groups(output_rows, "question"),
            "context": duplicate_groups(output_rows, "context"),
            "factoid": duplicate_groups(output_rows, "factoid"),
        },
        "sampling_readiness": {
            "target": "N=140: 20 per category, including 10 supported and 10 unsupported.",
            "can_sample_20_per_category": can_sample_20_per_category,
            "can_sample_10_per_category_and_support_state": can_sample_10_per_stratum,
            "additional_candidates_needed_for_20_per_category": {
                "by_category": category_deficits,
                "minimum_total": sum(category_deficits.values()),
            },
            "additional_candidates_needed_for_10_per_category_and_support_state": {
                "by_stratum": stratum_deficits,
                "minimum_total": sum(stratum_deficits.values()),
            },
            "note": (
                "The pool can support balanced N=140 sampling. Final selection must still "
                "enforce duplicate/context constraints and human-audit requirements."
                if can_sample_10_per_stratum
                else "This pool is independent from Main-700 but lacks enough candidates in one or more strata. "
                "Additional independently sourced candidates are required before balanced replication sampling."
            ),
        },
    }
    write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
