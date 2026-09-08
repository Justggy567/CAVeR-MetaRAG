"""Build a deterministic, source-balanced RQ3 replication set of 140 rows."""

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


FIELDS = (
    "id",
    "source_dataset",
    "source_id",
    "question",
    "context",
    "factoid",
    "category",
    "context_support",
)
CATEGORIES = tuple(f"Cat{number}" for number in range(1, 8))
SUPPORT_STATES = (True, False)
SOURCE_DATASETS = ("ASQA", "HALUEVALQA")
SAMPLE_SALT = "rq3-replication140-v1"
TARGET_PER_STRATUM = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            missing = [field for field in FIELDS if field not in row]
            if missing:
                raise ValueError(f"Missing fields at {path}:{line_number}: {missing}")
            rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def provenance_key(row: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(row["source_dataset"]), normalize_text(row["source_id"])


def sample_priority(candidate_id: str) -> str:
    value = f"{SAMPLE_SALT}|{candidate_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temp_path, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
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


def main_overlap_counts(
    selected_rows: list[dict[str, Any]], main_rows: list[dict[str, Any]]
) -> dict[str, int]:
    main_keys = {
        "provenance": {provenance_key(row) for row in main_rows},
        "question": {normalize_text(row["question"]) for row in main_rows},
        "context": {normalize_text(row["context"]) for row in main_rows},
        "factoid": {normalize_text(row["factoid"]) for row in main_rows},
    }
    return {
        "source_dataset+source_id": sum(
            provenance_key(row) in main_keys["provenance"] for row in selected_rows
        ),
        "normalized_question": sum(
            normalize_text(row["question"]) in main_keys["question"]
            for row in selected_rows
        ),
        "normalized_context": sum(
            normalize_text(row["context"]) in main_keys["context"]
            for row in selected_rows
        ),
        "normalized_factoid": sum(
            normalize_text(row["factoid"]) in main_keys["factoid"]
            for row in selected_rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidate_pool)
    main_rows = read_jsonl(args.main)

    grouped: dict[tuple[str, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        stratum = (row["category"], row["context_support"], row["source_dataset"])
        if stratum[0] not in CATEGORIES:
            raise ValueError(f"Unexpected category for {row['id']}: {stratum[0]}")
        if not isinstance(stratum[1], bool):
            raise ValueError(f"context_support must be boolean for {row['id']}")
        if stratum[2] not in SOURCE_DATASETS:
            raise ValueError(f"Unexpected source_dataset for {row['id']}: {stratum[2]}")
        grouped[stratum].append(row)

    expected_strata = [
        (category, support, source)
        for category in CATEGORIES
        for support in SUPPORT_STATES
        for source in SOURCE_DATASETS
    ]
    shortages = {
        "|".join((category, str(support).lower(), source)): TARGET_PER_STRATUM - len(grouped[stratum])
        for stratum in expected_strata
        for category, support, source in (stratum,)
        if len(grouped[stratum]) < TARGET_PER_STRATUM
    }
    if shortages:
        raise ValueError(f"Insufficient candidates for balanced sampling: {shortages}")

    selected_candidate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    selected_contexts: set[str] = set()

    for stratum in expected_strata:
        category, support, source = stratum
        ranked = sorted(
            grouped[stratum], key=lambda row: (sample_priority(row["id"]), row["id"])
        )
        selected_in_stratum: list[dict[str, Any]] = []
        skipped_duplicate_context_ids: set[str] = set()
        for row in ranked:
            context_key = normalize_text(row["context"])
            if len(selected_in_stratum) < TARGET_PER_STRATUM:
                if context_key in selected_contexts:
                    skipped_duplicate_context_ids.add(row["id"])
                    continue
                selected_in_stratum.append(row)
                selected_contexts.add(context_key)

        if len(selected_in_stratum) != TARGET_PER_STRATUM:
            raise ValueError(
                "Could not fill stratum after enforcing context uniqueness: "
                f"{category}|{str(support).lower()}|{source}"
            )

        selected_ids = {row["id"] for row in selected_in_stratum}
        for row in selected_in_stratum:
            selected_candidate_rows.append(row)
        for rank, row in enumerate(ranked, start=1):
            if row["id"] in selected_ids:
                continue
            reserve_rows.append(
                {
                    **row,
                    "sample_priority": sample_priority(row["id"]),
                    "stratum_rank": rank,
                    "initial_selection_status": (
                        "SKIPPED_DUPLICATE_CONTEXT"
                        if row["id"] in skipped_duplicate_context_ids
                        else "RESERVE"
                    ),
                }
            )

    final_rows: list[dict[str, Any]] = []
    for index, candidate_row in enumerate(selected_candidate_rows, start=1):
        replication_id = f"REP_{index:04d}"
        final_row = dict(candidate_row)
        final_row["id"] = replication_id
        final_rows.append(final_row)
        selection_rows.append(
            {
                "replication_id": replication_id,
                "candidate_id": candidate_row["id"],
                "category": candidate_row["category"],
                "context_support": candidate_row["context_support"],
                "source_dataset": candidate_row["source_dataset"],
                "source_id": candidate_row["source_id"],
                "sample_priority": sample_priority(candidate_row["id"]),
            }
        )

    if len(final_rows) != 140:
        raise AssertionError(f"Expected 140 selected rows, got {len(final_rows)}")

    internal_duplicates = {
        "provenance": duplicate_groups(final_rows, "provenance"),
        "question": duplicate_groups(final_rows, "question"),
        "context": duplicate_groups(final_rows, "context"),
        "factoid": duplicate_groups(final_rows, "factoid"),
    }
    if any(internal_duplicates.values()):
        raise ValueError(f"Selected set contains duplicate groups: {internal_duplicates}")
    overlaps = main_overlap_counts(final_rows, main_rows)
    if any(overlaps.values()):
        raise ValueError(f"Selected set overlaps Main-700: {overlaps}")

    write_jsonl_atomic(args.output, final_rows)
    write_jsonl_atomic(args.selection, selection_rows)
    write_jsonl_atomic(args.reserve, reserve_rows)

    three_way_counts = Counter(
        (
            row["category"],
            str(row["context_support"]).lower(),
            row["source_dataset"],
        )
        for row in final_rows
    )
    source_counts = Counter(row["source_dataset"] for row in final_rows)
    category_support_counts = Counter(
        (row["category"], str(row["context_support"]).lower())
        for row in final_rows
    )
    manifest = {
        "name": "RQ3-FactoidReplication140-v1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_design": {
            "total": 140,
            "target_per_category_support_source_stratum": TARGET_PER_STRATUM,
            "sample_priority": f'SHA256("{SAMPLE_SALT}|" + candidate_id)',
            "selection_rule": (
                "Within each category × context_support × source_dataset stratum, "
                "select the five lowest hash priorities while enforcing normalized-context uniqueness."
            ),
            "outcome_blinding": (
                "Sampling does not use router, verifier, escalation, or RQ3 experimental outputs."
            ),
        },
        "inputs": {
            "candidate_pool": {
                "path": str(args.candidate_pool),
                "sha256": sha256(args.candidate_pool),
                "rows": len(candidates),
            },
            "main700": {
                "path": str(args.main),
                "sha256": sha256(args.main),
                "rows": len(main_rows),
            },
        },
        "outputs": {
            "dataset": {
                "path": str(args.output),
                "sha256": sha256(args.output),
                "rows": len(final_rows),
            },
            "selection_crosswalk": {
                "path": str(args.selection),
                "sha256": sha256(args.selection),
                "rows": len(selection_rows),
            },
            "reserve_queue": {
                "path": str(args.reserve),
                "sha256": sha256(args.reserve),
                "rows": len(reserve_rows),
            },
        },
        "selected_distribution": {
            "by_source_dataset": dict(sorted(source_counts.items())),
            "by_category_and_support": {
                f"{category}|{support}": category_support_counts[(category, support)]
                for category in CATEGORIES
                for support in ("true", "false")
            },
            "by_category_support_and_source": {
                f"{category}|{support}|{source}": three_way_counts[(category, support, source)]
                for category in CATEGORIES
                for support in ("true", "false")
                for source in SOURCE_DATASETS
            },
        },
        "validation": {
            "main700_overlap_counts": overlaps,
            "internal_duplicate_groups": internal_duplicates,
            "unique_replication_ids": len({row["id"] for row in final_rows}),
            "unique_source_questions": len({provenance_key(row) for row in final_rows}),
            "all_28_strata_equal_5": all(count == 5 for count in three_way_counts.values())
            and len(three_way_counts) == 28,
        },
        "replacement_policy": (
            "Replacement must use the next eligible row from the same category × support × "
            "source_dataset reserve stratum and must preserve all overlap and duplicate constraints."
        ),
    }
    write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
