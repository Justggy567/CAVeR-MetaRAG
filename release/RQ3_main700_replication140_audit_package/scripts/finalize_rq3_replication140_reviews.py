#!/usr/bin/env python3
"""Create the canonical reviewed Replication-140 JSONL from A/B/C exports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HARD_FIELDS = ("atomicity", "category", "context_support")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def norm_label(value: Any) -> str:
    return norm_text(value).upper()


def norm_category(value: Any) -> str:
    value = norm_text(value)
    if value.lower().startswith("cat"):
        return f"Cat{value[3:].strip()}"
    return value


def norm_support(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    value = norm_label(value)
    return {"1": "TRUE", "0": "FALSE", "YES": "TRUE", "NO": "FALSE"}.get(value, value)


def labels_a(row: dict[str, Any]) -> dict[str, str]:
    return {
        "atomicity": norm_label(row.get("Atomicity")),
        "category": norm_category(row.get("primary_category")),
        "context_support": norm_support(row.get("context_support")),
    }


def labels_b(row: dict[str, Any]) -> dict[str, str]:
    return {
        "atomicity": norm_label(row.get("atomicity")),
        "category": norm_category(row.get("primary_category")),
        "context_support": norm_support(row.get("context_support")),
    }


def labels_c(row: dict[str, Any]) -> dict[str, str]:
    return {
        "atomicity": norm_label(row.get("adjudicator__atomicity")),
        "category": norm_category(row.get("adjudicator_final_category")),
        "context_support": norm_support(row.get("adjudicator_context_support")),
    }


def unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = norm_text(row.get(key))
        if not row_id or row_id in result:
            raise ValueError(f"Missing or duplicate ID in {label}: {row_id!r}")
        result[row_id] = row
    return result


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--adjudicator-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    baseline = unique_index(baseline_rows, "id", "baseline")
    a = unique_index(read_csv(args.annotator_a), "id", "Annotator A")
    b_rows = read_csv(args.annotator_b)
    b_id_field = "annotation_id" if b_rows and "annotation_id" in b_rows[0] else "id"
    b = unique_index(b_rows, b_id_field, "Annotator B")
    c = unique_index(read_csv(args.adjudicator_c), "id", "Adjudicator C")
    if len(baseline) != 140 or set(a) != set(baseline) or set(b) != set(baseline):
        raise ValueError("Baseline and both annotator exports must contain the same 140 IDs")
    if not set(c).issubset(baseline):
        raise ValueError(f"Unknown adjudication IDs: {sorted(set(c) - set(baseline))}")

    final_rows: list[dict[str, Any]] = []
    changed_text_ids: list[str] = []
    changed_label_ids: list[str] = []
    unadjudicated: dict[str, list[str]] = {field: [] for field in HARD_FIELDS}
    atomicity_fail: list[str] = []

    for base_row in baseline_rows:
        row_id = str(base_row["id"])
        a_row, b_row = a[row_id], b[row_id]
        for field in ("source_dataset", "source_id", "question", "factoid"):
            if norm_text(a_row.get(field)) != norm_text(b_row.get(field)):
                raise ValueError(f"A/B text disagreement for {row_id}.{field}")
        for field in ("source_dataset", "source_id"):
            if norm_text(base_row.get(field)) != norm_text(b_row.get(field)):
                raise ValueError(f"Provenance changed for {row_id}.{field}")
        if norm_text(base_row.get("question")) != norm_text(b_row.get("question")):
            raise ValueError(f"Question changed for {row_id}")
        if norm_text(base_row.get("context")) != norm_text(b_row.get("context")):
            raise ValueError(f"Context changed for {row_id}")

        a_labels, b_labels = labels_a(a_row), labels_b(b_row)
        c_labels = labels_c(c[row_id]) if row_id in c else None
        final_labels: dict[str, str] = {}
        for field in HARD_FIELDS:
            original_label = None
            if field == "category":
                original_label = norm_category(a_row.get("original_category"))
            elif field == "context_support":
                original_label = norm_support(a_row.get("original_context_support"))
            needs_adjudication = a_labels[field] != b_labels[field]
            if original_label and a_labels[field] == b_labels[field] and a_labels[field] != original_label:
                needs_adjudication = True
            if needs_adjudication and c_labels is None:
                unadjudicated[field].append(row_id)
                continue
            if needs_adjudication:
                candidate = c_labels[field]
                if candidate in {"", "N/A"}:
                    unadjudicated[field].append(row_id)
                    continue
                final_labels[field] = candidate
            else:
                final_labels[field] = a_labels[field]

        if any(unadjudicated[field] and unadjudicated[field][-1] == row_id for field in HARD_FIELDS):
            continue
        if final_labels["atomicity"] not in {"PASS", "FAIL"}:
            raise ValueError(f"Invalid final atomicity for {row_id}: {final_labels['atomicity']!r}")
        if final_labels["category"] not in {f"Cat{i}" for i in range(1, 8)}:
            raise ValueError(f"Invalid final category for {row_id}: {final_labels['category']!r}")
        if final_labels["context_support"] not in {"TRUE", "FALSE"}:
            raise ValueError(f"Invalid final support for {row_id}: {final_labels['context_support']!r}")
        if final_labels["atomicity"] != "PASS":
            atomicity_fail.append(row_id)

        final_row = dict(base_row)
        final_row["category"] = final_labels["category"]
        final_row["context_support"] = final_labels["context_support"] == "TRUE"
        if norm_text(final_row["factoid"]) != norm_text(base_row.get("factoid")):
            changed_text_ids.append(row_id)
        if (
            final_row["category"] != base_row.get("category")
            or final_row["context_support"] != bool(base_row.get("context_support"))
        ):
            changed_label_ids.append(row_id)
        final_rows.append(final_row)

    unresolved = sorted({row_id for ids in unadjudicated.values() for row_id in ids})
    if unresolved:
        raise ValueError(f"Unadjudicated hard-field disagreements: {unresolved}")
    if atomicity_fail:
        raise ValueError(f"Final dataset contains non-atomic records requiring replacement: {atomicity_fail}")
    if len(final_rows) != 140:
        raise ValueError(f"Expected 140 finalized rows, got {len(final_rows)}")

    write_jsonl_atomic(args.output, final_rows)
    print(
        json.dumps(
            {
                "rows": len(final_rows),
                "changed_text_ids": changed_text_ids,
                "changed_label_ids": changed_label_ids,
                "adjudication_rows_provided": len(c),
                "atomicity_fail_ids": atomicity_fail,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
