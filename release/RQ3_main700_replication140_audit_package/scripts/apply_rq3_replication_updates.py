#!/usr/bin/env python3
"""Apply clean JSONL record replacements to a complete RQ3 replication set."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


FIELDS = ("id", "source_dataset", "source_id", "question", "context", "factoid", "category", "context_support")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if tuple(row) != FIELDS:
                raise ValueError(f"Non-canonical schema at {path}:{line_number}: {tuple(row)}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    updates = read_jsonl(args.updates)
    by_id = {row["id"]: row for row in baseline}
    if len(by_id) != len(baseline):
        raise ValueError("Duplicate IDs in baseline")
    changed_ids: list[str] = []
    for update in updates:
        row_id = update["id"]
        current = by_id.get(row_id)
        if current is None:
            raise ValueError(f"Unknown update ID: {row_id}")
        for field in ("source_dataset", "source_id", "question", "context"):
            if update[field] != current[field]:
                raise ValueError(f"Update changes protected field {row_id}.{field}")
        if update != current:
            changed_ids.append(row_id)
        by_id[row_id] = update
    output = [by_id[row["id"]] for row in baseline]
    if len(output) != 140:
        raise ValueError(f"Expected 140 rows, got {len(output)}")
    write_jsonl(args.output, output)
    print(json.dumps({"rows": len(output), "updates": len(updates), "changed_ids": changed_ids}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
