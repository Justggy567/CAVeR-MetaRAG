"""Append-only JSONL result ledger with non-overwriting run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ResponseResult


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class RunLedger:
    def __init__(self, run_directory: str | Path, *, resume: bool = False):
        self.run_directory = Path(run_directory)
        if self.run_directory.exists() and any(self.run_directory.iterdir()) and not resume:
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {self.run_directory}")
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.responses_path = self.run_directory / "responses.jsonl"
        self.factoids_path = self.run_directory / "factoids.jsonl"
        self.calls_path = self.run_directory / "calls.jsonl"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.run_directory / "manifest.json", manifest)

    def completed_response_ids(self) -> set[str]:
        if not self.responses_path.exists():
            return set()
        ids: set[str] = set()
        with self.responses_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                response_id = str(json.loads(line)["response_id"])
                if response_id in ids:
                    raise ValueError(f"Duplicate completed response at line {line_number}: {response_id}")
                ids.add(response_id)
        return ids

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()

    def append_result(self, result: ResponseResult) -> None:
        self._append(self.responses_path, result.to_dict())
        for factoid in result.factoids:
            row = factoid.to_dict()
            row["response_id"] = result.response_id
            row["question_id"] = result.question_id
            row["response_metadata"] = dict(result.metadata)
            self._append(self.factoids_path, row)
            for observation in factoid.observations:
                call = observation.to_dict()
                call["response_id"] = result.response_id
                call["question_id"] = result.question_id
                call["factoid_id"] = factoid.factoid_id
                self._append(self.calls_path, call)

    def write_summary(self, summary: dict[str, Any]) -> None:
        atomic_write_json(self.run_directory / "summary.json", summary)
