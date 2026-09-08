

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CANONICAL_FIELDS = (
    "id",
    "question",
    "context",
    "right_answer",
    "hallucinated_answer",
)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} Number {line_number} Movement is not legal JSON：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} Number {line_number}  Must beJSON object。")
            records.append(item)
        return records

    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} Must be JSON list or JSONL。")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True).strip()
    return str(value).strip()


def normalize_records(raw_records: List[Dict[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    
    normalized = []
    seen_ids = set()
    for index, raw in enumerate(raw_records, 1):
        if dataset == "asqa":
            item = {
                "id": raw.get("id"),
                "question": _text(raw.get("question")),
                "context": _text(raw.get("docs_context")),
                "right_answer": _text(raw.get("standard_answer")),
                "hallucinated_answer": "",
            }
        elif dataset == "haluevalqa":
            item = {
                "id": f"haluevalqa-{index:05d}",
                "question": _text(raw.get("question")),
                "context": _text(raw.get("knowledge")),
                "right_answer": _text(raw.get("right_answer")),
                "hallucinated_answer": _text(raw.get("hallucinated_answer")),
            }
        else:
            raise ValueError(f"Unknown dataset:{dataset}")

        errors = validate_item(item, require_hallucination=(dataset == "haluevalqa"))
        if errors:
            raise ValueError(f"Number {index} The record is invalid:{errors}")
        key = str(item["id"])
        if key in seen_ids:
            raise ValueError(f"Duplicate detected id：{key}")
        seen_ids.add(key)
        normalized.append(item)
    return normalized


def validate_item(item: Dict[str, Any], require_hallucination: bool) -> List[str]:
    errors = []
    if tuple(item.keys()) != CANONICAL_FIELDS:
        errors.append(f"The field must be {CANONICAL_FIELDS}")
    for field in ("id", "question", "context", "right_answer"):
        if item.get(field) in (None, ""):
            errors.append(f"{field} Empty")
    if require_hallucination and not item.get("hallucinated_answer"):
        errors.append("hallucinated_answer Empty")
    return errors


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} Number {line_number} Row damage:{exc}") from exc
    return records


def latest_success_by_id(
    ledger_path: Path,
    stage: str,
    prompt_hash: str,
) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(ledger_path):
        if (
            record.get("stage") == stage
            and record.get("status") == "success"
            and record.get("prompt_hash") == prompt_hash
        ):
            latest[str(record.get("id"))] = record
    return latest

