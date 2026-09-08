"""Dataset normalization and frozen prepared-response schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .labels import BinaryTarget
from .probes import make_probe_set
from .schema import ProbeSet
from .taxonomy import RQ3_CATEGORY_NAMES


@dataclass(frozen=True)
class SourceResponse:
    response_id: str
    question_id: str
    answer_pair_id: str
    answer_role: str
    question: str
    context: str
    answer: str
    gold_target: BinaryTarget
    source_factoids: tuple[str, ...] = ()
    frozen_probes: tuple[ProbeSet, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedResponse:
    response_id: str
    question_id: str
    answer_pair_id: str
    answer_role: str
    question: str
    context: str
    answer: str
    gold_target: BinaryTarget
    probe_sets: tuple[ProbeSet, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "question_id": self.question_id,
            "answer_pair_id": self.answer_pair_id,
            "answer_role": self.answer_role,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "gold_target": self.gold_target.value,
            "probe_sets": [item.to_dict() for item in self.probe_sets],
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PreparedResponse":
        probe_sets = tuple(ProbeSet.from_mapping(item) for item in data["probe_sets"])
        return cls(
            response_id=str(data["response_id"]),
            question_id=str(data["question_id"]),
            answer_pair_id=str(data.get("answer_pair_id", data["question_id"])),
            answer_role=str(data["answer_role"]),
            question=str(data["question"]),
            context=str(data["context"]),
            answer=str(data.get("answer", "")),
            gold_target=BinaryTarget(str(data["gold_target"])),
            probe_sets=probe_sets,
            metadata=dict(data.get("metadata", {})),
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL, and provide clear errors for line numbers."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
                if not isinstance(item, dict):
                    raise TypeError(f"JSONL line {line_number} must be an object.")
                records.append(item)
        return records
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON dataset root must be an array.")
    if any(not isinstance(item, dict) for item in data):
        raise TypeError("Every JSON dataset record must be an object.")
    return data


def load_source_responses(
    path: str | Path,
    *,
    dataset_format: str,
    answer_roles: str = "both",
    dataset_name: str | None = None,
    limit: int | None = None,
) -> list[SourceResponse]:
    data = _load_records(path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        data = data[:limit]
    responses: list[SourceResponse] = []

    if dataset_format == "paired_answers":
        if answer_roles not in {"both", "right", "hallucinated"}:
            raise ValueError("answer_roles must be both, right, or hallucinated.")
        for index, item in enumerate(data):
            pair_id = str(item.get("id", index))
            question = str(item.get("question", "")).strip()
            context = str(item.get("context", "")).strip()
            right_answer = str(item.get("right_answer", "")).strip()
            hallucinated_answer = str(item.get("hallucinated_answer", "")).strip()
            if not all((question, context, right_answer, hallucinated_answer)):
                raise ValueError(f"Paired-answer record {pair_id} has an empty canonical field.")
            common = {
                "question_id": pair_id,
                "answer_pair_id": pair_id,
                "question": question,
                "context": context,
                "metadata": {
                    "source_row_id": pair_id,
                    "source_dataset": dataset_name or Path(path).stem,
                },
            }
            if answer_roles in {"both", "right"}:
                responses.append(
                    SourceResponse(
                        response_id=pair_id + ":right",
                        answer_role="right",
                        answer=right_answer,
                        gold_target=BinaryTarget.SUPPORTED,
                        **common,
                    )
                )
            if answer_roles in {"both", "hallucinated"}:
                responses.append(
                    SourceResponse(
                        response_id=pair_id + ":hallucinated",
                        answer_role="hallucinated",
                        answer=hallucinated_answer,
                        gold_target=BinaryTarget.UNSUPPORTED,
                        **common,
                    )
                )
    elif dataset_format == "factoids":
        for index, item in enumerate(data):
            item_id = str(item.get("id", index))
            supported = item.get("context_support")
            if not isinstance(supported, bool):
                raise TypeError(f"context_support must be boolean for factoid {item_id}.")
            factoid = str(item.get("factoid", "")).strip()
            question = str(item.get("question", "")).strip()
            context = str(item.get("context", "")).strip()
            if not all((factoid, question, context)):
                raise ValueError(f"Factoid record {item_id} has an empty canonical field.")
            frozen: tuple[ProbeSet, ...] = ()
            mutations = item.get("mutations_list")
            if isinstance(mutations, dict):
                frozen = (
                    make_probe_set(
                        item_id + ":f0",
                        factoid,
                        mutations.get("synonyms", ()),
                        mutations.get("antonyms", ()),
                    ),
                )
            responses.append(
                SourceResponse(
                    response_id=item_id,
                    question_id=str(item.get("question_id", item.get("source_id", item_id))),
                    answer_pair_id=str(item.get("answer_pair_id", item.get("source_id", item_id))),
                    answer_role=str(item.get("answer_type", "factoid")),
                    question=question,
                    context=context,
                    answer=str(item.get("source_answer", factoid)),
                    gold_target=BinaryTarget.SUPPORTED if supported else BinaryTarget.UNSUPPORTED,
                    source_factoids=(factoid,),
                    frozen_probes=frozen,
                    metadata={
                        "source_row_id": item_id,
                        "factoid_category_code": item.get("category"),
                        "factoid_type": item.get("factoid_type")
                        or RQ3_CATEGORY_NAMES.get(
                            str(item.get("category")), str(item.get("category"))
                        ),
                        "answer_type": item.get("answer_type"),
                        "source_answer_present": bool(item.get("source_answer")),
                        "source_dataset": item.get("source_dataset", dataset_name or Path(path).stem),
                        "source_id": item.get("source_id"),
                    },
                )
            )
    else:
        raise ValueError(f"Unsupported dataset_format: {dataset_format}")

    ids = [item.response_id for item in responses]
    if len(ids) != len(set(ids)):
        raise ValueError("Normalized response_id values are not unique.")
    return responses


def save_prepared_responses(path: str | Path, responses: Iterable[PreparedResponse]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for response in responses:
            handle.write(json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(target)


def load_prepared_responses(path: str | Path) -> list[PreparedResponse]:
    responses: list[PreparedResponse] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                responses.append(PreparedResponse.from_mapping(json.loads(line)))
            except Exception as error:
                raise ValueError(f"Invalid prepared record at line {line_number}: {error}") from error
    if not responses:
        raise ValueError("Prepared dataset is empty.")
    ids = [item.response_id for item in responses]
    if len(ids) != len(set(ids)):
        raise ValueError("Prepared response_id values are not unique.")
    return responses
