"""Probe construction, hashing, validation, and frozen-cache I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import ProbeSet


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_probe_set(
    factoid_id: str,
    original: str,
    synonyms: Iterable[str],
    antonyms: Iterable[str],
) -> ProbeSet:
    syn = tuple(str(item).strip() for item in synonyms)
    ant = tuple(str(item).strip() for item in antonyms)
    payload = {
        "factoid_id": str(factoid_id),
        "original": str(original).strip(),
        "synonyms": list(syn),
        "antonyms": list(ant),
    }
    return ProbeSet(
        factoid_id=str(factoid_id),
        original=str(original),
        synonyms=syn,
        antonyms=ant,
        probe_hash=stable_hash(payload),
    )


def probe_warnings(probes: ProbeSet) -> tuple[str, ...]:
    warnings: list[str] = []
    normalized_original = " ".join(probes.original.lower().split())
    normalized_synonyms = [" ".join(item.lower().split()) for item in probes.synonyms]
    normalized_antonyms = [" ".join(item.lower().split()) for item in probes.antonyms]
    if len(set(normalized_synonyms)) != 2:
        warnings.append("duplicate_synonyms")
    if len(set(normalized_antonyms)) != 2:
        warnings.append("duplicate_antonyms")
    if normalized_original in normalized_synonyms:
        warnings.append("synonym_identical_to_original")
    if set(normalized_synonyms).intersection(normalized_antonyms):
        warnings.append("same_text_on_positive_and_negative_sides")
    return tuple(warnings)


class ProbeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._items: dict[str, ProbeSet] = {}

    def load(self) -> "ProbeCache":
        self._items.clear()
        if not self.path.exists():
            return self
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                data = json.loads(line)
                probes = ProbeSet.from_mapping(data)
                expected = make_probe_set(
                    probes.factoid_id,
                    probes.original,
                    probes.synonyms,
                    probes.antonyms,
                ).probe_hash
                if probes.probe_hash != expected:
                    raise ValueError(f"Probe hash mismatch at {self.path}:{line_number}")
                if probes.factoid_id in self._items:
                    raise ValueError(f"Duplicate factoid_id in probe cache: {probes.factoid_id}")
                self._items[probes.factoid_id] = probes
        return self

    def get(self, factoid_id: str) -> ProbeSet | None:
        return self._items.get(str(factoid_id))

    def add(self, probes: ProbeSet) -> None:
        existing = self._items.get(probes.factoid_id)
        if existing and existing.probe_hash != probes.probe_hash:
            raise ValueError(f"Refusing to overwrite frozen probes for {probes.factoid_id}")
        self._items[probes.factoid_id] = probes

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for factoid_id in sorted(self._items):
                handle.write(json.dumps(self._items[factoid_id].to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        temporary.replace(self.path)

    def __len__(self) -> int:
        return len(self._items)
