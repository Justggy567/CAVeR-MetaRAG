"""Validated immutable objects shared by every experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .labels import BinaryTarget, Verdict


def _require_nonempty(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


@dataclass(frozen=True)
class ProbeSet:
    factoid_id: str
    original: str
    synonyms: tuple[str, str]
    antonyms: tuple[str, str]
    probe_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "factoid_id", _require_nonempty(self.factoid_id, "factoid_id"))
        object.__setattr__(self, "original", _require_nonempty(self.original, "original"))
        if len(self.synonyms) != 2 or len(self.antonyms) != 2:
            raise ValueError("The canonical protocol requires exactly two synonyms and two antonyms.")
        object.__setattr__(
            self,
            "synonyms",
            tuple(_require_nonempty(value, f"synonyms[{index}]") for index, value in enumerate(self.synonyms)),
        )
        object.__setattr__(
            self,
            "antonyms",
            tuple(_require_nonempty(value, f"antonyms[{index}]") for index, value in enumerate(self.antonyms)),
        )

    def statements(self) -> tuple[tuple[str, str], ...]:
        return (
            ("original", self.original),
            ("synonym_1", self.synonyms[0]),
            ("synonym_2", self.synonyms[1]),
            ("antonym_1", self.antonyms[0]),
            ("antonym_2", self.antonyms[1]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProbeSet":
        synonyms = tuple(data.get("synonyms", ()))
        antonyms = tuple(data.get("antonyms", ()))
        return cls(
            factoid_id=str(data["factoid_id"]),
            original=str(data["original"]),
            synonyms=synonyms,
            antonyms=antonyms,
            probe_hash=str(data.get("probe_hash", "")),
        )


@dataclass(frozen=True)
class VerificationBundle:
    original: Verdict
    synonyms: tuple[Verdict, Verdict]
    antonyms: tuple[Verdict, Verdict]

    def __post_init__(self) -> None:
        if len(self.synonyms) != 2 or len(self.antonyms) != 2:
            raise ValueError("A verification bundle must contain 1 original, 2 synonyms, and 2 antonyms.")

    def verdicts(self) -> tuple[Verdict, ...]:
        return (self.original, *self.synonyms, *self.antonyms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original.value,
            "synonyms": [item.value for item in self.synonyms],
            "antonyms": [item.value for item in self.antonyms],
        }


@dataclass(frozen=True)
class TriggerSignals:
    original_uncertainty: bool
    explicit_paradox: bool
    synonym_anomaly_count: int
    antonym_anomaly_count: int
    two_antonym_anomalies: bool
    cross_side_anomaly: bool

    def active_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.original_uncertainty:
            names.append("original_uncertainty")
        if self.explicit_paradox:
            names.append("explicit_paradox")
        if self.two_antonym_anomalies:
            names.append("two_antonym_anomalies")
        if self.cross_side_anomaly:
            names.append("cross_side_anomaly")
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 1
    charge: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.calls) < 0:
            raise ValueError("Token and call counts cannot be negative.")


@dataclass(frozen=True)
class VerificationObservation:
    call_id: str
    model_id: str
    backend: str
    phase: str
    statement_type: str
    statement: str
    verdict: Verdict
    raw_response: str
    latency_seconds: float
    usage: CallUsage = field(default_factory=CallUsage)
    request_hash: str = ""
    response_hash: str = ""
    cache_hit: bool = False
    provider_response_id: str = ""
    returned_model_id: str = ""
    system_fingerprint: str = ""
    finish_reason: str = ""
    attempts: int = 1
    request_parameters: Mapping[str, Any] = field(default_factory=dict)
    usage_details: Mapping[str, Any] = field(default_factory=dict)
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "success"
    prompt_bundle_version: str = ""
    system_prompt_hash: str = ""
    user_prompt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["usage"]["total_tokens"] = self.usage.total_tokens
        return data


@dataclass(frozen=True)
class FactoidResult:
    factoid_id: str
    probe_hash: str
    stage1: VerificationBundle
    signals: TriggerSignals
    routed: bool
    route_reasons: tuple[str, ...]
    stage1_score: float
    final_bundle: VerificationBundle
    final_score: float
    routing_latency_seconds: float = 0.0
    stage2: VerificationBundle | None = None
    observations: tuple[VerificationObservation, ...] = ()
    probes: ProbeSet | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factoid_id": self.factoid_id,
            "probe_hash": self.probe_hash,
            "stage1": self.stage1.to_dict(),
            "signals": self.signals.to_dict(),
            "routed": self.routed,
            "route_reasons": list(self.route_reasons),
            "stage1_score": self.stage1_score,
            "stage2": self.stage2.to_dict() if self.stage2 else None,
            "final_bundle": self.final_bundle.to_dict(),
            "final_score": self.final_score,
            "routing_latency_seconds": self.routing_latency_seconds,
            "probes": self.probes.to_dict() if self.probes else None,
        }


@dataclass(frozen=True)
class ResponseResult:
    response_id: str
    question_id: str
    gold_target: BinaryTarget
    factoids: tuple[FactoidResult, ...]
    stage1_score: float
    final_score: float
    stage1_hallucinated: bool
    final_hallucinated: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.factoids:
            raise ValueError("A response result must contain at least one factoid.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "question_id": self.question_id,
            "gold_target": self.gold_target.value,
            "factoid_ids": [item.factoid_id for item in self.factoids],
            "stage1_score": self.stage1_score,
            "final_score": self.final_score,
            "stage1_hallucinated": self.stage1_hallucinated,
            "final_hallucinated": self.final_hallucinated,
            "metadata": dict(self.metadata),
        }
