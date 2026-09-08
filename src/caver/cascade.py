"""Canonical five-verdict Stage-1, router, and Stage-2 orchestration."""

from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any, Mapping

from .labels import BinaryTarget
from .routing import RoutingPolicy, compute_signals, routing_reasons, should_escalate
from .schema import (
    FactoidResult,
    ProbeSet,
    ResponseResult,
    VerificationBundle,
    VerificationObservation,
)
from .scoring import DEFAULT_THRESHOLD, aggregate_response_scores, predicts_hallucination, score_verification
from .verifiers import EvidenceVerifier


class CanonicalCascade:
    def __init__(
        self,
        *,
        stage1_verifier: EvidenceVerifier,
        stage2_verifier: EvidenceVerifier | None = None,
        routing_policy: RoutingPolicy = RoutingPolicy.STRUCTURE_AWARE,
        threshold: float = DEFAULT_THRESHOLD,
        anomaly_count_threshold: int = 2,
    ):
        self.stage1_verifier = stage1_verifier
        self.stage2_verifier = stage2_verifier
        self.routing_policy = routing_policy
        self.threshold = threshold
        self.anomaly_count_threshold = anomaly_count_threshold

    @staticmethod
    def _bundle_from_observations(
        observations: Sequence[VerificationObservation],
    ) -> VerificationBundle:
        if len(observations) != 5:
            raise ValueError("The canonical protocol requires five verification observations.")
        return VerificationBundle(
            original=observations[0].verdict,
            synonyms=(observations[1].verdict, observations[2].verdict),
            antonyms=(observations[3].verdict, observations[4].verdict),
        )

    @staticmethod
    def _verify_probe_set(
        verifier: EvidenceVerifier,
        probes: ProbeSet,
        context: str,
        phase: str,
    ) -> tuple[VerificationBundle, tuple[VerificationObservation, ...]]:
        observations = tuple(
            verifier.verify(
                statement=statement,
                context=context,
                phase=phase,
                statement_type=statement_type,
            )
            for statement_type, statement in probes.statements()
        )
        return CanonicalCascade._bundle_from_observations(observations), observations

    def process_factoid(self, *, probes: ProbeSet, context: str) -> FactoidResult:
        stage1, stage1_observations = self._verify_probe_set(
            self.stage1_verifier,
            probes,
            context,
            "stage1_verify",
        )
        routing_started = time.perf_counter()
        signals = compute_signals(stage1)
        routed = should_escalate(
            signals,
            self.routing_policy,
            anomaly_count_threshold=self.anomaly_count_threshold,
        )
        route_reasons = routing_reasons(
            signals,
            self.routing_policy,
            anomaly_count_threshold=self.anomaly_count_threshold,
        )
        routing_latency_seconds = time.perf_counter() - routing_started
        stage1_score = score_verification(stage1)
        stage2 = None
        stage2_observations: tuple[VerificationObservation, ...] = ()
        final_bundle = stage1
        if routed and self.stage2_verifier is not None:
            stage2, stage2_observations = self._verify_probe_set(
                self.stage2_verifier,
                probes,
                context,
                "stage2_verify",
            )
            final_bundle = stage2
        return FactoidResult(
            factoid_id=probes.factoid_id,
            probe_hash=probes.probe_hash,
            stage1=stage1,
            signals=signals,
            routed=routed,
            route_reasons=route_reasons,
            stage1_score=stage1_score,
            stage2=stage2,
            final_bundle=final_bundle,
            final_score=score_verification(final_bundle),
            routing_latency_seconds=routing_latency_seconds,
            observations=stage1_observations + stage2_observations,
            probes=probes,
        )

    def process_response(
        self,
        *,
        response_id: str,
        question_id: str,
        gold_target: BinaryTarget,
        context: str,
        probe_sets: Sequence[ProbeSet],
        metadata: Mapping[str, Any] | None = None,
    ) -> ResponseResult:
        factoids = tuple(self.process_factoid(probes=probes, context=context) for probes in probe_sets)
        stage1_score = aggregate_response_scores(item.stage1_score for item in factoids)
        final_score = aggregate_response_scores(item.final_score for item in factoids)
        return ResponseResult(
            response_id=str(response_id),
            question_id=str(question_id),
            gold_target=gold_target,
            factoids=factoids,
            stage1_score=stage1_score,
            final_score=final_score,
            stage1_hallucinated=predicts_hallucination(stage1_score, self.threshold),
            final_hallucinated=predicts_hallucination(final_score, self.threshold),
            metadata=dict(metadata or {}),
        )
