import unittest

from src.caver.cascade import CanonicalCascade
from src.caver.clients import StaticSequenceClient
from src.caver.labels import BinaryTarget
from src.caver.probes import make_probe_set
from src.caver.routing import RoutingPolicy
from src.caver.verifiers import EvidenceVerifier


class CascadeTests(unittest.TestCase):
    def test_stage2_reuses_all_five_frozen_statements(self):
        probes = make_probe_set(
            "r1:f0",
            "Paris is in France.",
            ("Paris lies in France.", "France contains Paris."),
            ("Paris is not in France.", "France excludes Paris."),
        )
        stage1 = EvidenceVerifier(
            StaticSequenceClient(["NO", "YES", "NO", "YES", "NO"], "small")
        )
        stage2 = EvidenceVerifier(
            StaticSequenceClient(["YES", "YES", "YES", "NO", "NO"], "strong")
        )
        engine = CanonicalCascade(
            stage1_verifier=stage1,
            stage2_verifier=stage2,
            routing_policy=RoutingPolicy.STRUCTURE_AWARE,
        )
        result = engine.process_response(
            response_id="r1",
            question_id="q1",
            gold_target=BinaryTarget.SUPPORTED,
            context="Paris is the capital of France.",
            probe_sets=(probes,),
        )
        factoid = result.factoids[0]
        self.assertTrue(factoid.routed)
        self.assertEqual(len(factoid.observations), 10)
        stage1_texts = [item.statement for item in factoid.observations[:5]]
        stage2_texts = [item.statement for item in factoid.observations[5:]]
        self.assertEqual(stage1_texts, stage2_texts)
        self.assertEqual(stage1_texts, [text for _, text in probes.statements()])
        self.assertTrue(result.stage1_hallucinated)
        self.assertFalse(result.final_hallucinated)

    def test_no_escalation_uses_only_stage1(self):
        probes = make_probe_set(
            "r2:f0",
            "Paris is in France.",
            ("Paris lies in France.", "France contains Paris."),
            ("Paris is not in France.", "France excludes Paris."),
        )
        stage1 = EvidenceVerifier(
            StaticSequenceClient(["YES", "YES", "YES", "NO", "NO"], "single")
        )
        engine = CanonicalCascade(
            stage1_verifier=stage1,
            routing_policy=RoutingPolicy.NO_ESCALATION,
        )
        result = engine.process_response(
            response_id="r2",
            question_id="q2",
            gold_target=BinaryTarget.SUPPORTED,
            context="Paris is the capital of France.",
            probe_sets=(probes,),
        )
        self.assertEqual(len(result.factoids[0].observations), 5)
        self.assertIsNone(result.factoids[0].stage2)


if __name__ == "__main__":
    unittest.main()
