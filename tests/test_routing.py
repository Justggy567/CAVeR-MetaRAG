import unittest
from itertools import product

from src.caver.labels import Verdict
from src.caver.routing import RoutingPolicy, compute_signals, should_escalate
from src.caver.schema import VerificationBundle


def bundle(original, syn1, syn2, ant1, ant2):
    return VerificationBundle(original, (syn1, syn2), (ant1, ant2))


class RoutingTests(unittest.TestCase):
    def test_exhaustive_paper_formula(self):
        count = 0
        for values in product(Verdict, repeat=5):
            signals = compute_signals(bundle(*values))
            expected = (
                signals.original_uncertainty
                or signals.explicit_paradox
                or signals.antonym_anomaly_count >= 2
                or (signals.synonym_anomaly_count >= 1 and signals.antonym_anomaly_count >= 1)
            )
            self.assertEqual(should_escalate(signals, RoutingPolicy.STRUCTURE_AWARE), expected)
            count += 1
        self.assertEqual(count, 243)

    def test_original_uncertainty_escalates(self):
        signals = compute_signals(
            bundle(Verdict.NOT_SURE, Verdict.YES, Verdict.YES, Verdict.NO, Verdict.NO)
        )
        self.assertTrue(signals.original_uncertainty)
        self.assertTrue(should_escalate(signals))
        self.assertEqual(signals.synonym_anomaly_count, 0)
        self.assertEqual(signals.antonym_anomaly_count, 0)

    def test_isolated_synonym_anomaly_does_not_escalate(self):
        signals = compute_signals(
            bundle(Verdict.YES, Verdict.NOT_SURE, Verdict.YES, Verdict.NO, Verdict.NO)
        )
        self.assertEqual(signals.synonym_anomaly_count, 1)
        self.assertFalse(signals.explicit_paradox)
        self.assertFalse(should_escalate(signals))

    def test_single_non_paradox_antonym_anomaly_does_not_escalate(self):
        signals = compute_signals(
            bundle(Verdict.YES, Verdict.YES, Verdict.YES, Verdict.NOT_SURE, Verdict.NO)
        )
        self.assertEqual(signals.antonym_anomaly_count, 1)
        self.assertFalse(signals.explicit_paradox)
        self.assertFalse(should_escalate(signals))

    def test_two_antonym_anomalies_escalate(self):
        signals = compute_signals(
            bundle(Verdict.YES, Verdict.YES, Verdict.YES, Verdict.NOT_SURE, Verdict.NOT_SURE)
        )
        self.assertEqual(signals.antonym_anomaly_count, 2)
        self.assertTrue(should_escalate(signals))

    def test_cross_side_anomaly_escalates(self):
        signals = compute_signals(
            bundle(Verdict.YES, Verdict.NOT_SURE, Verdict.YES, Verdict.NOT_SURE, Verdict.NO)
        )
        self.assertTrue(signals.cross_side_anomaly)
        self.assertTrue(should_escalate(signals))

    def test_explicit_paradox_uses_deterministic_intersection(self):
        signals = compute_signals(
            bundle(Verdict.YES, Verdict.YES, Verdict.YES, Verdict.YES, Verdict.NO)
        )
        self.assertTrue(signals.explicit_paradox)
        self.assertTrue(should_escalate(signals))


if __name__ == "__main__":
    unittest.main()
