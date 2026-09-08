import unittest

from src.caver.labels import Verdict
from src.caver.schema import VerificationBundle
from src.caver.scoring import aggregate_response_scores, predicts_hallucination, score_verification


class ScoringTests(unittest.TestCase):
    def test_fully_consistent_supported_bundle_scores_zero(self):
        item = VerificationBundle(
            Verdict.YES,
            (Verdict.YES, Verdict.YES),
            (Verdict.NO, Verdict.NO),
        )
        self.assertEqual(score_verification(item), 0.0)

    def test_fully_reversed_bundle_scores_one(self):
        item = VerificationBundle(
            Verdict.NO,
            (Verdict.NO, Verdict.NO),
            (Verdict.YES, Verdict.YES),
        )
        self.assertEqual(score_verification(item), 1.0)

    def test_response_uses_max_factoid_score(self):
        self.assertEqual(aggregate_response_scores([0.1, 0.7, 0.3]), 0.7)
        self.assertTrue(predicts_hallucination(0.5))


if __name__ == "__main__":
    unittest.main()
