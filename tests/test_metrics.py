import unittest

from src.caver.labels import BinaryTarget, Verdict
from src.caver.metrics import TransitionCounts, summarize_results
from src.caver.routing import compute_signals
from src.caver.schema import FactoidResult, ResponseResult, VerificationBundle


def make_result(response_id, gold, stage1_hallucinated, final_hallucinated, routed):
    consistent = VerificationBundle(
        Verdict.YES,
        (Verdict.YES, Verdict.YES),
        (Verdict.NO, Verdict.NO),
    )
    factoid = FactoidResult(
        factoid_id=response_id + ":f0",
        probe_hash="hash",
        stage1=consistent,
        signals=compute_signals(consistent),
        routed=routed,
        route_reasons=("test",) if routed else (),
        stage1_score=0.0,
        final_bundle=consistent,
        final_score=0.0,
    )
    return ResponseResult(
        response_id=response_id,
        question_id=response_id,
        gold_target=gold,
        factoids=(factoid,),
        stage1_score=0.0,
        final_score=0.0,
        stage1_hallucinated=stage1_hallucinated,
        final_hallucinated=final_hallucinated,
    )


class MetricsTests(unittest.TestCase):
    def test_integer_counts_and_transition_invariant(self):
        results = [
            make_result("rescued", BinaryTarget.SUPPORTED, True, False, True),
            make_result("harmed", BinaryTarget.SUPPORTED, False, True, True),
            make_result("correct", BinaryTarget.UNSUPPORTED, True, True, False),
            make_result("wrong", BinaryTarget.UNSUPPORTED, False, False, False),
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["transitions"]["rescued"], 1)
        self.assertEqual(summary["transitions"]["harmed"], 1)
        self.assertEqual(summary["transitions"]["net_correct_gain"], 0)
        self.assertEqual(summary["transitions"]["rescue_rate_response"], 0.5)
        self.assertEqual(summary["transitions"]["harm_rate_response"], 0.5)
        self.assertEqual(summary["transitions"]["error_conditional_rescue_rate"], 1.0)
        self.assertEqual(summary["transitions"]["rescue_rate_all_responses"], 0.25)
        self.assertEqual(summary["stage1"]["n"], 4)
        self.assertEqual(summary["final"]["n"], 4)

    def test_zero_denominator_is_unavailable_not_zero_percent(self):
        metrics = TransitionCounts(unchanged_correct=2).metrics()
        self.assertIsNone(metrics["rescue_rate_response"])
        self.assertIsNone(metrics["harm_rate_response"])
        self.assertIsNone(metrics["error_conditional_rescue_rate"])
        self.assertEqual(metrics["rescue_rate_all_responses"], 0.0)
        self.assertIn("rescue_rate_response", metrics["undefined_rates"])


if __name__ == "__main__":
    unittest.main()
