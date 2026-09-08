import unittest

from src.caver.statistics import PairedPrediction, clustered_paired_difference, mcnemar_exact


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            PairedPrediction("q1", False, True, False),
            PairedPrediction("q1", True, False, True),
            PairedPrediction("q2", False, False, False),
            PairedPrediction("q3", True, True, True),
        ]

    def test_clustered_accuracy_difference(self):
        result = clustered_paired_difference(
            self.rows,
            metric="accuracy",
            repetitions=200,
            seed=7,
        )
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["responses"], 4)
        self.assertGreater(result["difference"], 0)

    def test_mcnemar_counts_discordant_pairs(self):
        result = mcnemar_exact(self.rows)
        self.assertEqual(result["first_wrong_second_correct"], 2)
        self.assertEqual(result["first_correct_second_wrong"], 0)
        self.assertEqual(result["discordant"], 2)


if __name__ == "__main__":
    unittest.main()
