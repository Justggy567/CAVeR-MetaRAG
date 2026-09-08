import unittest

from src.caver.baselines import random_matched_budget, random_matched_count


class BaselineTests(unittest.TestCase):
    def test_count_matching_is_seeded(self):
        first = random_matched_count(["a", "b", "c", "d"], 2, seed=42)
        second = random_matched_count(["a", "b", "c", "d"], 2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_budget_is_not_exceeded(self):
        costs = {"a": 5, "b": 7, "c": 11, "d": 13}
        selected = random_matched_budget(costs, 20, seed=7)
        self.assertLessEqual(sum(costs[item] for item in selected), 20)


if __name__ == "__main__":
    unittest.main()
