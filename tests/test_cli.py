import unittest

from src.caver.cli import smoke_test, validate_method


class CliTests(unittest.TestCase):
    def test_method_validation(self):
        result = validate_method()
        self.assertEqual(result["combinations_checked"], 243)
        self.assertTrue(result["paper_formula_verified"])

    def test_smoke(self):
        self.assertEqual(smoke_test()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
