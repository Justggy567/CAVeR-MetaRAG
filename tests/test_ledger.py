import json
import tempfile
import unittest
from pathlib import Path

from src.caver.cli import smoke_test
from src.caver.ledger import RunLedger


class LedgerTests(unittest.TestCase):
    def test_non_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            ledger = RunLedger(target)
            ledger.write_manifest({"run_id": "test"})
            with self.assertRaises(FileExistsError):
                RunLedger(target)
            with (target / "manifest.json").open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["run_id"], "test")

    def test_offline_smoke(self):
        result = smoke_test()
        self.assertEqual(result["verification_calls"], 10)
        self.assertTrue(result["probe_hash_reused"])


if __name__ == "__main__":
    unittest.main()
