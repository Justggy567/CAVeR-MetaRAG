import tempfile
import unittest
from pathlib import Path

from src.caver.cache import VerificationCache
from src.caver.clients import StaticSequenceClient
from src.caver.verifiers import EvidenceVerifier


class VerificationCacheTests(unittest.TestCase):
    def test_second_identical_request_is_a_cache_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verifier.jsonl"
            client = StaticSequenceClient(["YES"], model_id="cached-model")
            verifier = EvidenceVerifier(client, cache=VerificationCache(path).load())
            first = verifier.verify(
                statement="Paris is in France.",
                context="Paris is in France.",
                phase="stage1_verify",
                statement_type="original",
            )
            second = verifier.verify(
                statement="Paris is in France.",
                context="Paris is in France.",
                phase="stage2_verify",
                statement_type="original",
            )
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.request_hash, second.request_hash)
            self.assertEqual(first.verdict, second.verdict)
            reloaded = VerificationCache(path).load()
            self.assertEqual(len(reloaded), 1)


if __name__ == "__main__":
    unittest.main()
