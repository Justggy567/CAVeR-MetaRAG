import json
import tempfile
import unittest
from pathlib import Path

from src.caver.datasets import load_source_responses
from src.caver.configuration import validate_run_config
from src.caver.prompts import prompt_manifest
from src.caver.taxonomy import taxonomy_manifest
from src.caver.routing import RoutingPolicy, compute_signals, should_escalate
from src.caver.schema import VerificationBundle
from src.caver.labels import Verdict


class RevisionProtocolTests(unittest.TestCase):
    def test_formal_model_config_rejects_hidden_decoding_defaults(self):
        config = {
            "prepared_path": "prepared.jsonl",
            "output_root": "runs",
            "routing_policy": "no_escalation",
            "scoring_profile": "paper_five_statement_mean",
            "response_aggregation": "max_factoid_score",
            "stage1": {"backend": "ollama", "model_id": "gemma3:4b"},
            "stage2": None,
            "environment": {"concurrency": 1, "execution": "sequential"},
        }
        with self.assertRaisesRegex(ValueError, "reproducibility parameters missing"):
            validate_run_config(config)

    def test_prompt_bundle_is_character_frozen(self):
        manifest = prompt_manifest()
        self.assertEqual(manifest["version"], "2026-08-22.v1")
        self.assertEqual(
            manifest["bundle_hash"],
            "bf91e48cc31a23b82815c8fed18870e75cff6f8589656c5f1b7ea445a9fe3766",
        )

    def test_revised_rq3_taxonomy_is_frozen(self):
        manifest = taxonomy_manifest()
        self.assertEqual(manifest["version"], "2026-08-22.user-v2")
        self.assertEqual(
            manifest["manifest_sha256"],
            "7f3b4ee6a7c27bbc3640e210f97b54cc860eca81c7163fdac34c774040a72ad9",
        )
        self.assertEqual(
            manifest["category_names"]["Cat1"],
            "Entity / Alias / Lexical Relation",
        )

    def test_rq3_jsonl_uses_existing_factoid_without_decomposition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rq3.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "MAIN_0001",
                        "source_id": "q1",
                        "question": "Where is Paris?",
                        "context": "Paris is in France.",
                        "factoid": "Paris is in France.",
                        "category": "Cat1",
                        "context_support": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_source_responses(path, dataset_format="factoids")
            self.assertEqual(rows[0].source_factoids, ("Paris is in France.",))
            self.assertEqual(rows[0].question_id, "q1")
            self.assertEqual(
                rows[0].metadata["factoid_type"], "Entity / Alias / Lexical Relation"
            )
            self.assertEqual(rows[0].metadata["factoid_category_code"], "Cat1")

    def test_three_manuscript_policies_are_distinct(self):
        isolated_synonym = VerificationBundle(
            Verdict.YES,
            (Verdict.NOT_SURE, Verdict.YES),
            (Verdict.NO, Verdict.NO),
        )
        signals = compute_signals(isolated_synonym)
        self.assertTrue(should_escalate(signals, RoutingPolicy.STRICT))
        self.assertFalse(should_escalate(signals, RoutingPolicy.PARADOX_ONLY))
        self.assertFalse(should_escalate(signals, RoutingPolicy.STRUCTURE_AWARE))

    def test_rq3_style_policy_requires_stage2(self):
        config = {
            "prepared_path": "prepared.jsonl",
            "output_root": "results",
            "routing_policy": "structure_aware",
            "scoring_profile": "paper_five_statement_mean",
            "response_aggregation": "max_factoid_score",
            "stage1": {"backend": "static", "model_id": "small", "responses": []},
            "stage2": None,
        }
        with self.assertRaises(ValueError):
            validate_run_config(config)


if __name__ == "__main__":
    unittest.main()
