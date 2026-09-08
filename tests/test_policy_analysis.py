import json
import tempfile
import unittest
from pathlib import Path

from src.caver.datasets import PreparedResponse, save_prepared_responses
from src.caver.experiment import run_from_config
from src.caver.labels import BinaryTarget
from src.caver.policy_analysis import _pricing_charge, run_policy_suite
from src.caver.probes import make_probe_set


class PolicyAnalysisTests(unittest.TestCase):
    def test_deepseek_price_uses_provider_cache_hit_and_miss_tokens(self):
        call = {
            "model_id": "deepseek-v4-flash",
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "usage_details": {
                "prompt_cache_hit_tokens": 750_000,
                "prompt_cache_miss_tokens": 250_000,
            },
        }
        pricing = {
            "deepseek-v4-flash": {
                "cached_input_per_million": 0.1,
                "uncached_input_per_million": 3,
                "output_per_million": 9,
            }
        }
        self.assertAlmostEqual(_pricing_charge(call, pricing), 1.725)

    def test_three_policies_replay_one_full_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "method_spec.md").write_text("frozen", encoding="utf-8")
            probes1 = make_probe_set(
                "r1:f0",
                "Paris is in France.",
                ("Paris lies in France.", "France contains Paris."),
                ("Paris is not in France.", "France excludes Paris."),
            )
            probes2 = make_probe_set(
                "r2:f0",
                "Rome is in Germany.",
                ("Rome lies in Germany.", "Germany contains Rome."),
                ("Rome is not in Germany.", "Germany excludes Rome."),
            )
            prepared = [
                PreparedResponse(
                    response_id="r1",
                    question_id="q1",
                    answer_pair_id="q1",
                    answer_role="right",
                    question="Where is Paris?",
                    context="Paris is in France.",
                    answer="Paris is in France.",
                    gold_target=BinaryTarget.SUPPORTED,
                    probe_sets=(probes1,),
                ),
                PreparedResponse(
                    response_id="r2",
                    question_id="q2",
                    answer_pair_id="q2",
                    answer_role="hallucinated",
                    question="Where is Rome?",
                    context="Rome is in Italy.",
                    answer="Rome is in Germany.",
                    gold_target=BinaryTarget.UNSUPPORTED,
                    probe_sets=(probes2,),
                ),
            ]
            save_prepared_responses(root / "prepared.jsonl", prepared)
            stage1_first = ["NO", "NO", "NO", "YES", "NO"]
            stage1_second = ["NO", "NO", "NO", "YES", "YES"]
            stage2_first = ["YES", "YES", "YES", "NO", "NO"]
            stage2_second = ["NO", "NO", "NO", "YES", "YES"]
            matrix_config = {
                "prepared_path": "prepared.jsonl",
                "output_root": "matrix",
                "run_id": "primary",
                "routing_policy": "full_escalation",
                "threshold": 0.5,
                "scoring_profile": "paper_five_statement_mean",
                "response_aggregation": "max_factoid_score",
                "stage1": {
                    "backend": "static",
                    "model_id": "small",
                    "responses": stage1_first + stage1_second,
                },
                "stage2": {
                    "backend": "static",
                    "model_id": "strong",
                    "responses": stage2_first + stage2_second,
                },
            }
            run_from_config(matrix_config, repository_root=root)
            policy_config = {
                "matrix_run_directory": "matrix/primary",
                "output_path": "analysis.json",
                "policies": ["strict", "paradox_only", "structure_aware"],
                "include_routing_controls": False,
                "bootstrap_repetitions": 100,
                "bootstrap_seed": 7,
                "include_response_rows": False,
            }
            output = run_policy_suite(policy_config, repository_root=root)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["policies"]["structure_aware"]["final"]["accuracy"], 1.0)
            self.assertEqual(
                report["policies"]["structure_aware"]["routing"]["routed_factoids"], 1
            )
            structure = report["policies"]["structure_aware"]
            self.assertEqual(structure["routing"]["response_escalation_rate"], 0.5)
            self.assertEqual(structure["routing"]["factoid_escalation_rate"], 0.5)
            self.assertEqual(structure["transitions"]["rescued"], 1)
            self.assertEqual(structure["transitions"]["routed_stage1_wrong"], 1)
            self.assertEqual(structure["transitions"]["rescue_rate_response"], 1.0)
            self.assertEqual(
                structure["transitions"]["error_conditional_rescue_rate"], 1.0
            )
            self.assertEqual(structure["transitions"]["harm_rate_response"], 0.0)
            self.assertEqual(report["policies"]["paradox_only"]["routing"]["routed_factoids"], 1)
            self.assertEqual(report["learned_router"]["status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
