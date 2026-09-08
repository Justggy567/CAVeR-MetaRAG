import json
import tempfile
import unittest
from pathlib import Path

from src.caver.datasets import PreparedResponse, load_prepared_responses, save_prepared_responses
from src.caver.experiment import prepare_from_config, run_from_config
from src.caver.labels import BinaryTarget
from src.caver.probes import make_probe_set


class ExperimentTests(unittest.TestCase):
    def test_offline_preparation_freezes_factoids_and_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = [
                {
                    "id": 1,
                    "question": "Where is Paris?",
                    "context": "Paris is in France.",
                    "right_answer": "Paris is in France.",
                    "hallucinated_answer": "Paris is in Germany.",
                }
            ]
            (root / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
            config = {
                "dataset": {
                    "path": "data.json",
                    "format": "paired_answers",
                    "answer_roles": "right",
                },
                "prepared_path": "prepared.jsonl",
                "preparation_model": {
                    "backend": "static",
                    "model_id": "offline-generator",
                    "responses": [
                        "[\"Paris is in France.\"]",
                        "Paris lies in France.\nFrance contains Paris.",
                        "Paris is not in France.",
                        "Paris is not in France.\nFrance excludes Paris.",
                    ],
                },
                "decomposition_format_max_attempts": 3,
                "decomposition_failure_policy": "record_and_continue",
                "mutation_temperature": 0.2,
                "mutation_format_max_attempts": 3,
                "mutation_failure_policy": "record_and_continue",
                "resume": True,
            }
            prepared_path = prepare_from_config(config, repository_root=root)
            prepared = load_prepared_responses(prepared_path)
            self.assertEqual(len(prepared), 1)
            self.assertEqual(len(prepared[0].probe_sets), 1)
            self.assertTrue(prepared[0].probe_sets[0].probe_hash)
            call_lines = prepared_path.with_suffix(".jsonl.calls.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(call_lines), 4)
            calls = [json.loads(line) for line in call_lines]
            mutation_calls = [row for row in calls if row["phase"].startswith("mutation_")]
            self.assertEqual(
                [row["format_valid"] for row in mutation_calls],
                [True, False, True],
            )
            self.assertEqual(
                [row["generation_attempt"] for row in mutation_calls],
                [1, 1, 2],
            )
            self.assertTrue(all(row["response_content"] for row in mutation_calls))
            self.assertTrue(prepared_path.with_suffix(".jsonl.manifest.json").exists())

    def test_unresolved_mutation_is_recorded_and_next_factoid_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = [
                {
                    "id": 1,
                    "question": "Test question?",
                    "context": "Fact A and Fact B are supported.",
                    "right_answer": "Fact A. Fact B.",
                    "hallucinated_answer": "Unused.",
                }
            ]
            (root / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
            config = {
                "dataset": {
                    "path": "data.json",
                    "format": "paired_answers",
                    "answer_roles": "right",
                },
                "prepared_path": "prepared.jsonl",
                "preparation_model": {
                    "backend": "static",
                    "model_id": "offline-generator",
                    "responses": [
                        '["Fact A.", "Fact B."]',
                        "Fact A paraphrase 1.\nFact A paraphrase 2.",
                        "Only one antonym.",
                        "Still one antonym.",
                        "Fact B paraphrase 1.\nFact B paraphrase 2.",
                        "Fact B negation 1.\nFact B negation 2.",
                    ],
                },
                "decomposition_format_max_attempts": 3,
                "decomposition_failure_policy": "record_and_continue",
                "mutation_temperature": 0.2,
                "mutation_format_max_attempts": 2,
                "mutation_failure_policy": "record_and_continue",
                "resume": True,
            }

            prepared_path = prepare_from_config(config, repository_root=root)
            prepared = load_prepared_responses(prepared_path)
            self.assertEqual([item.factoid_id for item in prepared[0].probe_sets], ["1:right:f1"])
            preparation = prepared[0].metadata["preparation"]
            self.assertEqual(preparation["source_factoid_count"], 2)
            self.assertEqual(preparation["prepared_factoid_count"], 1)
            self.assertEqual(preparation["unresolved_factoid_ids"], ["1:right:f0"])

            unresolved_path = prepared_path.with_suffix(
                ".jsonl.unresolved_mutations.jsonl"
            )
            unresolved = [
                json.loads(line)
                for line in unresolved_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(unresolved), 1)
            self.assertEqual(unresolved[0]["failure_phase"], "mutation_antonym")
            self.assertEqual(unresolved[0]["generation_attempts"], 2)
            manifest = json.loads(
                prepared_path.with_suffix(".jsonl.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "complete_with_unresolved_preparation")
            self.assertEqual(manifest["source_factoids"], 2)
            self.assertEqual(manifest["prepared_factoids"], 1)
            self.assertEqual(manifest["unresolved_mutations"], 1)
            self.assertEqual(manifest["unresolved_mutation_rate"], 0.5)

    def test_unresolved_decomposition_is_recorded_and_next_response_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = [
                {
                    "id": 1,
                    "question": "First?",
                    "context": "Context one.",
                    "right_answer": "Salaam Bombay",
                    "hallucinated_answer": "Unused one.",
                },
                {
                    "id": 2,
                    "question": "Second?",
                    "context": "Paris is in France.",
                    "right_answer": "Paris is in France.",
                    "hallucinated_answer": "Unused two.",
                },
            ]
            (root / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
            config = {
                "dataset": {
                    "path": "data.json",
                    "format": "paired_answers",
                    "answer_roles": "right",
                },
                "prepared_path": "prepared.jsonl",
                "preparation_model": {
                    "backend": "static",
                    "model_id": "offline-generator",
                    "responses": [
                        '["unterminated',
                        '["still unterminated',
                        '["Paris is in France."]',
                        "Paris lies in France.\nFrance contains Paris.",
                        "Paris is not in France.\nFrance excludes Paris.",
                    ],
                },
                "decomposition_format_max_attempts": 2,
                "decomposition_failure_policy": "record_and_continue",
                "mutation_temperature": 0.2,
                "mutation_format_max_attempts": 2,
                "mutation_failure_policy": "record_and_continue",
                "resume": True,
            }

            prepared_path = prepare_from_config(config, repository_root=root)
            prepared = load_prepared_responses(prepared_path)
            self.assertEqual(len(prepared), 2)
            self.assertEqual(prepared[0].probe_sets, ())
            self.assertTrue(
                prepared[0].metadata["preparation"]["decomposition_unresolved"]
            )
            self.assertEqual(len(prepared[1].probe_sets), 1)

            unresolved_path = prepared_path.with_suffix(
                ".jsonl.unresolved_decompositions.jsonl"
            )
            unresolved = [
                json.loads(line)
                for line in unresolved_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(unresolved), 1)
            self.assertEqual(unresolved[0]["response_id"], "1:right")
            self.assertEqual(unresolved[0]["answer_character_count"], 13)
            manifest = json.loads(
                prepared_path.with_suffix(".jsonl.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "complete_with_unresolved_preparation")
            self.assertEqual(manifest["unresolved_decompositions"], 1)
            self.assertEqual(manifest["unresolved_decomposition_rate"], 0.5)

    def test_offline_end_to_end_run_writes_authoritative_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "method_spec.md").write_text("test method", encoding="utf-8")
            probes = make_probe_set(
                "r1:f0",
                "Paris is in France.",
                ("Paris lies in France.", "France contains Paris."),
                ("Paris is not in France.", "France excludes Paris."),
            )
            prepared = PreparedResponse(
                response_id="r1",
                question_id="q1",
                answer_pair_id="q1",
                answer_role="right",
                question="Where is Paris?",
                context="Paris is in France.",
                answer="Paris is in France.",
                gold_target=BinaryTarget.SUPPORTED,
                probe_sets=(probes,),
            )
            prepared_path = root / "prepared.jsonl"
            save_prepared_responses(prepared_path, [prepared])
            config = {
                "prepared_path": "prepared.jsonl",
                "output_root": "results",
                "run_id": "offline-test",
                "routing_policy": "no_escalation",
                "threshold": 0.5,
                "scoring_profile": "paper_five_statement_mean",
                "response_aggregation": "max_factoid_score",
                "stage1": {
                    "backend": "static",
                    "model_id": "offline",
                    "responses": ["YES", "YES", "YES", "NO", "NO"],
                },
                "stage2": None,
            }
            run_directory = run_from_config(config, repository_root=root)
            self.assertTrue((run_directory / "manifest.json").exists())
            self.assertTrue((run_directory / "responses.jsonl").exists())
            self.assertEqual(
                len((run_directory / "calls.jsonl").read_text(encoding="utf-8").splitlines()),
                5,
            )
            summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stage1"]["n"], 1)
            resource = summary["resources"]["stage1_verify:static"]
            self.assertEqual(resource["policy_calls"], 5)
            self.assertEqual(resource["total_tokens"], 10)

    def test_run_excludes_zero_probe_response_and_reports_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "method_spec.md").write_text(
                "test method", encoding="utf-8"
            )
            probes = make_probe_set(
                "r2:f0",
                "Paris is in France.",
                ("Paris lies in France.", "France contains Paris."),
                ("Paris is not in France.", "France excludes Paris."),
            )
            excluded = PreparedResponse(
                response_id="r1",
                question_id="q1",
                answer_pair_id="q1",
                answer_role="right",
                question="Unresolved?",
                context="Context.",
                answer="Answer.",
                gold_target=BinaryTarget.SUPPORTED,
                probe_sets=(),
                metadata={
                    "preparation": {
                        "source_factoid_count": 1,
                        "prepared_factoid_count": 0,
                        "unresolved_mutation_count": 1,
                        "analysis_eligible": False,
                    }
                },
            )
            included = PreparedResponse(
                response_id="r2",
                question_id="q2",
                answer_pair_id="q2",
                answer_role="right",
                question="Where is Paris?",
                context="Paris is in France.",
                answer="Paris is in France.",
                gold_target=BinaryTarget.SUPPORTED,
                probe_sets=(probes,),
                metadata={
                    "preparation": {
                        "source_factoid_count": 1,
                        "prepared_factoid_count": 1,
                        "unresolved_mutation_count": 0,
                        "analysis_eligible": True,
                    }
                },
            )
            save_prepared_responses(root / "prepared.jsonl", [excluded, included])
            config = {
                "prepared_path": "prepared.jsonl",
                "output_root": "results",
                "run_id": "coverage-test",
                "routing_policy": "no_escalation",
                "threshold": 0.5,
                "scoring_profile": "paper_five_statement_mean",
                "response_aggregation": "max_factoid_score",
                "stage1": {
                    "backend": "static",
                    "model_id": "offline",
                    "responses": ["YES", "YES", "YES", "NO", "NO"],
                },
                "stage2": None,
            }

            run_directory = run_from_config(config, repository_root=root)
            summary = json.loads(
                (run_directory / "summary.json").read_text(encoding="utf-8")
            )
            coverage = summary["preparation_coverage"]
            self.assertEqual(coverage["source_responses"], 2)
            self.assertEqual(coverage["analyzed_responses"], 1)
            self.assertEqual(coverage["excluded_zero_probe_responses"], ["r1"])
            self.assertEqual(coverage["source_factoids"], 2)
            self.assertEqual(coverage["analyzed_factoids"], 1)
            self.assertEqual(coverage["unresolved_mutations"], 1)


if __name__ == "__main__":
    unittest.main()
