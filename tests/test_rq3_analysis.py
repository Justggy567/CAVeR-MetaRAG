import json
import tempfile
import unittest
from pathlib import Path

from src.caver.rq3_analysis import _category_summary, run_rq3_analysis


def _row(name, *, gold, stage1, final, routed):
    return {
        "response_id": name,
        "question_id": name,
        "gold_hallucinated": gold,
        "stage1_hallucinated": stage1,
        "final_hallucinated": final,
        "routed_factoids": [name + ":f0"] if routed else [],
        "failure_family": "test",
    }


class Rq3MetricTests(unittest.TestCase):
    def test_paper_table_outcomes_and_repair_denominators(self):
        rows = [
            _row("rescued", gold=True, stage1=False, final=True, routed=True),
            _row("harmed", gold=False, stage1=False, final=True, routed=True),
            _row("missed", gold=True, stage1=False, final=False, routed=False),
            _row("success", gold=False, stage1=False, final=False, routed=False),
        ]
        summary = _category_summary(rows)
        routing = summary["routing"]
        repair = summary["repair"]

        self.assertEqual(routing["routed"], 2)
        self.assertEqual(routing["escalation_rate"], 0.5)
        self.assertEqual(routing["missed_detection"], 1)
        self.assertEqual(routing["missed_detection_rate"], 0.25)
        self.assertEqual(routing["over_escalation"], 1)
        self.assertEqual(routing["over_escalation_rate"], 0.25)
        self.assertEqual(routing["success_pass"], 1)
        self.assertEqual(routing["success_pass_rate"], 0.25)
        self.assertEqual(routing["valid_escalation"], 1)
        self.assertEqual(routing["valid_escalation_rate"], 0.25)

        self.assertEqual(repair["rescued"], 1)
        self.assertEqual(repair["rescue_rate"], 0.5)
        self.assertEqual(repair["error_conditional_rescue_rate"], 1.0)
        self.assertEqual(repair["harmed"], 1)
        self.assertEqual(repair["harm_rate"], 0.5)
        self.assertEqual(repair["escalation_efficiency_per_100"], 0.0)

    def test_non_routed_prediction_change_fails_loudly(self):
        with self.assertRaisesRegex(AssertionError, "non-routed response"):
            _category_summary(
                [_row("invalid", gold=True, stage1=False, final=True, routed=False)]
            )

    def test_analysis_uses_call_ledger_resources_in_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()

            response = {
                "response_id": "r1",
                "question_id": "q1",
                "gold_target": "UNSUPPORTED",
                "stage1_hallucinated": False,
                "final_hallucinated": True,
                "metadata": {
                    "factoid_type": "Entity / Alias / Lexical Relation",
                    "source_dataset": "TEST",
                },
            }
            factoid = {
                "response_id": "r1",
                "question_id": "q1",
                "factoid_id": "r1:f0",
                "routed": True,
                "routing_latency_seconds": 0.25,
                "signals": {
                    "original_uncertainty": False,
                    "explicit_paradox": True,
                    "cross_side_anomaly": False,
                    "two_antonym_anomalies": False,
                    "synonym_anomaly_count": 0,
                },
            }
            calls = [
                {
                    "response_id": "r1",
                    "backend": "ollama",
                    "phase": "stage1_verify",
                    "latency_seconds": 1.5,
                    "usage": {"calls": 1, "input_tokens": 10, "output_tokens": 2},
                    "provider_metadata": {
                        "total_duration_ns": 1_200_000_000,
                        "load_duration_ns": 100_000_000,
                        "prompt_eval_duration_ns": 300_000_000,
                        "eval_duration_ns": 800_000_000,
                    },
                },
                {
                    "response_id": "r1",
                    "backend": "openai_compatible",
                    "phase": "stage2_verify",
                    "model_id": "cloud-test",
                    "latency_seconds": 2.0,
                    "usage": {"calls": 1, "input_tokens": 20, "output_tokens": 3},
                },
            ]

            for name, values in (
                ("responses.jsonl", [response]),
                ("factoids.jsonl", [factoid]),
                ("calls.jsonl", calls),
            ):
                (run / name).write_text(
                    "".join(json.dumps(value) + "\n" for value in values),
                    encoding="utf-8",
                )

            output = run_rq3_analysis(
                {
                    "run_directory": "run",
                    "output_path": "analysis.json",
                    "bootstrap_repetitions": 20,
                    "bootstrap_seed": 7,
                    "include_factoid_rows": True,
                },
                repository_root=root,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            uncertainty = report["overall_cluster_bootstrap_95ci"]
            row = report["factoid_rows"][0]

            self.assertEqual(uncertainty["cloud_tokens"]["estimate"], 23.0)
            self.assertEqual(uncertainty["local_tokens"]["estimate"], 12.0)
            self.assertEqual(uncertainty["total_tokens"]["estimate"], 35.0)
            self.assertEqual(uncertainty["cloud_calls"]["estimate"], 1.0)
            self.assertEqual(uncertainty["local_calls"]["estimate"], 1.0)
            self.assertEqual(uncertainty["latency_seconds"]["estimate"], 3.75)
            self.assertEqual(uncertainty["routing_latency_seconds"]["estimate"], 0.25)
            self.assertEqual(row["phase_latency_seconds"]["routing"], 0.25)


if __name__ == "__main__":
    unittest.main()
