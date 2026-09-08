import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.caver.publication import _rq2_rows, export_publication_tables


def _confusion():
    return {
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
        "n": 2,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


class PublicationExportTests(unittest.TestCase):
    def test_exports_only_from_existing_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uncertainty = {
                name: {"estimate": 1.0, "ci_low": 0.8, "ci_high": 1.0}
                for name in ("accuracy", "precision", "recall", "f1")
            }
            rq1 = {
                "datasets": {
                    "D": {
                        "M": {
                            "confusion": _confusion(),
                            "uncertainty": uncertainty,
                            "latency_distribution_seconds": {},
                            "resource_means": {},
                            "cloud_charge": {"available": False, "total": None},
                        }
                    }
                }
            }
            policy = {
                "stage1": _confusion(),
                "final": _confusion(),
                "uncertainty": uncertainty,
                "transitions": {
                    "rescued": 1,
                    "harmed": 0,
                    "routed_stage1_wrong": 1,
                    "net_correct_gain": 1,
                    "rescue_rate_response": 0.5,
                    "error_conditional_rescue_rate": 1.0,
                    "harm_rate_response": 0.0,
                    "rescue_rate_all_responses": 0.25,
                    "harm_rate_all_responses": 0.0,
                },
                "routing": {
                    "routed_factoids": 1,
                    "total_factoids": 2,
                    "factoid_escalation_rate": 0.5,
                    "routed_responses": 2,
                    "total_responses": 4,
                    "response_escalation_rate": 0.5,
                },
                "latency_distribution_seconds": {},
                "resource_means": {},
                "cloud_charge": {"available": False, "total": None},
            }
            rq2 = {"policies": {"structure_aware": policy}}
            category = {
                "n": 2,
                "supported": 1,
                "unsupported": 1,
                "stage1": _confusion(),
                "final": _confusion(),
                "routing": {
                    "routed": 1,
                    "escalation_rate": 0.5,
                    "missed_detection": 0,
                    "missed_detection_rate": 0.0,
                    "over_escalation": 1,
                    "over_escalation_rate": 0.5,
                    "success_pass": 1,
                    "success_pass_rate": 0.5,
                    "valid_escalation": 0,
                    "valid_escalation_rate": 0.0,
                },
                "repair": {
                    "rescued": 0,
                    "rescue_rate": 0.0,
                    "harmed": 0,
                    "harm_rate": 0.0,
                    "net_correct_gain": 0,
                    "error_conditional_rescue_rate": None,
                    "escalation_efficiency_per_100": 0.0,
                },
                "failure_families": {},
            }
            rq3 = {"overall": category, "by_category": {"type": category}}
            for name, value in (("rq1.json", rq1), ("rq2.json", rq2), ("rq3.json", rq3)):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            config = {
                "output_directory": "tables",
                "rq1_report": "rq1.json",
                "rq2_reports": [{"dataset": "D", "path": "rq2.json"}],
                "rq3_report": "rq3.json",
            }
            output = export_publication_tables(config, repository_root=root)
            self.assertTrue((output / "publication_manifest.json").exists())
            with (output / "rq1_effectiveness_resources.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["model"], "M")
            self.assertEqual(rows[0]["final_tp"], "1")
            with (output / "rq2_policy_comparison.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rq2_rows = list(csv.DictReader(handle))
            self.assertEqual(rq2_rows[0]["rescue_rate_response"], "0.5")
            self.assertEqual(rq2_rows[0]["error_conditional_rescue_rate"], "1.0")
            self.assertEqual(rq2_rows[0]["harm_rate_response"], "0.0")
            with (output / "rq3_category_analysis.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rq3_rows = list(csv.DictReader(handle))
            overall = next(row for row in rq3_rows if row["category"] == "ALL")
            self.assertEqual(overall["missed_detection_rate"], "0.0")
            self.assertEqual(overall["over_escalation_rate"], "0.5")
            self.assertEqual(overall["success_pass_rate"], "0.5")
            self.assertEqual(overall["valid_escalation_rate"], "0.0")
            self.assertEqual(overall["rescue_rate"], "0.0")
            self.assertEqual(overall["harm_rate"], "0.0")

    def test_refuses_missing_source_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                export_publication_tables(
                    {
                        "output_directory": "tables",
                        "rq1_report": "missing.json",
                        "rq2_reports": [{"dataset": "D", "path": "missing2.json"}],
                        "rq3_report": "missing3.json",
                    },
                    repository_root=root,
                )

    def test_rq2_publication_exports_all_policies_in_stable_order(self):
        policy = {
            "stage1": _confusion(),
            "final": _confusion(),
            "uncertainty": {},
            "transitions": {
                "rescued": 1,
                "harmed": 0,
                "routed_stage1_wrong": 1,
                "net_correct_gain": 1,
                "rescue_rate_response": 0.5,
                "error_conditional_rescue_rate": 1.0,
                "harm_rate_response": 0.0,
                "rescue_rate_all_responses": 0.25,
                "harm_rate_all_responses": 0.0,
            },
            "routing": {
                "routed_factoids": 1,
                "total_factoids": 2,
                "factoid_escalation_rate": 0.5,
                "routed_responses": 2,
                "total_responses": 4,
                "response_escalation_rate": 0.5,
            },
            "latency_distribution_seconds": {},
            "resource_means": {},
            "cloud_charge": {"available": False, "total": None},
        }
        report = {
            "policies": {
                name: policy
                for name in (
                    "strict",
                    "paradox_only",
                    "structure_aware",
                    "structure_aware_no_uncertainty",
                    "structure_aware_no_paradox",
                    "structure_aware_no_two_antonym",
                    "structure_aware_no_cross_side",
                )
            }
        }
        rows = _rq2_rows("D", report)
        self.assertEqual(
            [row["policy"] for row in rows],
            sorted(report["policies"]),
        )

    def test_refuses_legacy_rq2_report_missing_paper_metrics(self):
        with self.assertRaisesRegex(ValueError, "Re-run the offline analyze command"):
            _rq2_rows(
                "D",
                {
                    "policies": {
                        "structure_aware": {
                            "routing": {"routed_factoids": 1},
                            "transitions": {"rescued": 1},
                        }
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
