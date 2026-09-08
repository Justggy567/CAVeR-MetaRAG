from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import run_cleaning
from simple_cleaning import config
from simple_cleaning.api_client import ApiResult, DeepSeekClient, parse_boolean
from simple_cleaning.data_io import CANONICAL_FIELDS, atomic_write_json, load_json_or_jsonl
from simple_cleaning.prompts import (
    EXPECTED_PROMPT_HASHES,
    INJECT_SYSTEM_PROMPT,
    NO3_HALLUCINATION_SYSTEM_PROMPT,
    NO3_RIGHT_SYSTEM_PROMPT,
    PROMPTS,
    assert_prompts_locked,
    prompt_hashes,
)
from simple_cleaning.steps import clean_no1, clean_no2, clean_no3, inject_hallucinations


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_SOURCE = PACKAGE_ROOT / "prompts.txt"


class FakeModelClient:
    def __init__(self, outputs: List[str]) -> None:
        self.outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ApiResult:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.outputs:
            raise AssertionError("The pseudo API return value is insufficient.")
        return ApiResult(ok=True, text=self.outputs.pop(0), attempts=1)


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs

        class Message:
            content = "True"

        class Choice:
            message = Message()

        class Response:
            id = "fake-response"
            model = "deepseek-v4-flash"
            system_fingerprint = "fake-fingerprint"
            usage = None
            choices = [Choice()]

        return Response()


class FakeSdk:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions()


def canonical_asqa() -> Dict[str, str]:
    return {
        "id": "asqa-test-1",
        "question": "Who wrote the book?",
        "context": "Ada wrote the book in 2020.",
        "right_answer": "Ada wrote the book.",
        "hallucinated_answer": "",
    }


def canonical_halu() -> Dict[str, str]:
    return {
        "id": "haluevalqa-test-1",
        "question": "Who wrote the book?",
        "context": "Ada wrote the book in 2020.",
        "right_answer": "Ada wrote the book.",
        "hallucinated_answer": "Grace wrote the book.",
    }


class PromptTests(unittest.TestCase):
    def test_runtime_prompt_snapshot_matches_code_exactly(self) -> None:
        snapshot = json.loads(PROMPT_SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(
            snapshot["authoritative_source"], "simple_cleaning/prompts.py"
        )
        self.assertEqual(snapshot["prompts"], PROMPTS)
        self.assertEqual(snapshot["combined_prompt_sha256"], EXPECTED_PROMPT_HASHES)
        prompt_code = PACKAGE_ROOT / snapshot["authoritative_source"]
        self.assertEqual(
            snapshot["authoritative_source_sha256"],
            hashlib.sha256(prompt_code.read_bytes()).hexdigest(),
        )

    def test_prompt_hashes_are_locked(self) -> None:
        assert_prompts_locked()
        self.assertEqual(prompt_hashes(), EXPECTED_PROMPT_HASHES)


class StructureTests(unittest.TestCase):
    def test_exactly_four_public_cleaning_functions(self) -> None:
        tree = ast.parse(
            (PACKAGE_ROOT / "simple_cleaning" / "steps.py").read_text(
                encoding="utf-8"
            )
        )
        public = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        self.assertEqual(
            public,
            ["clean_no1", "clean_no2", "inject_hallucinations", "clean_no3"],
        )

    def test_protocol_has_no_preprocessing_stage_for_haluevalqa(self) -> None:
        protocol = run_cleaning._protocol_snapshot()
        self.assertEqual(
            protocol["route"]["haluevalqa"],
            ["NO1", "NO2_original", "NO3_original_dual_validation"],
        )
        self.assertNotIn("no0", json.dumps(protocol).lower())

    def test_boolean_parser_is_strict(self) -> None:
        self.assertIs(parse_boolean("True"), True)
        self.assertIs(parse_boolean("False."), False)
        self.assertIsNone(parse_boolean("The answer is True"))

    def test_protocol_locks_requested_model(self) -> None:
        config.validate_config()
        self.assertEqual(config.MODEL_NAME, "deepseek-v4-flash")
        self.assertEqual(config.THINKING_MODE, "disabled")
        self.assertEqual(config.BOOLEAN_MAX_TOKENS, 5)


class StageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_interval = config.REQUEST_INTERVAL_SECONDS
        self.old_checkpoint = config.CHECKPOINT_EVERY
        config.REQUEST_INTERVAL_SECONDS = 0
        config.CHECKPOINT_EVERY = 1

    def tearDown(self) -> None:
        config.REQUEST_INTERVAL_SECONDS = self.old_interval
        config.CHECKPOINT_EVERY = self.old_checkpoint

    def test_asqa_route_keeps_user_specified_injection(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            run_dir = Path(directory)
            item = canonical_asqa()

            no2_client = FakeModelClient(["True"])
            grounded, rejected, unresolved = clean_no2([item], no2_client, run_dir)
            self.assertEqual((len(grounded), len(rejected), len(unresolved)), (1, 0, 0))

            injection_client = FakeModelClient(["Grace wrote the book."])
            injected, injection_unresolved = inject_hallucinations(
                grounded, injection_client, run_dir
            )
            self.assertEqual((len(injected), len(injection_unresolved)), (1, 0))
            self.assertEqual(
                injection_client.calls[0]["messages"][0]["content"],
                INJECT_SYSTEM_PROMPT,
            )

            no3_client = FakeModelClient(["True", "True"])
            final, no3_rejected, no3_unresolved = clean_no3(
                injected, no3_client, run_dir
            )
            self.assertEqual(
                (len(final), len(no3_rejected), len(no3_unresolved)),
                (1, 0, 0),
            )
            self.assertEqual(len(no3_client.calls), 2)
            self.assertEqual(
                no3_client.calls[0]["messages"][0]["content"],
                NO3_RIGHT_SYSTEM_PROMPT,
            )
            self.assertEqual(
                no3_client.calls[1]["messages"][0]["content"],
                NO3_HALLUCINATION_SYSTEM_PROMPT,
            )

    def test_halueval_route_skips_injection(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            run_dir = Path(directory)
            item = canonical_halu()
            grounded, _, no2_unresolved = clean_no2(
                [item], FakeModelClient(["True"]), run_dir
            )
            client = FakeModelClient(["True", "True"])
            final, _, no3_unresolved = clean_no3(grounded, client, run_dir)
        self.assertEqual(len(final), 1)
        self.assertFalse(no2_unresolved)
        self.assertFalse(no3_unresolved)
        self.assertEqual(len(client.calls), 2)

    def test_no3_right_failure_short_circuits_hallucination_check(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            client = FakeModelClient(["False"])
            final, rejected, unresolved = clean_no3(
                [canonical_halu()], client, Path(directory)
            )
        self.assertEqual((len(final), len(rejected), len(unresolved)), (0, 1, 0))
        self.assertEqual(len(client.calls), 1)

    def test_successful_result_is_not_called_again_on_resume(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            run_dir = Path(directory)
            item = canonical_halu()
            clean_no2([item], FakeModelClient(["True"]), run_dir, resume=True)
            second = FakeModelClient([])
            clean, rejected, unresolved = clean_no2(
                [item], second, run_dir, resume=True
            )
        self.assertEqual((len(clean), len(rejected), len(unresolved)), (1, 0, 0))
        self.assertEqual(second.calls, [])

    def test_no1_outputs_canonical_fields(self) -> None:
        raw = [
            {
                "id": "x",
                "question": "Q",
                "docs_context": "Ada wrote a book.",
                "standard_answer": "Ada wrote the book.",
            }
        ]
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            clean, rejected = clean_no1(raw, "asqa", Path(directory))
        self.assertEqual((len(clean), len(rejected)), (1, 0))
        self.assertEqual(tuple(clean[0].keys()), CANONICAL_FIELDS)


class RouteTests(unittest.TestCase):
    def test_halueval_main_starts_directly_from_no1(self) -> None:
        raw_halu = [
            {
                "knowledge": "Ada wrote the book.",
                "question": "Who wrote the book?",
                "right_answer": "Ada",
                "hallucinated_answer": "Grace",
            }
        ]
        fake_client = FakeModelClient(["True", "True", "True"])
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            temp_root = Path(directory)
            input_path = temp_root / "haluevalqa.json"
            atomic_write_json(input_path, raw_halu)
            output_root = temp_root / "runs"
            run_tag = "halueval_original_route_test"
            with (
                patch.object(config, "DATASETS_TO_RUN", ["haluevalqa"]),
                patch.object(config, "INPUT_FILES", {"haluevalqa": input_path}),
                patch.object(config, "OUTPUT_ROOT", output_root),
                patch.object(config, "TEST_LIMIT", 1),
                patch.object(config, "RUN_TAG", run_tag),
                patch.object(config, "DRY_RUN", False),
                patch.object(config, "RESUME", False),
                patch.object(run_cleaning, "DeepSeekClient", return_value=fake_client),
            ):
                self.assertEqual(run_cleaning.main(), 0)

            run_dir = output_root / f"haluevalqa_{run_tag}"
            normalized = json.loads(
                (run_dir / "00_normalized.json").read_text(encoding="utf-8")
            )
            final = json.loads(
                (run_dir / "final_clean.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(fake_client.calls), 3)
        self.assertEqual(normalized[0]["right_answer"], "Ada")
        self.assertEqual(normalized[0]["hallucinated_answer"], "Grace")
        self.assertEqual(final, normalized)
        self.assertEqual(manifest["status"], "complete")
        self.assertNotIn("no0", json.dumps(manifest).lower())
        self.assertNotIn("injection_counts", manifest)


class ApiClientTests(unittest.TestCase):
    def test_payload_records_model_and_disables_thinking(self) -> None:
        sdk = FakeSdk()
        client = DeepSeekClient(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            thinking_mode="disabled",
            max_retries=1,
            _sdk_client=sdk,
        )
        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=5,
        )
        self.assertTrue(result.ok)
        self.assertEqual(sdk.chat.completions.kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(
            sdk.chat.completions.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


class RealInputNo1Tests(unittest.TestCase):
    """Perform read-only counting when the original data exists; skip if it does not exist after copying to another machine."""

    def test_actual_no1_counts(self) -> None:
        asqa_path = Path("fixtures/asqa_eval_data.json")
        halu_path = Path("fixtures/halueval_qa_data.json")
        if not asqa_path.exists() or not halu_path.exists():
            self.skipTest("The current machine did not provide the two raw data files.")
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as directory:
            base = Path(directory)
            asqa_clean, asqa_rejected = clean_no1(
                load_json_or_jsonl(asqa_path), "asqa", base / "asqa"
            )
            halu_clean, halu_rejected = clean_no1(
                load_json_or_jsonl(halu_path), "haluevalqa", base / "haluevalqa"
            )
        self.assertEqual((len(asqa_clean), len(asqa_rejected)), (740, 208))
        self.assertEqual((len(halu_clean), len(halu_rejected)), (9367, 633))


if __name__ == "__main__":
    unittest.main(verbosity=2)
