import unittest

from src.caver.clients import StaticSequenceClient
from src.caver.generation import (
    DecompositionFormatError,
    FactoidDecomposer,
    MutationFormatError,
    MutationGenerator,
    clean_mutation_lines,
)


class MutationGenerationTests(unittest.TestCase):
    def test_decomposition_retries_invalid_json_and_audits_every_response(self):
        client = StaticSequenceClient(
            ['["unterminated', '["Salaam Bombay is a film."]']
        )
        attempts = []
        decomposer = FactoidDecomposer(client, format_max_attempts=3)

        factoids = decomposer.decompose(
            "Salaam Bombay", on_attempt=attempts.append
        )

        self.assertEqual(factoids, ["Salaam Bombay is a film."])
        self.assertEqual(
            [
                (item.generation_attempt, item.format_valid)
                for item in attempts
            ],
            [(1, False), (2, True)],
        )
        self.assertIn("JSONDecodeError", attempts[0].validation_error)

    def test_decomposition_exhaustion_raises_structured_error(self):
        client = StaticSequenceClient(['["bad', '["still bad'])
        attempts = []
        decomposer = FactoidDecomposer(client, format_max_attempts=2)

        with self.assertRaises(DecompositionFormatError) as context:
            decomposer.decompose("Salaam Bombay", on_attempt=attempts.append)

        self.assertEqual(context.exception.attempts, 2)
        self.assertIn("JSONDecodeError", context.exception.validation_error)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(not item.format_valid for item in attempts))

    def test_format_failure_retries_only_failed_side_and_keeps_audit_attempts(self):
        client = StaticSequenceClient(
            [
                "Paris lies in France.\nFrance contains Paris.",
                "Paris is not in France.",
                "Paris is outside France.\nFrance does not contain Paris.",
            ]
        )
        attempts = []
        generator = MutationGenerator(client, temperature=0.2, format_max_attempts=3)
        probes = generator.generate(
            factoid_id="r1:f0",
            question="Where is Paris?",
            factoid="Paris is in France.",
            on_attempt=attempts.append,
        )

        self.assertEqual(probes.synonyms, ("Paris lies in France.", "France contains Paris."))
        self.assertEqual(
            probes.antonyms,
            ("Paris is outside France.", "France does not contain Paris."),
        )
        self.assertEqual(
            [(item.phase, item.generation_attempt, item.format_valid) for item in attempts],
            [
                ("mutation_synonym", 1, True),
                ("mutation_antonym", 1, False),
                ("mutation_antonym", 2, True),
            ],
        )
        self.assertIn("received 1", attempts[1].validation_error)

    def test_all_malformed_attempts_fail_loudly(self):
        client = StaticSequenceClient(
            [
                "Paris lies in France.\nFrance contains Paris.",
                "Only one negation.",
                "Still only one negation.",
            ]
        )
        attempts = []
        generator = MutationGenerator(client, temperature=0.2, format_max_attempts=2)
        with self.assertRaisesRegex(MutationFormatError, "after 2 generation attempts"):
            generator.generate(
                factoid_id="r1:f0",
                question="Where is Paris?",
                factoid="Paris is in France.",
                on_attempt=attempts.append,
            )
        self.assertEqual(len(attempts), 3)
        self.assertFalse(attempts[-1].format_valid)

    def test_numbered_two_line_output_is_accepted_without_semantic_repair(self):
        self.assertEqual(
            clean_mutation_lines("1. First sentence.\n2) Second sentence."),
            ("First sentence.", "Second sentence."),
        )


if __name__ == "__main__":
    unittest.main()
