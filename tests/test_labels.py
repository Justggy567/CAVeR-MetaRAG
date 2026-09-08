import unittest

from src.caver.labels import Verdict, VerdictParseError, parse_verdict


class VerdictParsingTests(unittest.TestCase):
    def test_leading_labels(self):
        self.assertIs(parse_verdict("YES. Supported."), Verdict.YES)
        self.assertIs(parse_verdict("Answer: NO. Contradicted."), Verdict.NO)
        self.assertIs(parse_verdict("NOT SURE. Missing evidence."), Verdict.NOT_SURE)
        self.assertIs(parse_verdict("NOT_SURE"), Verdict.NOT_SURE)

    def test_not_sure_is_not_parsed_as_no(self):
        self.assertIs(parse_verdict("NOT SURE"), Verdict.NOT_SURE)

    def test_ambiguous_and_empty_fail_loudly(self):
        with self.assertRaises(VerdictParseError):
            parse_verdict("")
        with self.assertRaises(VerdictParseError):
            parse_verdict("The answer could be YES or NO.")


if __name__ == "__main__":
    unittest.main()
