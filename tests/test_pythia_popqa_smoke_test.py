import unittest

from src.pythia_popqa_smoke_test import exact_match, normalize_answer


class AnswerScoringTests(unittest.TestCase):
    def test_normalizes_case_punctuation_articles_and_whitespace(self):
        self.assertEqual(normalize_answer(" The  Eiffel Tower! "), "eiffel tower")

    def test_matches_normalized_answer_as_substring(self):
        self.assertTrue(exact_match("The answer is WILLIAM Shakespeare.", ["William Shakespeare"]))

    def test_does_not_match_empty_or_unrelated_answers(self):
        self.assertFalse(exact_match("No idea.", ["", "Paris"]))

    def test_gold_answer_matches_itself(self):
        self.assertTrue(exact_match("William Shakespeare", ["William Shakespeare"]))


if __name__ == "__main__":
    unittest.main()
