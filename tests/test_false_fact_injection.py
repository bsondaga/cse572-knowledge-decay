
import unittest

from src.false_fact_injection import inject_false_facts


class FalseFactInjectionTests(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "id": "1",
                "subj": "Alice",
                "prop": "occupation",
                "obj": "engineer",
                "question": "What is Alice's occupation?",
            },
            {
                "id": "2",
                "subj": "Bob",
                "prop": "occupation",
                "obj": "teacher",
                "question": "What is Bob's occupation?",
            },
            {
                "id": "3",
                "subj": "Carol",
                "prop": "occupation",
                "obj": "doctor",
                "question": "What is Carol's occupation?",
            },
        ]

    def test_generates_requested_number(self):
        result = inject_false_facts(self.records, sample_size=2)
        self.assertEqual(len(result), 2)

    def test_false_answer_differs_from_original(self):
        result = inject_false_facts(self.records, sample_size=3)

        for row in result:
            self.assertNotEqual(
                row["original_answer"].casefold(),
                row["false_answer"].casefold(),
            )

    def test_preserves_original_fact_information(self):
        result = inject_false_facts(self.records, sample_size=3)
        originals = {row["id"]: row for row in self.records}

        for row in result:
            original = originals[row["id"]]
            self.assertEqual(row["subj"], original["subj"])
            self.assertEqual(row["prop"], original["prop"])
            self.assertEqual(
                row["original_answer"], original["obj"]
            )
            self.assertEqual(
                row["question"], original["question"]
            )

    def test_reproducible_with_same_seed(self):
        first = inject_false_facts(
            self.records, sample_size=2, seed=42
        )
        second = inject_false_facts(
            self.records, sample_size=2, seed=42
        )
        self.assertEqual(first, second)

    def test_rejects_invalid_sample_size(self):
        with self.assertRaises(ValueError):
            inject_false_facts(self.records, sample_size=0)

        with self.assertRaises(ValueError):
            inject_false_facts(self.records, sample_size=10)

    def test_rejects_missing_columns(self):
        with self.assertRaises(ValueError):
            inject_false_facts(
                [{"id": "1", "subj": "Alice"}],
                sample_size=1,
            )

    def test_rejects_no_alternative_answers(self):
        records = [
            {
                "id": "1",
                "subj": "Alice",
                "prop": "occupation",
                "obj": "engineer",
                "question": "What is Alice's occupation?",
            }
        ]

        with self.assertRaises(ValueError):
            inject_false_facts(records, sample_size=1)


if __name__ == "__main__":
    unittest.main()
