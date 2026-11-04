import unittest
from src.forgetting_labels import build_forgetting_labels


def result(fact_id, correct, prediction=None):
    return {
        "id": fact_id,
        "question": f"Question {fact_id}?",
        "prediction": prediction or f"answer-{fact_id}",
        "answers": [f"answer-{fact_id}"],
        "exact_match": correct,
    }


class ForgettingLabelTests(unittest.TestCase):
    def setUp(self):
        self.pre = {
            "f1": result("f1", True),
            "f2": result("f2", True),
            "f3": result("f3", True),
            "f4": result("f4", True),
            "f5": result("f5", False),
        }
        self.post = {
            "f1": result("f1", False, "wrong"),
            "f2": result("f2", False, "wrong"),
            "f3": result("f3", True),
            "f4": result("f4", True),
            "f5": result("f5", True),
        }

    def test_labels_only_previously_known_facts_and_marks_balanced_subset(self):
        rows, summary = build_forgetting_labels(self.pre, self.post, seed=7)
        self.assertEqual([row["id"] for row in rows], ["f1", "f2", "f3", "f4"])
        self.assertEqual([row["label"] for row in rows], ["forgotten", "forgotten", "retained", "retained"])
        self.assertEqual(summary["baseline_incorrect_excluded"], 1)
        self.assertEqual(summary["class_counts"], {"forgotten": 2, "retained": 2})
        self.assertEqual(sum(row["balanced_subset"] for row in rows), 4)

    def test_balancing_is_deterministic_and_preserves_every_label(self):
        post = dict(self.post)
        post["f4"] = result("f4", True)
        pre = dict(self.pre)
        pre["f5"] = result("f5", True)
        rows, summary = build_forgetting_labels(pre, post, seed=23)
        again, _ = build_forgetting_labels(pre, post, seed=23)
        self.assertEqual(
            [row["balanced_subset"] for row in rows],
            [row["balanced_subset"] for row in again],
        )
        self.assertEqual(summary["class_counts"], {"forgotten": 2, "retained": 3})
        self.assertEqual(sum(row["balanced_subset"] for row in rows), 4)

    def test_rejects_mismatched_facts_and_single_class_data(self):
        with self.assertRaisesRegex(ValueError, "fact ids differ"):
            build_forgetting_labels(self.pre, {"f1": self.post["f1"]})
        all_retained = {fact_id: result(fact_id, True) for fact_id in self.pre}
        with self.assertRaisesRegex(ValueError, "requires at least one forgotten"):
            build_forgetting_labels(
                {fact_id: result(fact_id, True) for fact_id in self.pre}, all_retained
            )


if __name__ == "__main__":
    unittest.main()
