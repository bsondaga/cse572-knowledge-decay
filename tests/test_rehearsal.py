"""Unit tests for the rehearsal selection strategies."""

import unittest

import pandas as pd

from src.rehearsal import (
    select_none,
    select_random,
    select_popanchor,
    select_risk_based,
)


class RehearsalTests(unittest.TestCase):

    def setUp(self):
        self.facts = pd.DataFrame({
            "id": [101, 102, 103, 104, 105],
            "s_pop": [100, 900, 300, 700, 50],
            "forgetting_risk": [0.8, 0.1, 0.9, 0.4, 0.6],
        })

    def test_no_rehearsal(self):
        result = select_none(self.facts)
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), list(self.facts.columns))

    def test_random_selection(self):
        first = select_random(self.facts, k=3, seed=42)
        second = select_random(self.facts, k=3, seed=42)

        self.assertEqual(len(first), 3)
        self.assertEqual(first["id"].tolist(), second["id"].tolist())
        self.assertEqual(first["id"].nunique(), 3)

    def test_popanchor_selection(self):
        result = select_popanchor(self.facts, k=2)
        self.assertEqual(result["id"].tolist(), [102, 104])

    def test_risk_based_selection(self):
        result = select_risk_based(self.facts, k=2)
        self.assertEqual(result["id"].tolist(), [103, 101])

    def test_zero_budget(self):
        self.assertTrue(select_random(self.facts, 0).empty)
        self.assertTrue(select_popanchor(self.facts, 0).empty)
        self.assertTrue(select_risk_based(self.facts, 0).empty)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            select_random(self.facts, -1)

        with self.assertRaises(ValueError):
            select_popanchor(self.facts, 6)

    def test_missing_columns(self):
        with self.assertRaises(ValueError):
            select_popanchor(self.facts.drop(columns="s_pop"), 2)

        with self.assertRaises(ValueError):
            select_risk_based(
                self.facts.drop(columns="forgetting_risk"), 2
            )

    def test_custom_risk_column(self):
        facts = self.facts.rename(
            columns={"forgetting_risk": "predicted_risk"}
        )
        result = select_risk_based(
            facts, k=2, risk_column="predicted_risk"
        )
        self.assertEqual(result["id"].tolist(), [103, 101])


if __name__ == "__main__":
    unittest.main()