"""Unit tests for the rehearsal selection strategies."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.rehearsal import (
    select_none,
    select_random,
    select_popanchor,
    select_risk_based,
    select_rarest,
    budget_to_k,
    normalize_facts,
    generate_selection,
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

    def test_rarest_selection(self):
        result = select_rarest(self.facts, k=2)
        self.assertEqual(result["id"].tolist(), [105, 101])

    def test_rarest_zero_budget(self):
        self.assertTrue(select_rarest(self.facts, 0).empty)

    def test_rarest_invalid_budget(self):
        with self.assertRaises(ValueError):
            select_rarest(self.facts, -1)
        with self.assertRaises(ValueError):
            select_rarest(self.facts, 6)

    def test_rarest_missing_popularity(self):
        with self.assertRaises(ValueError):
            select_rarest(self.facts.drop(columns="s_pop"), 2)

    def test_rarest_missing_values(self):
        facts = self.facts.copy()
        facts.loc[0, "s_pop"] = float("nan")
        with self.assertRaises(ValueError):
            select_rarest(facts, 2)

    def test_rarest_tie_breaking(self):
        facts = pd.DataFrame({
            "id": [103, 101, 102],
            "s_pop": [50, 50, 50],
        })
        result = select_rarest(facts, k=2)
        self.assertEqual(result["id"].tolist(), [101, 102])

    def test_budget_conversion(self):
        self.assertEqual(budget_to_k(929, 5), 46)
        self.assertEqual(budget_to_k(929, 10), 92)
        self.assertEqual(budget_to_k(929, 20), 185)
        self.assertEqual(budget_to_k(0, 5), 0)

    def test_invalid_percentage_budget(self):
        with self.assertRaises(ValueError):
            budget_to_k(929, 15)
        with self.assertRaises(ValueError):
            budget_to_k(-1, 5)

    def test_normalize_analysis_columns(self):
        facts = self.facts.rename(columns={
            "id": "fact_id",
            "s_pop": "popularity",
        })
        normalized = normalize_facts(facts)
        self.assertIn("id", normalized.columns)
        self.assertIn("s_pop", normalized.columns)
        self.assertIn("fact_id", facts.columns)

    def test_selection_csv_all_strategies(self):
        expected = {
            "none": [],
            "popanchor": [102],
            "rarest": [105],
            "risk_based": [103],
        }

        with TemporaryDirectory() as tmp:
            for strategy in (
                "none", "random", "popanchor",
                "rarest", "risk_based"
            ):
                path = generate_selection(
                    self.facts, strategy, 20,
                    seed=0, output_dir=Path(tmp)
                )
                result = pd.read_csv(path)

                self.assertEqual(
                    list(result.columns), ["fact_id"]
                )
                self.assertEqual(
                    len(result), 0 if strategy == "none" else 1
                )
                self.assertEqual(
                    path.name,
                    f"{strategy}_budget20_seed0.csv"
                )

                if strategy in expected:
                    self.assertEqual(
                        result["fact_id"].tolist(),
                        expected[strategy]
                    )
                else:
                    self.assertTrue(
                        result["fact_id"].isin(self.facts["id"]).all()
                    )

    def test_selection_reproducibility(self):
        with TemporaryDirectory() as tmp:
            path = generate_selection(
                self.facts, "random", 20,
                seed=0, output_dir=Path(tmp)
            )
            first = path.read_bytes()

            path = generate_selection(
                self.facts, "random", 20,
                seed=0, output_dir=Path(tmp)
            )
            self.assertEqual(first, path.read_bytes())

    def test_unknown_strategy(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generate_selection(
                    self.facts, "unknown", 20,
                    output_dir=Path(tmp)
                )

if __name__ == "__main__":
    unittest.main()
