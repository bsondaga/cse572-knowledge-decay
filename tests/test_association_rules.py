import unittest

import numpy as np
import pandas as pd

from src.association_rules import mine_rules, to_items, top_rules_markdown


def make_facts(n=400, seed=0):
    """Synthetic facts where 'country' facts with rare subjects are almost always forgotten."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "fact_id": np.arange(n),
            "popularity": rng.integers(1, 100_000, n),
            "o_pop": rng.integers(1, 100_000, n),
            "prop": rng.choice(["country", "capital", "genre"], n),
            "degree": rng.integers(0, 30, n),
        }
    )
    rare = df["popularity"].rank(pct=True) <= 0.25
    planted = rare & (df["prop"] == "country")
    df["forgotten"] = np.where(planted, 1, rng.binomial(1, 0.3, n))
    return df


class ItemEncodingTests(unittest.TestCase):
    def test_every_fact_gets_one_item_per_feature_and_one_outcome(self):
        items = to_items(make_facts())
        for prefix in ("pop=", "opop=", "degree=", "rel="):
            cols = [c for c in items if c.startswith(prefix)]
            self.assertTrue((items[cols].sum(axis=1) == 1).all(), prefix)
        self.assertTrue((items["forgotten"] ^ items["retained"]).all())

    def test_degree_bands(self):
        df = make_facts(4).assign(degree=[0, 2, 9, 10])
        items = to_items(df)
        self.assertEqual(
            [items.columns[items.iloc[i].to_numpy()].tolist()[2] for i in range(4)],
            ["degree=0", "degree=1-2", "degree=3-9", "degree=10+"],
        )

    def test_rejects_missing_columns_and_non_binary_labels(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            to_items(make_facts().drop(columns="degree"))
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            to_items(make_facts().assign(forgotten=2))


class RuleMiningTests(unittest.TestCase):
    def test_finds_planted_rule_with_high_lift(self):
        rules = mine_rules(to_items(make_facts()), min_support=0.03, min_confidence=0.5)
        planted = rules[rules["antecedents"] == "pop=q1 & rel=country"]
        self.assertEqual(len(planted), 1)
        self.assertAlmostEqual(planted["confidence"].iloc[0], 1.0)
        self.assertGreater(planted["lift"].iloc[0], 1.5)

    def test_rules_only_predict_forgotten_and_never_use_outcome_items(self):
        rules = mine_rules(to_items(make_facts()), min_support=0.03, min_confidence=0.3)
        self.assertTrue((rules["consequents"] == "forgotten").all())
        self.assertFalse(rules["antecedents"].str.contains("forgotten|retained").any())
        self.assertTrue(rules["lift"].is_monotonic_decreasing)

    def test_no_rules_returns_empty_table_and_markdown_says_so(self):
        rules = mine_rules(to_items(make_facts()), min_support=0.99, min_confidence=0.5)
        self.assertTrue(rules.empty)
        self.assertIn("no rules", top_rules_markdown(rules, 0.5, 400, 10))


if __name__ == "__main__":
    unittest.main()
