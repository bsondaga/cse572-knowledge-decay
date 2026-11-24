import unittest

import numpy as np
import pandas as pd

from src.relation_controlled_analysis import (
    benjamini_hochberg,
    bootstrap_spearman_ci,
    design_matrix,
    logistic_regression,
    spearman_by_relation,
)


def simpson_facts(n_per_rel=150, seed=0):
    """Within each relation, rarer subjects are forgotten more; but the relation
    with popular subjects has a much higher base rate, so pooled rho is positive."""
    rng = np.random.default_rng(seed)
    parts = []
    for rel, centre, base in (("capital", 2.0, -2.0), ("country", 5.0, 2.0)):
        log_pop = rng.normal(centre, 0.5, n_per_rel)
        logit = base - 2.0 * (log_pop - centre)
        parts.append(pd.DataFrame({
            "prop": rel,
            "popularity": np.round(10 ** log_pop).astype(int),
            "o_pop": rng.integers(1, 10_000, n_per_rel),
            "degree": rng.integers(0, 20, n_per_rel),
            "forgotten": rng.binomial(1, 1 / (1 + np.exp(-logit))),
        }))
    df = pd.concat(parts, ignore_index=True)
    df["fact_id"] = np.arange(len(df))
    return df


class MultipleTestingTests(unittest.TestCase):
    def test_benjamini_hochberg_matches_hand_computation(self):
        p = np.array([0.01, 0.04, 0.03, 0.20])
        # sorted: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.0533 -> 0.0533..., 0.20
        expected = np.array([0.04, 0.05333333, 0.05333333, 0.20])
        np.testing.assert_allclose(benjamini_hochberg(p), expected, rtol=1e-6)

    def test_benjamini_hochberg_is_capped_at_one_and_handles_empty(self):
        self.assertTrue((benjamini_hochberg(np.array([0.9, 0.95])) <= 1).all())
        self.assertEqual(len(benjamini_hochberg(np.array([]))), 0)


class SpearmanTests(unittest.TestCase):
    def test_bootstrap_ci_brackets_the_estimate(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=200)
        y = (x + rng.normal(size=200) > 0).astype(int)
        lo, hi = bootstrap_spearman_ci(x, y, 500, np.random.default_rng(2))
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(lo, 0)

    def test_detects_simpsons_paradox(self):
        out = spearman_by_relation(simpson_facts(), min_facts=20, n_boot=200)
        pop = out[out["predictor"] == "popularity"].set_index("relation")
        self.assertGreater(pop.loc["ALL (pooled)", "spearman_rho"], 0)
        for rel in ("capital", "country"):
            self.assertLess(pop.loc[rel, "spearman_rho"], 0)
            self.assertTrue(pop.loc[rel, "significant_fdr_0.05"])
        self.assertTrue(np.isnan(pop.loc["ALL (pooled)", "p_fdr_bh"]))

    def test_small_relations_are_skipped(self):
        df = simpson_facts()
        df.loc[:4, "prop"] = "tiny"
        out = spearman_by_relation(df, min_facts=20, n_boot=50)
        self.assertNotIn("tiny", set(out["relation"]))


class LogisticRegressionTests(unittest.TestCase):
    def test_relation_fixed_effects_reveal_negative_popularity_effect(self):
        out = logistic_regression(simpson_facts(), min_facts=20, n_boot=100)
        pop = out[out["term"] == "log_s_pop"].set_index("model")
        self.assertGreater(pop.loc["no_relation", "odds_ratio"], 1)
        self.assertLess(pop.loc["with_relation", "odds_ratio"], 1)
        self.assertTrue(pop.loc["with_relation", "ci_excludes_1"])

    def test_design_matrix_groups_rare_relations_and_drops_reference(self):
        df = simpson_facts()
        df.loc[:4, "prop"] = "tiny"
        x = design_matrix(df, with_relation=True, min_facts=20)
        rel_cols = [c for c in x if c.startswith("rel_")]
        self.assertIn("rel_other", rel_cols)
        self.assertEqual(len(rel_cols), 2)  # capital, country, other -> one dropped
        np.testing.assert_allclose(x["log_s_pop"].mean(), 0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
