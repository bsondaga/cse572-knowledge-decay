import unittest

import numpy as np
import pandas as pd

from src.compare_label_sets import (
    analyse_run,
    parse_named,
    popularity_bin_metrics,
    summarise,
    verdict,
)


def fake_table(seed, popularity_effect=0.0, n=300):
    """Analysis table; popularity_effect < 0 makes rare facts forgotten more."""
    rng = np.random.default_rng(seed)
    pop = rng.integers(1, 100_000, n)
    z = (np.log(pop) - np.log(pop).mean()) / np.log(pop).std()
    p = 1 / (1 + np.exp(-(popularity_effect * z)))
    return pd.DataFrame({
        "fact_id": np.arange(n),
        "popularity": pop,
        "o_pop": rng.integers(1, 100_000, n),
        "degree": rng.integers(0, 30, n),
        "prop": rng.choice(["country", "capital"], n),
        "forgotten": rng.binomial(1, p),
    })


class ParsingTests(unittest.TestCase):
    def test_parse_named_pairs(self):
        self.assertEqual(
            {k: str(v) for k, v in parse_named(["a=x.csv", "b=y.csv"], "run").items()},
            {"a": "x.csv", "b": "y.csv"},
        )

    def test_parse_named_rejects_bad_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "NAME=PATH"):
            parse_named(["x.csv"], "run")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_named(["a=x.csv", "a=y.csv"], "run")


class MetricTests(unittest.TestCase):
    def test_popularity_bins_and_gap(self):
        df = pd.DataFrame({"popularity": range(8), "forgotten": [1, 1, 1, 0, 0, 0, 0, 0]})
        m = popularity_bin_metrics(df)
        self.assertEqual((m["forget_q1"], m["forget_q2"], m["forget_q4"]), (1.0, 0.5, 0.0))
        self.assertEqual(m["gap_q1_minus_q4"], 1.0)

    def test_analyse_run_has_all_metric_groups(self):
        row = analyse_run("s1", fake_table(0), k=3, apriori=False,
                          min_support=0.03, min_confidence=0.5)
        for key in ("forget_rate", "rho_s_pop", "p_degree", "gap_q1_minus_q4",
                    "kmeans_forget_spread", "kmeans_silhouette"):
            self.assertIn(key, row)
        self.assertEqual(row["kmeans_k"], 3)

    def test_strong_popularity_effect_gives_negative_rho(self):
        row = analyse_run("s", fake_table(1, popularity_effect=-2.0), 3, False, 0.03, 0.5)
        self.assertLess(row["rho_s_pop"], -0.3)
        self.assertLess(row["p_s_pop"], 0.05)
        self.assertGreater(row["gap_q1_minus_q4"], 0.3)


class SummaryTests(unittest.TestCase):
    def runs(self, effects):
        return pd.DataFrame([
            analyse_run(f"seed{i}", fake_table(i, e), 3, False, 0.03, 0.5)
            for i, e in enumerate(effects)
        ])

    def test_summary_has_mean_and_spread(self):
        summary = summarise(self.runs([0, 0, 0])).set_index("metric")
        self.assertEqual(list(summary.columns), ["mean", "sd", "min", "max"])
        self.assertTrue(summary.loc["rho_s_pop", "min"] <= summary.loc["rho_s_pop", "mean"]
                        <= summary.loc["rho_s_pop", "max"])

    @staticmethod
    def fixed_runs(p_values):
        return pd.DataFrame({
            "rho_s_pop": [0.02] * len(p_values), "p_s_pop": p_values,
            "rho_degree": [0.2] * len(p_values), "p_degree": [0.001] * len(p_values),
        })

    def test_verdict_holds_when_no_run_has_a_popularity_effect(self):
        text = "\n".join(verdict(self.fixed_runs([0.4, 0.7, 0.2])))
        self.assertIn("holds in every run", text)
        self.assertIn("positive predictor in 3 of 3", text)

    def test_verdict_flags_instability(self):
        self.assertIn("NOT stable", "\n".join(verdict(self.fixed_runs([0.4, 0.001, 0.2]))))

    def test_verdict_with_real_strong_effect_in_one_run(self):
        self.assertIn("NOT stable", "\n".join(verdict(self.runs([0, -2.5, 0]))))


if __name__ == "__main__":
    unittest.main()
