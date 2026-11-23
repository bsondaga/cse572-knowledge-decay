"""Task 4: Spearman rank correlation between fact popularity and forgetting.

Hypothesis: rarer (less popular) facts are forgotten more often after a model
update, so popularity should correlate *negatively* with forgetting.

This is the skeleton version: it runs end to end on synthetic dummy data.
Task 17 swaps in the real table produced by Task 7 (popularity features) and
Task 16 (post-update forgetting labels) via ``--input``.

Expected input columns (one row per fact):
    fact_id      unique identifier
    popularity   popularity signal, e.g. PopQA ``s_pop`` (Wikipedia page views)
    forgotten    1 if the fact was answered correctly before the update and
                 incorrectly after it, else 0
    acc_drop     optional, continuous pre-minus-post accuracy/confidence drop

Usage:
    python src/spearman_analysis.py                    # dummy data
    python src/spearman_analysis.py --input data/processed/facts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_COLUMNS = {"fact_id", "popularity", "forgotten"}
TARGET_COLUMNS = ("forgotten", "acc_drop")
PREDICTOR_COLUMNS = ("popularity", "o_pop", "degree")


def make_dummy_data(n_facts: int = 2000, seed: int = 572) -> pd.DataFrame:
    """Synthetic facts with a planted negative popularity/forgetting link.

    Popularity is heavy-tailed (log-normal), like PopQA page views.
    """
    rng = np.random.default_rng(seed)
    popularity = rng.lognormal(mean=6.0, sigma=2.0, size=n_facts).round().astype(int)
    log_pop = np.log1p(popularity)
    z = (log_pop - log_pop.mean()) / log_pop.std()

    # Probability of forgetting falls as popularity rises.
    p_forget = 1.0 / (1.0 + np.exp(-(-0.8 - 1.2 * z)))
    forgotten = rng.binomial(1, p_forget)
    acc_drop = np.clip(p_forget + rng.normal(0, 0.15, n_facts), -1.0, 1.0)

    return pd.DataFrame(
        {
            "fact_id": np.arange(n_facts),
            "popularity": popularity,
            "forgotten": forgotten,
            "acc_drop": acc_drop.round(4),
        }
    )


def load_facts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho and p-value of each predictor against each forgetting target.

    ``popularity`` is always tested; ``o_pop`` (object popularity) and
    ``degree`` (fact-graph degree) are tested too when present.
    """
    rows = []
    for predictor in PREDICTOR_COLUMNS:
        for target in TARGET_COLUMNS:
            if predictor not in df.columns or target not in df.columns:
                continue
            sub = df[[predictor, target]].dropna()
            rho, p = spearmanr(sub[predictor], sub[target])
            rows.append(
                {"predictor": predictor, "target": target, "n": len(sub),
                 "spearman_rho": rho, "p_value": p}
            )
    return pd.DataFrame(rows)


def spearman_by_relation(df: pd.DataFrame, min_facts: int = 20) -> pd.DataFrame:
    """The Spearman table computed separately within each relation type (``prop``).

    Relation type can confound the pooled correlation: some relations have both
    very popular objects and high forgetting (e.g. "country"), which can flip
    the sign of the overall result. Relations with fewer than ``min_facts``
    facts, or with no variation in the target, are skipped.
    """
    if "prop" not in df.columns:
        return pd.DataFrame()
    parts = []
    for prop, group in df.groupby("prop"):
        if len(group) < min_facts or group["forgotten"].nunique() < 2:
            continue
        table = spearman_table(group)
        table.insert(0, "prop", prop)
        table.insert(2, "forget_rate", group["forgotten"].mean())
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def forgetting_by_popularity_bin(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Forgetting rate per popularity quantile bin (bin 0 = rarest facts)."""
    bins = pd.qcut(df["popularity"].rank(method="first"), q=n_bins, labels=False)
    grouped = df.groupby(bins)
    out = pd.DataFrame(
        {
            "pop_min": grouped["popularity"].min(),
            "pop_max": grouped["popularity"].max(),
            "n": grouped.size(),
            "forget_rate": grouped["forgotten"].mean(),
        }
    )
    out.index.name = "pop_bin"
    return out.reset_index()


def plot_bins(bin_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_df["pop_bin"], bin_df["forget_rate"], marker="o")
    ax.set_xlabel("Popularity decile (0 = rarest)")
    ax.set_ylabel("Forgetting rate")
    ax.set_title("Forgetting rate by fact popularity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, help="CSV of facts; omit to use dummy data")
    parser.add_argument("--tag", default=None, help="suffix for output files")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.input:
        df = load_facts(args.input)
        tag = args.tag or "real"
    else:
        df = make_dummy_data()
        tag = args.tag or "dummy"

    RESULTS_DIR.mkdir(exist_ok=True)
    corr = spearman_table(df)
    bins = forgetting_by_popularity_bin(df, n_bins=args.bins)

    corr.to_csv(RESULTS_DIR / f"spearman_{tag}.csv", index=False)
    bins.to_csv(RESULTS_DIR / f"spearman_bins_{tag}.csv", index=False)
    if not args.no_plot:
        plot_bins(bins, RESULTS_DIR / f"spearman_bins_{tag}.png")
    by_rel = spearman_by_relation(df)
    if not by_rel.empty:
        by_rel.to_csv(RESULTS_DIR / f"spearman_by_relation_{tag}.csv", index=False)

    print(f"Facts: {len(df)}  (source: {args.input or 'dummy data'})\n")
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if not by_rel.empty:
        print("\nWithin each relation type (>= 20 facts):")
        print(by_rel.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nForgetting rate by popularity bin:")
    print(bins.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote results to {RESULTS_DIR}/spearman_*_{tag}.*")


if __name__ == "__main__":
    main()
