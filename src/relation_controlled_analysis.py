"""Task 9: is the popularity effect on forgetting hidden by relation type?

Pooled over all relations, subject popularity barely correlates with
forgetting. That can happen when relation type confounds the result: some
relations (e.g. "country") have very popular objects, high graph degree AND
high forgetting, which can mask or flip the effect inside each relation
(Simpson's paradox). This script tests that in three ways:

1. Spearman rho of popularity, o_pop and degree vs forgotten, pooled and
   within each relation (relations with >= --min-facts facts), each with a
   bootstrap 95% CI. The within-relation p-values are corrected for multiple
   testing (Benjamini-Hochberg FDR, plus Bonferroni for reference).
2. Logistic regression of forgotten on standardised log popularity, log o_pop
   and log degree, with and without relation fixed effects, and bootstrap
   95% CIs for the odds ratios. If adding relation changes the popularity
   coefficient, relation was hiding (or creating) the effect.
3. One forest-plot figure of the within-relation and pooled rhos with CIs.

Input: an analysis table from build_analysis_table.py, with columns
fact_id, popularity, o_pop, prop, degree, forgotten.

Usage:
    python src/relation_controlled_analysis.py
    python src/relation_controlled_analysis.py --input data/processed/analysis_table_lr1e-5.csv \
        --tag _all_facts
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "analysis_table_lr1e-5_known.csv"
RESULTS_DIR = ROOT / "results"
REQUIRED_COLUMNS = {"fact_id", "popularity", "o_pop", "prop", "degree", "forgotten"}
PREDICTORS = ("popularity", "o_pop", "degree")
LOGIT_FEATURES = {"popularity": "log_s_pop", "o_pop": "log_o_pop", "degree": "log_degree"}
SEED = 572


# ---------------------------------------------------------------- statistics


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR), in the original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and p-value; NaN when either side is constant."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho (resampling facts)."""
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    rhos = np.array([safe_spearman(x[i], y[i])[0] for i in idx])
    rhos = rhos[~np.isnan(rhos)]
    if len(rhos) == 0:
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    lo, hi = np.quantile(rhos, [alpha, 1 - alpha])
    return float(lo), float(hi)


# ---------------------------------------------------------------- analyses


def validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if not set(df["forgotten"].unique()) <= {0, 1}:
        raise ValueError("'forgotten' must be 0 or 1")


def spearman_by_relation(
    df: pd.DataFrame, min_facts: int, n_boot: int, seed: int = SEED
) -> pd.DataFrame:
    """Pooled and within-relation Spearman with bootstrap CIs and corrected p-values."""
    rng = np.random.default_rng(seed)
    groups = [("ALL (pooled)", df)] + [
        (rel, g) for rel, g in sorted(df.groupby("prop"), key=lambda kv: -len(kv[1]))
        if len(g) >= min_facts and g["forgotten"].nunique() == 2
    ]
    rows = []
    for rel, g in groups:
        y = g["forgotten"].to_numpy()
        for pred in PREDICTORS:
            x = g[pred].to_numpy()
            rho, p = safe_spearman(x, y)
            lo, hi = bootstrap_spearman_ci(x, y, n_boot, rng)
            rows.append({"relation": rel, "predictor": pred, "n": len(g),
                         "forget_rate": y.mean(), "spearman_rho": rho,
                         "ci_low": lo, "ci_high": hi, "p_value": p})
    out = pd.DataFrame(rows)

    within = (out["relation"] != "ALL (pooled)") & out["p_value"].notna()
    out["p_fdr_bh"] = np.nan
    out["p_bonferroni"] = np.nan
    out.loc[within, "p_fdr_bh"] = benjamini_hochberg(out.loc[within, "p_value"].to_numpy())
    out.loc[within, "p_bonferroni"] = np.minimum(out.loc[within, "p_value"] * within.sum(), 1)
    out["significant_fdr_0.05"] = out["p_fdr_bh"] < 0.05
    return out


def design_matrix(df: pd.DataFrame, with_relation: bool, min_facts: int) -> pd.DataFrame:
    """Standardised log features, plus relation dummies (rare relations -> 'other')."""
    x = pd.DataFrame(index=df.index)
    for col, name in LOGIT_FEATURES.items():
        v = np.log1p(df[col].clip(lower=0).astype(float))
        sd = v.std()
        x[name] = (v - v.mean()) / sd if sd > 0 else 0.0
    if with_relation:
        counts = df["prop"].value_counts()
        rel = df["prop"].where(df["prop"].map(counts) >= min_facts, "other")
        # the largest relation is the reference level
        reference = rel.value_counts().index[0]
        dummies = pd.get_dummies(rel, prefix="rel").astype(float)
        x = x.join(dummies.drop(columns=f"rel_{reference}"))
    return x


def fit_logit(x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Unpenalised logistic regression coefficients (intercept first)."""
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(penalty=None, max_iter=5000).fit(x.to_numpy(), y)
    return np.concatenate([model.intercept_, model.coef_[0]])


def logistic_regression(
    df: pd.DataFrame, min_facts: int, n_boot: int, seed: int = SEED
) -> pd.DataFrame:
    """Odds ratios with bootstrap CIs, for models without and with relation effects."""
    rng = np.random.default_rng(seed)
    y = df["forgotten"].to_numpy()
    rows = []
    for model_name, with_rel in (("no_relation", False), ("with_relation", True)):
        x = design_matrix(df, with_rel, min_facts)
        names = ["intercept", *x.columns]
        coef = fit_logit(x, y)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(df), len(df))
            if len(np.unique(y[i])) < 2:
                continue
            boots.append(fit_logit(x.iloc[i], y[i]))
        boots = np.array(boots)
        lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
        for j, name in enumerate(names):
            rows.append({
                "model": model_name, "term": name, "coef": coef[j],
                "odds_ratio": np.exp(coef[j]),
                "or_ci_low": np.exp(lo[j]), "or_ci_high": np.exp(hi[j]),
                "ci_excludes_1": bool(lo[j] > 0 or hi[j] < 0),
                "n": len(df), "n_boot": len(boots),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output


def plot_forest(spearman: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relations = list(dict.fromkeys(spearman["relation"]))
    fig, axes = plt.subplots(1, len(PREDICTORS), figsize=(12, 0.6 * len(relations) + 1.6),
                             sharey=True)
    labels = {"popularity": "Subject popularity (s_pop)", "o_pop": "Object popularity (o_pop)",
              "degree": "Graph degree"}
    for ax, pred in zip(axes, PREDICTORS):
        sub = spearman[spearman["predictor"] == pred].set_index("relation").loc[relations]
        ypos = np.arange(len(relations))[::-1]
        pooled = np.array([r == "ALL (pooled)" for r in relations])
        sig = sub["significant_fdr_0.05"].to_numpy()
        colors = np.where(pooled, "black", np.where(sig, "tab:red", "tab:blue"))
        for y, color, (_, r) in zip(ypos, colors, sub.iterrows()):
            ax.errorbar(r["spearman_rho"], y,
                        xerr=[[r["spearman_rho"] - r["ci_low"]], [r["ci_high"] - r["spearman_rho"]]],
                        fmt="o", color=color, elinewidth=1.5, capsize=3)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(labels[pred], fontsize=10)
        ax.set_xlabel("Spearman ρ with forgotten (95% bootstrap CI)")
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r} (n={int(sub.loc[r, 'n'])})" for r in relations])
    fig.suptitle(title + "\nred = significant after FDR correction (q < 0.05); black = pooled",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_markdown(spearman: pd.DataFrame, logit: pd.DataFrame, n_facts: int) -> str:
    pooled = spearman[spearman["relation"] == "ALL (pooled)"].set_index("predictor")
    within = spearman[spearman["relation"] != "ALL (pooled)"]
    pop = logit[logit["term"] == "log_s_pop"].set_index("model")

    def fmt_or(model):
        r = pop.loc[model]
        return f"{r['odds_ratio']:.2f} [{r['or_ci_low']:.2f}, {r['or_ci_high']:.2f}]"

    lines = [
        "# Is the popularity effect hidden by relation type?",
        "",
        f"Facts: {n_facts}. Bootstrap 95% CIs; within-relation p-values corrected with "
        "Benjamini-Hochberg (FDR).",
        "",
        "## Spearman ρ with forgotten",
        "",
        "| Relation | n | Forget rate | Predictor | ρ [95% CI] | p | FDR q |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in spearman.iterrows():
        q = "" if pd.isna(r["p_fdr_bh"]) else f"{r['p_fdr_bh']:.3f}"
        lines.append(
            f"| {r['relation']} | {r['n']} | {r['forget_rate']:.0%} | {r['predictor']} | "
            f"{r['spearman_rho']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | "
            f"{r['p_value']:.3g} | {q} |"
        )
    lines += [
        "",
        "## Logistic regression: odds ratio per 1 SD of log subject popularity",
        "",
        f"- Without relation: {fmt_or('no_relation')}",
        f"- With relation fixed effects: {fmt_or('with_relation')}",
        "",
        "(OR < 1: more popular subjects are forgotten less.)",
        "",
        "## Answer",
        "",
        f"- Pooled ρ for subject popularity: {pooled.loc['popularity', 'spearman_rho']:+.2f} "
        f"[{pooled.loc['popularity', 'ci_low']:+.2f}, {pooled.loc['popularity', 'ci_high']:+.2f}].",
        f"- Within-relation tests significant after FDR correction: "
        f"{int(within['significant_fdr_0.05'].sum())} of {int(within['p_value'].notna().sum())}"
        + (": " + ", ".join(
            f"{r.relation}/{r.predictor} (ρ = {r.spearman_rho:+.2f})"
            for r in within[within["significant_fdr_0.05"]].itertuples()) if
           within["significant_fdr_0.05"].any() else "") + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--min-facts", type=int, default=20,
                        help="minimum facts for a relation to get its own test / dummy")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. _all_facts")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    validate(df)
    spearman = spearman_by_relation(df, args.min_facts, args.n_boot, args.seed)
    logit = logistic_regression(df, args.min_facts, args.n_boot, args.seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"relation_controlled{args.tag}"
    spearman.to_csv(f"{stem}_spearman.csv", index=False)
    logit.to_csv(f"{stem}_logit.csv", index=False)
    plot_forest(spearman, Path(f"{stem}.png"),
                f"Popularity vs forgetting within each relation ({len(df)} facts)")
    summary = summary_markdown(spearman, logit, len(df))
    Path(f"{stem}_summary.md").write_text(summary)

    print(summary)
    print(logit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {stem}_spearman.csv, {stem}_logit.csv, {stem}.png, {stem}_summary.md")


if __name__ == "__main__":
    main()
