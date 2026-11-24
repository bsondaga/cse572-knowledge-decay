"""Compare forgetting analyses across several label sets (Tasks 14, 18 and 25).

The same analyses run on each label set, and the results are compared:
    - forgetting rate
    - Spearman rho (and p) of subject popularity, object popularity and
      graph degree vs forgotten
    - forgetting rate per subject-popularity quartile (q1 = rarest), and the
      q1 - q4 gap
    - k-means (fixed k) on log popularity, log o_pop and log degree, and how
      much the forgetting rate differs between clusters
    - optionally (--apriori, needs src/association_rules.py from Task 4): the
      number of rules X -> forgotten with lift >= 1.2, and the top rule

Uses:
    Task 14  seed stability:   seed1=... seed2=... seed3=...  -> mean and spread
    Task 18  checkpoint run:   controlled=... checkpoint=...
    Task 25  cross-model:      pythia160m=... pythia410m=... olmo=...

Each run is NAME=PATH. PATH is either a forgetting-labels CSV from
forgetting_labels.py (columns id, label) or an analysis table from
build_analysis_table.py (columns fact_id, popularity, forgotten, ...).
Labels files are joined with the popularity features and graph degree here.
With --drop-guessable (the default), facts whose answer the model gives by
habit are removed. That needs each run's pre-update results: pass
--pre NAME=PATH, otherwise data/processed/pre_update_results.csv is used.

Usage:
    python src/compare_label_sets.py --tag seeds \
        seed1=data/processed/forgetting_labels_seed1.csv \
        seed2=data/processed/forgetting_labels_seed2.csv \
        seed3=data/processed/forgetting_labels_seed3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from .build_analysis_table import PROCESSED, build_table
except ImportError:
    from build_analysis_table import PROCESSED, build_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PREDICTORS = {"popularity": "s_pop", "o_pop": "o_pop", "degree": "degree"}
SEED = 572


# ---------------------------------------------------------------- loading


def parse_named(pairs: list[str], what: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"{what} must look like NAME=PATH, got {pair!r}")
        if name in out:
            raise ValueError(f"Duplicate {what} name {name!r}")
        out[name] = Path(path)
    return out


def load_run(
    path: Path,
    features: pd.DataFrame,
    edges: pd.DataFrame,
    pre_results: pd.DataFrame | None,
    drop_guessable: bool,
) -> pd.DataFrame:
    """Analysis table for one run, from a labels file or an analysis table."""
    df = pd.read_csv(path)
    if {"fact_id", "popularity", "forgotten"} <= set(df.columns):
        table = df
    elif {"id", "label"} <= set(df.columns):
        if "balanced_subset" not in df.columns:
            df = df.assign(balanced_subset=0)
        table = build_table(df, features, edges, pre_results)
    else:
        raise ValueError(
            f"{path} is neither a forgetting-labels file (id, label) "
            "nor an analysis table (fact_id, popularity, forgotten)"
        )
    if drop_guessable:
        if "guessable" not in table.columns:
            raise ValueError(f"{path} has no 'guessable' column; pass --keep-guessable")
        table = table[table["guessable"] == 0]
    if table["forgotten"].nunique() < 2:
        raise ValueError(f"{path}: need both forgotten and retained facts")
    return table.reset_index(drop=True)


# ---------------------------------------------------------------- metrics


def spearman_metrics(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col, short in PREDICTORS.items():
        if col not in df.columns:
            continue
        rho, p = spearmanr(df[col], df["forgotten"])
        out[f"rho_{short}"] = float(rho)
        out[f"p_{short}"] = float(p)
    return out


def popularity_bin_metrics(df: pd.DataFrame) -> dict[str, float]:
    bins = pd.qcut(df["popularity"].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    rates = df.groupby(bins, observed=True)["forgotten"].mean()
    out = {f"forget_{q}": float(rates[q]) for q in rates.index}
    out["gap_q1_minus_q4"] = out["forget_q1"] - out["forget_q4"]
    return out


def kmeans_metrics(df: pd.DataFrame, k: int, seed: int = SEED) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in PREDICTORS if c in df.columns]
    x = StandardScaler().fit_transform(np.log1p(df[cols].clip(lower=0).astype(float)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    rates = df.groupby(labels)["forgotten"].mean()
    return {
        "kmeans_k": k,
        "kmeans_silhouette": float(silhouette_score(x, labels)),
        "kmeans_forget_min": float(rates.min()),
        "kmeans_forget_max": float(rates.max()),
        "kmeans_forget_spread": float(rates.max() - rates.min()),
    }


def apriori_metrics(df: pd.DataFrame, min_support: float, min_confidence: float) -> dict:
    try:
        from .association_rules import mine_rules, to_items
    except ImportError:
        try:
            from association_rules import mine_rules, to_items
        except ImportError as error:
            raise RuntimeError(
                "--apriori needs src/association_rules.py (Task 4) on this branch"
            ) from error
    rules = mine_rules(to_items(df), min_support, min_confidence)
    strong = rules[rules["lift"] >= 1.2]
    top = rules.iloc[0] if len(rules) else None
    return {
        "apriori_rules": len(rules),
        "apriori_rules_lift_1.2": len(strong),
        "apriori_top_rule": top["antecedents"] if top is not None else "",
        "apriori_top_lift": float(top["lift"]) if top is not None else float("nan"),
    }


def analyse_run(name: str, df: pd.DataFrame, k: int, apriori: bool,
                min_support: float, min_confidence: float) -> dict:
    row = {"run": name, "n_facts": len(df), "forget_rate": float(df["forgotten"].mean())}
    row.update(spearman_metrics(df))
    row.update(popularity_bin_metrics(df))
    row.update(kmeans_metrics(df, k))
    if apriori:
        row.update(apriori_metrics(df, min_support, min_confidence))
    return row


# ---------------------------------------------------------------- summary


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD, min and max of every numeric metric across runs."""
    numeric = runs.select_dtypes("number")
    return pd.DataFrame({
        "mean": numeric.mean(), "sd": numeric.std(ddof=1),
        "min": numeric.min(), "max": numeric.max(),
    }).rename_axis("metric").reset_index()


def verdict(runs: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    n = len(runs)
    pop_null = int((runs["p_s_pop"] >= alpha).sum())
    lines = [
        f"- Subject popularity is NOT a significant predictor (p >= {alpha}) in "
        f"{pop_null} of {n} runs (rho from {runs['rho_s_pop'].min():+.2f} "
        f"to {runs['rho_s_pop'].max():+.2f}).",
    ]
    if "p_degree" in runs:
        deg = runs[(runs["p_degree"] < alpha) & (runs["rho_degree"] > 0)]
        lines.append(
            f"- Graph degree is a significant positive predictor in {len(deg)} of {n} runs "
            f"(rho from {runs['rho_degree'].min():+.2f} to {runs['rho_degree'].max():+.2f})."
        )
    if pop_null == n:
        lines.append("- \"Popularity doesn't predict forgetting\" holds in every run.")
    elif pop_null == 0:
        lines.append("- \"Popularity doesn't predict forgetting\" fails in every run.")
    else:
        lines.append("- \"Popularity doesn't predict forgetting\" is NOT stable across runs.")
    return lines


def summary_markdown(runs: pd.DataFrame, summary: pd.DataFrame, title: str) -> str:
    show = ["run", "n_facts", "forget_rate", "rho_s_pop", "p_s_pop", "rho_o_pop", "p_o_pop",
            "rho_degree", "p_degree", "gap_q1_minus_q4", "kmeans_forget_spread"]
    show += [c for c in ("apriori_rules_lift_1.2", "apriori_top_rule") if c in runs]
    show = [c for c in show if c in runs]

    def cell(v):
        return f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)

    lines = [f"# {title}", "", "## Per run", "",
             "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    lines += ["| " + " | ".join(cell(r[c]) for c in show) + " |" for _, r in runs.iterrows()]
    lines += ["", "## Across runs", "", "| metric | mean | sd | min | max |", "|---|---|---|---|---|"]
    keep = [c for c in show if c not in ("run", "apriori_top_rule")]
    for _, r in summary[summary["metric"].isin(keep)].iterrows():
        lines.append(f"| {r['metric']} | {r['mean']:.3g} | {r['sd']:.3g} | "
                     f"{r['min']:.3g} | {r['max']:.3g} |")
    lines += ["", "## Verdict", "", *verdict(runs)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="NAME=PATH, one per label set")
    parser.add_argument("--pre", action="append", default=[],
                        help="NAME=PATH pre-update results for a run (for guessable facts)")
    parser.add_argument("--features", type=Path, default=PROCESSED / "popqa_features.csv")
    parser.add_argument("--edges", type=Path, default=PROCESSED / "popqa_graph_edges.csv")
    parser.add_argument("--default-pre", type=Path, default=PROCESSED / "pre_update_results.csv")
    parser.add_argument("--keep-guessable", action="store_true",
                        help="keep facts the model answers by habit (default: drop them)")
    parser.add_argument("--k", type=int, default=3, help="k-means clusters (fixed across runs)")
    parser.add_argument("--apriori", action="store_true", help="also run Task 4 Apriori")
    parser.add_argument("--min-support", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--tag", required=True, help="output name, e.g. seeds or models")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    runs = parse_named(args.runs, "run")
    pres = parse_named(args.pre, "--pre")
    unknown = set(pres) - set(runs)
    if unknown:
        raise SystemExit(f"--pre names not among runs: {sorted(unknown)}")

    features = pd.read_csv(args.features)
    edges = pd.read_csv(args.edges)
    pre_cache: dict[Path, pd.DataFrame] = {}
    rows = []
    for name, path in runs.items():
        pre_path = pres.get(name, args.default_pre)
        if pre_path not in pre_cache:
            pre_cache[pre_path] = pd.read_csv(pre_path)
        table = load_run(path, features, edges, pre_cache[pre_path], not args.keep_guessable)
        rows.append(analyse_run(name, table, args.k, args.apriori,
                                args.min_support, args.min_confidence))
        print(f"{name}: {len(table)} facts, forget rate {table['forgotten'].mean():.3f}")

    result = pd.DataFrame(rows)
    summary = summarise(result)
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"compare_{args.tag}"
    result.to_csv(f"{stem}_runs.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)
    md = summary_markdown(result, summary, args.title or f"Comparison across runs: {args.tag}")
    Path(f"{stem}.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {stem}_runs.csv, {stem}_summary.csv, {stem}.md")


if __name__ == "__main__":
    main()
