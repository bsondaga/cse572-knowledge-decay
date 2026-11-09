"""Task 9: k-means clustering of facts by popularity and forgetting behaviour.

Goal: find natural groups of facts (e.g. "rare and forgotten", "popular and
stable") and describe each group, to see whether forgetting concentrates in
particular regions of the popularity space.

This is the skeleton version: it runs end to end on synthetic dummy data.
Task 18 swaps in the real table produced by Task 7 (popularity features) and
Task 16 (post-update forgetting labels) via ``--input``.

Expected input columns (one row per fact; same table as spearman_analysis.py):
    fact_id      unique identifier
    popularity   subject popularity, PopQA ``s_pop``
    forgotten    1 if correct before the update and wrong after, else 0
  optional (used if present):
    o_pop        object popularity, PopQA ``o_pop``
    acc_drop     continuous pre-minus-post accuracy/confidence drop
    prop         relation type, e.g. "occupation" (used to describe clusters only)

Popularity columns are log-transformed and every feature is standardised
before clustering. k is chosen by silhouette score unless ``--k`` is given.

Usage:
    python src/kmeans_analysis.py                    # dummy data
    python src/kmeans_analysis.py --input data/processed/facts.csv
    python src/kmeans_analysis.py --k 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_COLUMNS = {"fact_id", "popularity", "forgotten"}
POPULARITY_COLUMNS = ("popularity", "o_pop", "degree")  # log-transformed (heavy-tailed)
OPTIONAL_FEATURES = ("o_pop", "acc_drop")
DUMMY_RELATIONS = ("occupation", "place of birth", "genre", "country", "capital", "author")
SEED = 572


def make_dummy_data(n_facts: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic facts where rare subjects are forgotten more often."""
    rng = np.random.default_rng(seed)
    s_pop = rng.lognormal(mean=6.0, sigma=2.0, size=n_facts).round().astype(int)
    o_pop = rng.lognormal(mean=7.0, sigma=1.8, size=n_facts).round().astype(int)
    z = np.log1p(s_pop)
    z = (z - z.mean()) / z.std()

    p_forget = 1.0 / (1.0 + np.exp(-(-0.8 - 1.2 * z)))
    forgotten = rng.binomial(1, p_forget)
    acc_drop = np.clip(p_forget + rng.normal(0, 0.15, n_facts), -1.0, 1.0)

    return pd.DataFrame(
        {
            "fact_id": np.arange(n_facts),
            "popularity": s_pop,
            "o_pop": o_pop,
            "forgotten": forgotten,
            "acc_drop": acc_drop.round(4),
            "prop": rng.choice(DUMMY_RELATIONS, size=n_facts),
        }
    )


def load_facts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def build_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Numeric feature matrix (before scaling); rows with gaps are dropped.

    ``columns`` picks the input columns to cluster on. Leave ``forgotten`` out
    to cluster on fact properties only and then compare forgetting rates
    between clusters (as in the real run, Task 18).
    """
    if columns is None:
        columns = ["popularity", "forgotten"] + [c for c in OPTIONAL_FEATURES if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not in input: {missing}")
    feats = df[columns].copy()
    for col in POPULARITY_COLUMNS:
        if col in feats:
            feats[col] = np.log1p(feats[col].clip(lower=0))
            feats = feats.rename(columns={col: f"log_{col}"})
    return feats.dropna()


def choose_k(x: np.ndarray, k_range: range) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each candidate k."""
    sample = min(len(x), 5000)  # silhouette is O(n^2); sample on big data
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        sil = silhouette_score(x, km.labels_, sample_size=sample, random_state=SEED)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def profile_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Size, median popularity and forgetting rate for each cluster."""
    d = df.loc[labels.index].assign(cluster=labels)
    g = d.groupby("cluster")
    prof = pd.DataFrame(
        {
            "n": g.size(),
            "share": g.size() / len(d),
            "median_popularity": g["popularity"].median(),
            "forget_rate": g["forgotten"].mean(),
        }
    )
    if "o_pop" in d:
        prof["median_o_pop"] = g["o_pop"].median()
    if "degree" in d:
        prof["median_degree"] = g["degree"].median()
    if "acc_drop" in d:
        prof["mean_acc_drop"] = g["acc_drop"].mean()
    if "prop" in d:
        prof["top_relation"] = g["prop"].agg(lambda s: s.value_counts().index[0])
    return prof.sort_values("forget_rate", ascending=False).reset_index()


def plot_results(k_scores: pd.DataFrame, x: np.ndarray, labels: np.ndarray, out_dir: Path, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(k_scores["k"], k_scores["inertia"], marker="o", label="inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1b = ax1.twinx()
    ax1b.plot(k_scores["k"], k_scores["silhouette"], marker="s", color="tab:orange")
    ax1b.set_ylabel("Silhouette", color="tab:orange")
    ax1.set_title("Choosing k")

    xy = PCA(n_components=2, random_state=SEED).fit_transform(x) if x.shape[1] > 2 else x
    sc = ax2.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=6, alpha=0.6)
    ax2.legend(*sc.legend_elements(), title="cluster", fontsize=8, loc="best")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Clusters (PCA projection)")

    fig.tight_layout()
    fig.savefig(out_dir / f"kmeans_{tag}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, help="CSV of facts; omit to use dummy data")
    parser.add_argument("--tag", default=None, help="suffix for output files")
    parser.add_argument("--k", type=int, default=None, help="fix k instead of choosing it")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument(
        "--features",
        help="comma-separated input columns to cluster on, e.g. popularity,o_pop,degree "
        "(default: popularity, forgotten, plus o_pop/acc_drop if present)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.input:
        df = load_facts(args.input)
        tag = args.tag or "real"
    else:
        df = make_dummy_data()
        tag = args.tag or "dummy"

    columns = [c.strip() for c in args.features.split(",")] if args.features else None
    feats = build_features(df, columns)
    x = StandardScaler().fit_transform(feats)

    k_scores = choose_k(x, range(args.k_min, args.k_max + 1))
    k = args.k or int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
    labels = pd.Series(km.labels_, index=feats.index, name="cluster")
    profile = profile_clusters(df, labels)

    RESULTS_DIR.mkdir(exist_ok=True)
    k_scores.to_csv(RESULTS_DIR / f"kmeans_k_scores_{tag}.csv", index=False)
    profile.to_csv(RESULTS_DIR / f"kmeans_profiles_{tag}.csv", index=False)
    df.loc[labels.index, ["fact_id"]].assign(cluster=labels).to_csv(
        RESULTS_DIR / f"kmeans_assignments_{tag}.csv", index=False
    )
    if not args.no_plot:
        plot_results(k_scores, x, km.labels_, RESULTS_DIR, tag)

    print(f"Facts: {len(feats)}  (source: {args.input or 'dummy data'})")
    print(f"Features: {', '.join(feats.columns)}\n")
    print(k_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nUsing k = {k}{' (fixed)' if args.k else ' (best silhouette)'}\n")
    print(profile.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote results to {RESULTS_DIR}/kmeans_*_{tag}.*")


if __name__ == "__main__":
    main()
