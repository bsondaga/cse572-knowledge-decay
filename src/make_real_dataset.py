from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.logistic_regression_classifier import create_stratified_split
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.logistic_regression_classifier import create_stratified_split

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATASET_PATH = PROCESSED / "real_dataset.csv"
FROZEN_PATH = ROOT / "results" / "frozen" / "labels.csv"
FROZEN_DATASET_PATH = ROOT / "results" / "frozen" / "real_dataset.csv"
SPLITS_DIR = ROOT / "results" / "splits"
SEEDS = (0, 1, 2)
POPULATIONS = ("known", "all")
# Relations with fewer rows than this in either class are merged into "other".
MIN_STRATUM = 10
LABEL_COLUMNS = ["id", "guessable", "forgotten_lr1e5", "forgotten_default"]


def graph_degree(edges: pd.DataFrame) -> pd.Series:
    return pd.concat([edges["src_id"], edges["dst_id"]]).value_counts()


def _forgotten(labels: pd.DataFrame) -> pd.Series:
    unknown = set(labels["label"]) - {"forgotten", "retained"}
    if unknown:
        raise ValueError(f"Unexpected labels: {sorted(unknown)}")
    return (labels.set_index("id")["label"] == "forgotten").astype(int)


def build_real_dataset(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    analysis: pd.DataFrame,
    labels_lr1e5: pd.DataFrame,
    labels_default: pd.DataFrame,
) -> pd.DataFrame:
    """One row per known fact; checks the analysis table against the sources."""
    a = analysis.rename(columns={"fact_id": "id"})
    lr = _forgotten(labels_lr1e5)
    df = _forgotten(labels_default)
    if set(a["id"]) != set(lr.index) or set(a["id"]) != set(df.index):
        raise ValueError("fact ids differ between the analysis table and the label files")
    if a["id"].duplicated().any():
        raise ValueError("duplicate ids in the analysis table")

    base = a[["id", "guessable", "guess_count"]].merge(
        features, on="id", how="left", validate="one_to_one"
    )
    if base["s_pop"].isna().any():
        raise ValueError("some facts have no row in popqa_features.csv")
    base["degree"] = base["id"].map(graph_degree(edges)).fillna(0).astype(int)

    checks = {
        "popularity": (a.set_index("id")["popularity"], base.set_index("id")["s_pop"]),
        "o_pop": (a.set_index("id")["o_pop"], base.set_index("id")["o_pop"]),
        "degree": (a.set_index("id")["degree"], base.set_index("id")["degree"]),
        "prop": (a.set_index("id")["prop"], base.set_index("id")["prop"]),
    }
    for name, (from_table, from_source) in checks.items():
        if not np.array_equal(from_table.sort_index().to_numpy(), from_source.sort_index().to_numpy()):
            raise ValueError(f"{name} in the analysis table differs from the source files")
    if not np.array_equal(a.set_index("id")["forgotten"].sort_index().to_numpy(), lr.sort_index().to_numpy()):
        raise ValueError("analysis table forgotten column differs from the lr1e-5 labels")

    base["forgotten_lr1e5"] = base["id"].map(lr).astype(int)
    base["forgotten_default"] = base["id"].map(df).astype(int)
    base = base.rename(columns={"prop": "relation", "s_pop": "subject_pop",
                                "s_pop_log": "subject_pop_log", "o_pop": "object_pop",
                                "o_pop_log": "object_pop_log"})
    first = ["id", "relation", "subject_pop", "subject_pop_log", "object_pop",
             "object_pop_log", "degree", "guess_count", "guessable"]
    last = ["forgotten_lr1e5", "forgotten_default"]
    rest = [c for c in base.columns if c not in first + last]
    return base[first + rest + last].sort_values("id").reset_index(drop=True)


def merge_rare_relations(
    labels: np.ndarray, relations: pd.Series | np.ndarray, min_stratum: int = MIN_STRATUM
) -> np.ndarray:
    """Replace relations with fewer than min_stratum rows in either class by "other"."""
    d = pd.DataFrame({"y": np.asarray(labels), "r": np.asarray(relations, dtype=object)})
    counts = d.groupby(["r", "y"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    rare = set(counts.index[(counts < min_stratum).any(axis=1)])
    return np.where(d["r"].isin(rare), "other", d["r"]).astype(object)


def make_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministic 70/15/15 split of df (columns id, relation, forgotten_lr1e5)."""
    labels = df["forgotten_lr1e5"].to_numpy()
    groups = merge_rare_relations(labels, df["relation"])
    split = create_stratified_split(labels, groups, seed=seed)
    names = np.empty(len(df), dtype=object)
    names[split.train] = "train"
    names[split.validation] = "val"
    names[split.test] = "test"
    return pd.DataFrame({"id": df["id"].to_numpy(), "split": names})


def load_real_dataset(path: Path = FROZEN_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_split(seed: int, population: str = "known", splits_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {POPULATIONS}")
    return pd.read_csv(splits_dir / population / f"seed{seed}.csv")


def write_outputs(dataset: pd.DataFrame, dataset_path: Path, frozen_path: Path,
                  splits_dir: Path, seeds=SEEDS, frozen_dataset_path: Path | None = None) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)
    if frozen_dataset_path is not None:
        frozen_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(frozen_dataset_path, index=False)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    dataset[LABEL_COLUMNS].to_csv(frozen_path, index=False)
    populations = {"all": dataset, "known": dataset[dataset["guessable"] == 0]}
    for name, pop in populations.items():
        out = splits_dir / name
        out.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            make_split(pop.reset_index(drop=True), seed).to_csv(out / f"seed{seed}.csv", index=False)


def summarize(dataset: pd.DataFrame, splits_dir: Path, seeds=SEEDS) -> None:
    def share(s):
        return f"{int(s.sum())} ({s.mean():.1%})"

    print(f"rows: {len(dataset)}")
    print(f"forgotten lr1e-5: {share(dataset['forgotten_lr1e5'])}")
    print(f"forgotten default: {share(dataset['forgotten_default'])}")
    print(f"guessable: {int(dataset['guessable'].sum())}")
    known = dataset[dataset["guessable"] == 0]
    print(f"known rows: {len(known)}, forgotten lr1e-5: {share(known['forgotten_lr1e5'])}, "
          f"default: {share(known['forgotten_default'])}")
    print("\nrows per relation (all / known):")
    rel = pd.DataFrame({"all": dataset["relation"].value_counts(),
                        "known": known["relation"].value_counts()}).fillna(0).astype(int)
    print(rel.sort_values("all", ascending=False).to_string())
    for name, pop in (("known", known), ("all", dataset)):
        for seed in seeds:
            s = pd.read_csv(splits_dir / name / f"seed{seed}.csv").merge(
                pop[["id", "forgotten_lr1e5"]], on="id")
            parts = ", ".join(
                f"{k}={len(g)} ({g['forgotten_lr1e5'].mean():.1%})"
                for k, g in s.groupby("split")
            )
            print(f"split {name} seed{seed}: {parts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real dataset, frozen labels and splits.")
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--dataset-out", type=Path, default=DATASET_PATH)
    parser.add_argument("--frozen-out", type=Path, default=FROZEN_PATH)
    parser.add_argument("--frozen-dataset-out", type=Path, default=FROZEN_DATASET_PATH)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()

    p = args.processed
    dataset = build_real_dataset(
        pd.read_csv(p / "popqa_features.csv"),
        pd.read_csv(p / "popqa_graph_edges.csv"),
        pd.read_csv(p / "analysis_table_lr1e-5.csv"),
        pd.read_csv(p / "forgetting_labels_lr1e-5.csv"),
        pd.read_csv(p / "forgetting_labels.csv"),
    )
    write_outputs(dataset, args.dataset_out, args.frozen_out, args.splits_dir,
                  frozen_dataset_path=args.frozen_dataset_out)
    summarize(dataset, args.splits_dir)


if __name__ == "__main__":
    main()
