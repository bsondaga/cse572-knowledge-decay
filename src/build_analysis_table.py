"""Join forgetting labels with popularity features for Tasks 17 and 18.

Inputs (all generated locally; data/processed/ is gitignored):
    data/processed/forgetting_labels.csv   Task 16, src/forgetting_labels.py
    data/processed/popqa_features.csv      Task 7, src/build_features_graph.py
    data/processed/popqa_graph_edges.csv   Task 7, src/build_features_graph.py

Output: data/processed/analysis_table.csv, one row per fact the model answered
correctly before the update (the only facts that can be forgotten), with the
columns spearman_analysis.py and kmeans_analysis.py expect:
    fact_id, popularity (= s_pop), o_pop, prop, degree, forgotten, balanced_subset

``degree`` is the number of edges the fact has in the fact graph (facts that
share a subject or an object entity).

``guessable`` flags facts the model probably got right by habit, not
knowledge: its pre-update answer is one it gave to at least
``--guess-threshold`` questions of the same relation (e.g. "football" for
most "What sport does X play?" questions). ``--drop-guessable`` leaves those
facts out, so the analysis only covers facts the model actually knew.

Usage:
    python src/build_analysis_table.py
    python src/build_analysis_table.py --drop-guessable \
        --out data/processed/analysis_table_known.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def graph_degree(edges: pd.DataFrame) -> pd.Series:
    """Edge count per fact id; edges are undirected and stored once."""
    return pd.concat([edges["src_id"], edges["dst_id"]]).value_counts().rename("degree")


def normalize_prediction(text: object) -> str:
    return " ".join(str(text if isinstance(text, str) else "").lower().split())


def guess_counts(pre_results: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """For each fact id, how many questions of its relation got the same answer.

    Counted over all pre-update predictions, not only the correct ones.
    """
    d = pre_results[["id", "prediction"]].merge(features[["id", "prop"]], on="id")
    d["pred"] = d["prediction"].map(normalize_prediction)
    counts = d.groupby(["prop", "pred"])["id"].transform("size")
    return pd.Series(counts.to_numpy(), index=d["id"], name="guess_count")


def build_table(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    edges: pd.DataFrame,
    pre_results: pd.DataFrame | None = None,
    guess_threshold: int = 10,
) -> pd.DataFrame:
    unknown = set(labels["label"]) - {"forgotten", "retained"}
    if unknown:
        raise ValueError(f"Unexpected labels: {sorted(unknown)}")

    table = labels[["id", "label", "balanced_subset"]].merge(
        features[["id", "prop", "s_pop", "o_pop"]], on="id", how="left", validate="one_to_one"
    )
    missing = table["s_pop"].isna().sum()
    if missing:
        raise ValueError(f"{missing} labelled facts have no row in popqa_features.csv")

    table["degree"] = table["id"].map(graph_degree(edges)).fillna(0).astype(int)
    table["forgotten"] = (table["label"] == "forgotten").astype(int)
    cols = ["fact_id", "popularity", "o_pop", "prop", "degree", "forgotten", "balanced_subset"]
    if pre_results is not None:
        table["guess_count"] = table["id"].map(guess_counts(pre_results, features))
        if table["guess_count"].isna().any():
            raise ValueError("Some labelled facts are missing from the pre-update results")
        table["guess_count"] = table["guess_count"].astype(int)
        table["guessable"] = (table["guess_count"] >= guess_threshold).astype(int)
        cols += ["guess_count", "guessable"]
    table = table.rename(columns={"id": "fact_id", "s_pop": "popularity"})
    return table[cols].sort_values("fact_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=PROCESSED / "forgetting_labels.csv")
    parser.add_argument("--features", type=Path, default=PROCESSED / "popqa_features.csv")
    parser.add_argument("--edges", type=Path, default=PROCESSED / "popqa_graph_edges.csv")
    parser.add_argument("--pre", type=Path, default=PROCESSED / "pre_update_results.csv")
    parser.add_argument("--guess-threshold", type=int, default=10)
    parser.add_argument("--drop-guessable", action="store_true",
                        help="leave out facts flagged as guessable")
    parser.add_argument("--out", type=Path, default=PROCESSED / "analysis_table.csv")
    args = parser.parse_args()

    table = build_table(
        pd.read_csv(args.labels),
        pd.read_csv(args.features),
        pd.read_csv(args.edges),
        pd.read_csv(args.pre),
        args.guess_threshold,
    )
    n_guess = int(table["guessable"].sum())
    print(f"Guessable facts (same answer given to >= {args.guess_threshold} "
          f"questions of the relation): {n_guess} of {len(table)}")
    if args.drop_guessable:
        table = table[table["guessable"] == 0].reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    n_forgot = int(table["forgotten"].sum())
    print(f"Saved {len(table)} facts to {args.out}")
    print(f"forgotten: {n_forgot}  retained: {len(table) - n_forgot}")


if __name__ == "__main__":
    main()
