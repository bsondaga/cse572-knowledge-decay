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

Usage:
    python src/build_analysis_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def graph_degree(edges: pd.DataFrame) -> pd.Series:
    """Edge count per fact id; edges are undirected and stored once."""
    return pd.concat([edges["src_id"], edges["dst_id"]]).value_counts().rename("degree")


def build_table(labels: pd.DataFrame, features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
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
    table = table.rename(columns={"id": "fact_id", "s_pop": "popularity"})
    cols = ["fact_id", "popularity", "o_pop", "prop", "degree", "forgotten", "balanced_subset"]
    return table[cols].sort_values("fact_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=PROCESSED / "forgetting_labels.csv")
    parser.add_argument("--features", type=Path, default=PROCESSED / "popqa_features.csv")
    parser.add_argument("--edges", type=Path, default=PROCESSED / "popqa_graph_edges.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED / "analysis_table.csv")
    args = parser.parse_args()

    table = build_table(
        pd.read_csv(args.labels), pd.read_csv(args.features), pd.read_csv(args.edges)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    n_forgot = int(table["forgotten"].sum())
    print(f"Saved {len(table)} facts to {args.out}")
    print(f"forgotten: {n_forgot}  retained: {len(table) - n_forgot}")


if __name__ == "__main__":
    main()
