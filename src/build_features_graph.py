from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "popqa_clean.csv"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "popqa_features.csv"
DEFAULT_EDGES = ROOT / "data" / "processed" / "popqa_graph_edges.csv"
DEFAULT_STATS = ROOT / "results" / "graph_feature_stats.md"
N_BUCKETS = 4
Z_COLUMNS = ["s_pop", "o_pop", "s_pop_log", "o_pop_log"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Add popularity buckets, o_pop_log and z-scores; return the feature table and bucket edges."""
    df = df.copy()
    if "s_pop_log" not in df:
        df["s_pop_log"] = np.log10(df["s_pop"] + 1)
    df["o_pop_log"] = np.log10(df["o_pop"].clip(lower=0) + 1)  # o_pop has a few -2 values

    df["pop_bucket"], edges = pd.qcut(
        df["s_pop"], q=N_BUCKETS, labels=False, retbins=True, duplicates="drop"
    )
    for bucket in range(int(df["pop_bucket"].max()) + 1):
        df[f"pop_bucket_{bucket}"] = (df["pop_bucket"] == bucket).astype(int)

    for col in Z_COLUMNS:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    rel_cols = [c for c in df if c.startswith("rel_")]
    bucket_cols = [c for c in df if c.startswith("pop_bucket")]
    keep = ["id", "prop", *Z_COLUMNS, "pop_bucket", *bucket_cols[1:], *[f"{c}_z" for c in Z_COLUMNS], *rel_cols]
    return df[keep], edges


def build_edges(df: pd.DataFrame, max_group: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Link facts that share a head or tail entity; cap large groups with a seeded ring."""
    rng = np.random.default_rng(seed)
    pairs: dict[tuple[int, int], str] = {}
    capped = {"head": 0, "tail": 0}
    both = 0
    for edge_type, col in (("head", "s_uri"), ("tail", "o_uri")):
        for _, ids in df.groupby(col, sort=True)["id"]:
            ids = np.sort(ids.to_numpy())
            if len(ids) < 2:
                continue
            if len(ids) > max_group:
                capped[edge_type] += 1
                ids = rng.permutation(ids)
                links = zip(ids, np.roll(ids, -1))
            else:
                links = combinations(ids, 2)
            for a, b in links:
                key = (int(min(a, b)), int(max(a, b)))
                if key in pairs:
                    both += pairs[key] != edge_type
                else:
                    pairs[key] = edge_type
    edges = pd.DataFrame(
        [(a, b, t) for (a, b), t in pairs.items()], columns=["src_id", "dst_id", "edge_type"]
    )
    edges = edges.sort_values(["src_id", "dst_id"]).reset_index(drop=True)
    return edges, {"capped": capped, "both": both}


def summarize(features, edges, edge_info, bucket_edges, args) -> str:
    """Return the Markdown stats report."""
    nodes = features["id"]
    degree = (
        pd.concat([edges["src_id"], edges["dst_id"]]).value_counts().reindex(nodes, fill_value=0)
    )
    buckets = features["pop_bucket"].value_counts().sort_index()
    by_type = edges["edge_type"].value_counts()
    capped = edge_info["capped"]
    lines = [
        "# Graph and feature stats",
        "",
        "## Settings",
        f"- input: `{args.input.name}`",
        f"- max group size (K): {args.max_group}",
        f"- seed: {args.seed}",
        f"- popularity buckets: {N_BUCKETS} (quartiles of `s_pop`)",
        "",
        "## Features",
        f"- rows: {len(features)}",
        f"- bucket edges (`s_pop`): {', '.join(f'{e:,.1f}' for e in bucket_edges)}",
        *[f"- bucket {b}: {n} facts" for b, n in buckets.items()],
        "",
        "## Graph",
        f"- nodes: {len(nodes)}",
        f"- edges: {len(edges)} (head: {by_type.get('head', 0)}, tail: {by_type.get('tail', 0)})",
        f"- isolated nodes: {int((degree == 0).sum())}",
        f"- degree min / median / max: {degree.min()} / {degree.median():g} / {degree.max()}",
        f"- capped entity groups: {sum(capped.values())} (head: {capped['head']}, tail: {capped['tail']})",
        f"- pairs sharing both a head and a tail entity (stored once as head): {edge_info['both']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build popularity features and the fact graph from data/processed/popqa_clean.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--edges-out", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--max-group", type=int, default=50, help="largest entity group linked as a full clique")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_group < 2:
        parser.error("--max-group must be at least 2")

    df = pd.read_csv(args.input)
    features, bucket_edges = build_features(df)
    edges, edge_info = build_edges(df, args.max_group, args.seed)

    for path in (args.features_out, args.edges_out, args.stats_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_out, index=False)
    edges.to_csv(args.edges_out, index=False)
    report = summarize(features, edges, edge_info, bucket_edges, args)
    args.stats_out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
