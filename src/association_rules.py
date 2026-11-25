"""Task 4: Apriori association rules for which facts get forgotten.

Each fact becomes a "basket" of items:
    pop=<band>     subject popularity (s_pop) quartile: q1 (rarest) .. q4
    opop=<band>    object popularity (o_pop) quartile: q1 .. q4
    degree=<band>  fact-graph degree: 0, 1-2, 3-9, 10+
    rel=<name>     relation type, e.g. rel=country
plus one outcome item, ``forgotten`` or ``retained``.

Apriori (mlxtend) finds frequent itemsets, and we keep only rules whose
consequent is exactly {forgotten}. For each rule we report support,
confidence and lift. Lift > 1 means facts with those items are forgotten more
often than the base rate.

Input: an analysis table from build_analysis_table.py, with columns
fact_id, popularity, o_pop, prop, degree, forgotten.

Usage:
    python src/association_rules.py
    python src/association_rules.py --input data/processed/analysis_table_lr1e-5.csv \
        --min-support 0.03 --min-confidence 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "analysis_table_lr1e-5_known.csv"
RESULTS_DIR = ROOT / "results"
REQUIRED_COLUMNS = {"fact_id", "popularity", "o_pop", "prop", "degree", "forgotten"}

DEGREE_BINS = [-np.inf, 0, 2, 9, np.inf]
DEGREE_LABELS = ["0", "1-2", "3-9", "10+"]
OUTCOME = "forgotten"


def quartile_band(values: pd.Series) -> pd.Series:
    """Quartile labels q1 (lowest) to q4; ties broken by rank so bins are equal-sized."""
    return pd.qcut(values.rank(method="first"), q=4, labels=["q1", "q2", "q3", "q4"]).astype(str)


def to_items(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot item table (one boolean column per item), one row per fact."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if not set(df["forgotten"].unique()) <= {0, 1}:
        raise ValueError("'forgotten' must be 0 or 1")

    bands = pd.DataFrame(
        {
            "pop": quartile_band(df["popularity"]),
            "opop": quartile_band(df["o_pop"]),
            "degree": pd.cut(df["degree"], bins=DEGREE_BINS, labels=DEGREE_LABELS).astype(str),
            "rel": df["prop"].astype(str),
        },
        index=df.index,
    )
    items = pd.get_dummies(bands, prefix_sep="=").astype(bool)
    items[OUTCOME] = df["forgotten"].astype(bool)
    items["retained"] = ~items[OUTCOME]
    return items


def mine_rules(
    items: pd.DataFrame, min_support: float, min_confidence: float, max_len: int = 3
) -> pd.DataFrame:
    """Rules X -> {forgotten}, sorted by lift (highest first)."""
    from mlxtend.frequent_patterns import apriori, association_rules

    frequent = apriori(items, min_support=min_support, use_colnames=True, max_len=max_len)
    columns = ["antecedents", "consequents", "n_items", "support", "confidence", "lift",
               "antecedent_support", "n_facts"]
    if frequent.empty:
        return pd.DataFrame(columns=columns)

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["consequents"] == frozenset({OUTCOME})]
    # an outcome item on the left would make the rule trivial
    rules = rules[rules["antecedents"].map(lambda a: not a & {OUTCOME, "retained"})]
    if rules.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(
        {
            "antecedents": rules["antecedents"].map(lambda a: " & ".join(sorted(a))),
            "consequents": OUTCOME,
            "n_items": rules["antecedents"].map(len),
            "support": rules["support"],
            "confidence": rules["confidence"],
            "lift": rules["lift"],
            "antecedent_support": rules["antecedent support"],
        }
    )
    out["n_facts"] = (out["antecedent_support"] * len(items)).round().astype(int)
    return out.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)


def top_rules_markdown(rules: pd.DataFrame, base_rate: float, n_facts: int, top: int) -> str:
    lines = [
        "# Top association rules: X → forgotten",
        "",
        f"Facts: {n_facts}. Base forgetting rate: {base_rate:.1%} "
        "(lift = confidence / base rate).",
        "",
        "| Rule (X) | Facts with X | Support | Confidence | Lift |",
        "|---|---|---|---|---|",
    ]
    for _, r in rules.head(top).iterrows():
        lines.append(
            f"| {r['antecedents']} | {r['n_facts']} | {r['support']:.3f} | "
            f"{r['confidence']:.2f} | {r['lift']:.2f} |"
        )
    if rules.empty:
        lines.append("| (no rules met the thresholds) | | | | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--min-support", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=3,
                        help="max itemset size, including the consequent")
    parser.add_argument("--top", type=int, default=10, help="rules in the summary table")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. _lr1e-5")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    items = to_items(df)
    rules = mine_rules(items, args.min_support, args.min_confidence, args.max_len)
    base_rate = float(df["forgotten"].mean())

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"apriori_rules{args.tag}.csv"
    md_path = RESULTS_DIR / f"apriori_top_rules{args.tag}.md"
    rules.to_csv(csv_path, index=False)
    md_path.write_text(top_rules_markdown(rules, base_rate, len(df), args.top))

    print(f"Facts: {len(df)}  base forgetting rate: {base_rate:.3f}  items: {items.shape[1]}")
    print(f"Rules X -> forgotten (support >= {args.min_support}, "
          f"confidence >= {args.min_confidence}): {len(rules)}\n")
    print(rules.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
